import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from media_router import find_media, invalidate_media_cache
from rag_answer import answer_question, invalidate_store_cache, get_last_route_product
from rag_logger import log_error, get_recent_qa, get_recent_misses, get_recent_errors

# 每产品重建锁，防止并发 rebuild 导致文件损坏
_rebuild_locks: Dict[str, threading.Lock] = {}
_rebuild_locks_guard = threading.Lock()
from rag_runtime_config import KNOWLEDGE_DIR, SHARED_ENTITY_DIRS

# 预计算共享目录名集合，避免 health/admin_products 每次调用重建 set
_SHARED_DIR_NAMES = frozenset(SHARED_ENTITY_DIRS.values())

BASE_DIR = Path(__file__).resolve().parent
ADMIN_PAGE = BASE_DIR / "admin_page.html"
CHAT_PAGE = BASE_DIR / "web" / "chat.html"

app = FastAPI(title="Medical Aesthetics RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warmup_models():
    """启动时预热嵌入模型并预构建共享知识索引，避免首次查询延迟"""
    if os.environ.get("SKIP_WARMUP"):
        return
    try:
        from rag_answer import embed_query
        embed_query("预热查询")
        print("[INFO] 嵌入模型预热完成")
    except Exception as e:
        print(f"[WARN] 模型预热失败: {e}")
    # 预构建共享知识索引（procedures/equipment/anatomy 等），
    # 避免首次跨实体查询时 30s+ 的冷启动延迟
    try:
        from rag_answer import _ensure_shared_store
        _ensure_shared_store()
        print("[INFO] 共享知识库索引预检完成")
    except Exception as e:
        print(f"[WARN] 共享知识库预构建失败: {e}")

MAX_QUESTION_LEN = 500
MAX_HISTORY_TOTAL_CHARS = 3000  # 防止历史内容过大

# 输入清理：去除 HTML 标签和控制字符，防止注入
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_input(text: str) -> str:
    """清理用户输入：去除 HTML 标签和控制字符"""
    text = _HTML_TAG_RE.sub("", text)
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


# ===== 数据模型 =====

class HistoryItem(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str = Field(..., max_length=500)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LEN)
    mode: Literal["brief", "full"] = "brief"
    history: List[HistoryItem] = Field(default_factory=list, max_length=10)
    debug: bool = False


class MediaItem(BaseModel):
    title: str
    type: str
    url: str = ""


class AskResponse(BaseModel):
    ok: bool
    answer: str
    media: List[MediaItem] = []
    latency_ms: Optional[int] = None
    debug: Optional[Dict[str, Any]] = None


class RebuildRequest(BaseModel):
    product: str = Field(..., min_length=1, max_length=50)
    timeout_sec: int = Field(default=120, ge=10, le=600)


# ===== OpenAI 兼容数据模型 =====

_MODEL_NAME = os.environ.get("OPENAI_COMPAT_MODEL", "medical-rag")


class OAIMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class OAIChatRequest(BaseModel):
    model: str = _MODEL_NAME
    messages: List[OAIMessage] = Field(..., min_length=1)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False


# ===== 问答接口 =====

# 健康检查缓存：避免频繁的文件系统探测（Kubernetes probes 等）
_health_cache: Dict[str, Any] = {}
_health_cache_ts: float = 0.0
_HEALTH_CACHE_TTL = 15.0  # 秒
_health_lock = threading.Lock()


@app.get("/health")
def health():
    global _health_cache, _health_cache_ts
    now = time.monotonic()
    # 快速路径：无锁读取（GIL 保护 dict 引用读取安全性）
    if _health_cache and (now - _health_cache_ts) < _HEALTH_CACHE_TTL:
        return _health_cache

    from rag_runtime_config import STORE_ROOT
    shared_names = _SHARED_DIR_NAMES
    products = []
    if KNOWLEDGE_DIR.exists():
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if not p.is_dir() or p.name in shared_names:
                continue
            store = STORE_ROOT / p.name
            products.append({
                "name": p.name,
                "index_exists": (store / "index.faiss").exists(),
                "docs_exists": (store / "docs.jsonl").exists(),
            })
    # 共享知识索引状态
    shared_store = STORE_ROOT / "_shared"
    shared_indexed = (shared_store / "index.faiss").exists() and (shared_store / "docs.jsonl").exists()
    all_indexed = all(p["index_exists"] and p["docs_exists"] for p in products) if products else False
    result = {
        "status": "ok" if all_indexed else "degraded",
        "knowledge_exists": KNOWLEDGE_DIR.exists(),
        "products": products,
        "shared_knowledge_indexed": shared_indexed,
    }
    with _health_lock:
        _health_cache = result
        _health_cache_ts = now
    return result


@app.get("/chat")
def chat_page():
    if not CHAT_PAGE.exists():
        raise HTTPException(status_code=404, detail="chat.html 不存在")
    return FileResponse(CHAT_PAGE, media_type="text/html")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = _sanitize_input(req.question)
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    t0 = time.monotonic()
    rw = None
    try:
        history = [{"role": h.role, "content": _sanitize_input(h.content[:1000])} for h in req.history[-6:]]
        # 防御：限制历史总字符数，避免过大 payload 占用内存
        total_chars = sum(len(h.get("content", "")) for h in history)
        if total_chars > MAX_HISTORY_TOTAL_CHARS:
            # 从最早的消息裁剪，保留最近的对话（用累加代替 O(n) pop(0)）
            cum = 0
            trim_idx = 0
            excess = total_chars - MAX_HISTORY_TOTAL_CHARS
            for i, h in enumerate(history):
                cum += len(h.get("content", ""))
                if cum >= excess:
                    trim_idx = i + 1
                    break
            history = history[trim_idx:]
        # rewrite 只做一次，传给 answer_question 复用
        from query_rewrite import rewrite_query
        rw = rewrite_query(question, history=history)
        resolved_q = rw["original"]  # 上下文补全后的问题

        answer = answer_question(question, req.mode, history=history, rewrite=rw)
        latency_ms = int((time.monotonic() - t0) * 1000)
        # 复用 answer_one 中已计算的 route/product，避免重复检测
        route, product_id = get_last_route_product()
        if not route or not product_id:
            from rag_answer import detect_route, detect_product
            product_id = product_id or detect_product(resolved_q)
            route = route or detect_route(resolved_q)
        media = [MediaItem(**m) for m in find_media(resolved_q, product_id=product_id, route=route)]
        debug = None
        if req.debug:
            debug = {
                "question": question,
                "resolved_question": resolved_q if rw["context_resolved"] else None,
                "mode": req.mode,
                "route": route,
                "product": product_id,
                "expanded_query": rw["expanded"],
                "context_resolved": rw["context_resolved"],
                "history_summary": rw.get("history_summary") or None,
                "history_pairs_count": len(rw.get("history_pairs", [])),
                "latency_ms": latency_ms,
            }
        return AskResponse(
            ok=True,
            answer=answer,
            media=media,
            latency_ms=latency_ms,
            debug=debug,
        )
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        error_meta: Dict[str, Any] = {
            "question": question[:200],
            "latency_ms": latency_ms,
            "mode": req.mode,
        }
        # 尝试捕获已解析的上下文信息
        try:
            if rw:
                cached_route, cached_product = get_last_route_product()
                error_meta["route"] = cached_route or None
                error_meta["product"] = cached_product or None
        except Exception:
            pass
        log_error("api_ask", repr(e), meta=error_meta)
        return AskResponse(
            ok=False,
            answer="接口执行异常，请稍后重试",
        )


# ===== OpenAI 兼容接口 =====

def _oai_messages_to_question_and_history(messages: List[OAIMessage]):
    """将 OpenAI messages 格式转换为 question + history"""
    # 过滤掉 system 消息，提取 user/assistant 对话
    conv = [m for m in messages if m.role in ("user", "assistant")]
    if not conv:
        return "", []
    # 最后一条消息必须是 user 角色；如果是 assistant 则向前找最后一条 user 消息
    last = conv[-1]
    if last.role != "user":
        # 找最后一条 user 消息作为问题
        user_msgs = [m for m in conv if m.role == "user"]
        if not user_msgs:
            return "", []
        question = _sanitize_input(user_msgs[-1].content[:MAX_QUESTION_LEN])
        # 历史保留最后 user 消息之前的内容
        last_user_idx = len(conv) - 1 - conv[::-1].index(user_msgs[-1])
        history = [
            {"role": m.role, "content": _sanitize_input(m.content[:1000])}
            for m in conv[:last_user_idx]
        ][-6:]
    else:
        question = _sanitize_input(last.content[:MAX_QUESTION_LEN])
        history = [
            {"role": m.role, "content": _sanitize_input(m.content[:1000])}
            for m in conv[:-1]
        ][-6:]
    return question, history


def _build_oai_response(answer: str, model: str) -> Dict[str, Any]:
    """构建 OpenAI 兼容的响应格式"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _build_oai_stream_chunk(content: str, model: str, chunk_id: str, finish: bool = False) -> str:
    """构建 SSE 格式的流式响应块"""
    import json
    if finish:
        delta = {}
        finish_reason = "stop"
    else:
        delta = {"role": "assistant", "content": content}
        finish_reason = None
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
def oai_chat_completions(req: OAIChatRequest):
    question, history = _oai_messages_to_question_and_history(req.messages)
    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

    t0 = time.monotonic()
    try:
        # 限制历史总字符数
        total_chars = sum(len(h.get("content", "")) for h in history)
        if total_chars > MAX_HISTORY_TOTAL_CHARS:
            cum = 0
            trim_idx = 0
            excess = total_chars - MAX_HISTORY_TOTAL_CHARS
            for i, h in enumerate(history):
                cum += len(h.get("content", ""))
                if cum >= excess:
                    trim_idx = i + 1
                    break
            history = history[trim_idx:]

        from query_rewrite import rewrite_query
        rw = rewrite_query(question, history=history)
        answer = answer_question(question, "brief", history=history, rewrite=rw)
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        log_error("oai_chat", repr(e), meta={"question": question[:200], "latency_ms": latency_ms})
        answer = "接口执行异常，请稍后重试"

    if req.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        def _generate():
            yield _build_oai_stream_chunk(answer, req.model, chunk_id)
            yield _build_oai_stream_chunk("", req.model, chunk_id, finish=True)
            yield "data: [DONE]\n\n"

        return StreamingResponse(_generate(), media_type="text/event-stream")

    return _build_oai_response(answer, req.model)


@app.get("/v1/models")
def oai_models():
    return {
        "object": "list",
        "data": [
            {
                "id": _MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "local",
            }
        ],
    }


# ===== 管理接口 =====

@app.get("/admin")
def admin_page():
    if not ADMIN_PAGE.exists():
        raise HTTPException(status_code=404, detail="admin_page.html 不存在")
    return FileResponse(ADMIN_PAGE)


@app.get("/admin/products")
def admin_products():
    shared_names = _SHARED_DIR_NAMES
    products = []
    if KNOWLEDGE_DIR.exists():
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if not p.is_dir() or p.name in shared_names:
                continue
            products.append({
                "product": p.name,
                "files": sorted([x.name for x in p.iterdir() if x.is_file()]),
            })
    return {"products": products}


@app.post("/admin/rebuild")
def admin_rebuild(req: RebuildRequest):
    global _health_cache
    from build_faiss import build_for_product
    product = req.product.strip()
    # 安全校验：产品名不得包含路径分隔符或特殊字符（防止路径遍历）
    if "/" in product or "\\" in product or ".." in product or not product:
        raise HTTPException(status_code=400, detail="非法产品名称")
    # 规范化路径并确认仍在 KNOWLEDGE_DIR 下（防止 symlink 逃逸）
    product_dir = (KNOWLEDGE_DIR / product).resolve()
    knowledge_root = KNOWLEDGE_DIR.resolve()
    if not str(product_dir).startswith(str(knowledge_root) + "/"):
        raise HTTPException(status_code=400, detail="非法产品名称")
    # 校验产品目录确实存在于知识库中
    if not product_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"产品 '{product}' 不存在")
    # 获取产品级锁，防止并发重建同一产品
    with _rebuild_locks_guard:
        if product not in _rebuild_locks:
            _rebuild_locks[product] = threading.Lock()
        lock = _rebuild_locks[product]
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=f"产品 '{product}' 正在重建中，请稍后重试")
    try:
        build_for_product(product)
        # 重建后清除缓存，下次请求会加载新索引
        invalidate_store_cache(product)
        with _health_lock:
            _health_cache = {}  # 索引变更后清除健康检查缓存
        # 同时清除关联数据和媒体缓存
        from relation_engine import invalidate_relations_cache
        invalidate_relations_cache()
        invalidate_media_cache(product)
        return {"ok": True, "product": product}
    except Exception as e:
        log_error("admin_rebuild", repr(e), meta={"product": product})
        raise HTTPException(status_code=500, detail="索引重建失败，请查看服务器日志")
    finally:
        lock.release()


@app.post("/admin/rebuild_shared")
def admin_rebuild_shared():
    """重建共享知识索引（procedures、equipment、anatomy 等）"""
    global _health_cache
    from build_faiss import build_shared
    try:
        build_shared()
        invalidate_store_cache("_shared")
        with _health_lock:
            _health_cache = {}  # 索引变更后清除健康检查缓存
        from relation_engine import invalidate_relations_cache
        invalidate_relations_cache()
        return {"ok": True, "store": "_shared"}
    except Exception as e:
        log_error("admin_rebuild_shared", repr(e))
        raise HTTPException(status_code=500, detail="共享索引重建失败，请查看服务器日志")


# ===== 词库管理接口 =====

def _reload_synonym_runtime():
    """刷新运行时同义词扩展表，使增删改操作立即生效于检索。"""
    try:
        from search_utils import reload_learned_synonyms
        reload_learned_synonyms()
    except Exception:
        pass


@app.get("/admin/synonyms/all")
def admin_synonyms_all():
    """返回完整词库：静态同义词 + LLM 学习到的同义词"""
    from synonym_store import get_all_synonyms_combined
    data = get_all_synonyms_combined()
    # 附加运行时状态
    try:
        from search_utils import _LEARNED_SYNONYM_DIRECT, _learned_loaded
        data["runtime_active_count"] = len(_LEARNED_SYNONYM_DIRECT)
        data["runtime_loaded"] = _learned_loaded
    except Exception:
        data["runtime_active_count"] = 0
        data["runtime_loaded"] = False
    return data


@app.post("/admin/synonyms/reload")
def admin_synonyms_reload():
    """手动刷新运行时同义词扩展表"""
    try:
        from search_utils import reload_learned_synonyms
        count = reload_learned_synonyms()
        return {"ok": True, "active_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/synonyms/export")
def admin_synonyms_export():
    """导出全部词库为 JSON（供下载）"""
    from synonym_store import get_all_synonyms_combined
    return get_all_synonyms_combined()


@app.get("/admin/synonyms/learned")
def admin_synonyms_learned():
    """返回所有 LLM 学习到的同义词映射"""
    from synonym_store import get_all_learned
    return {"items": get_all_learned()}


@app.post("/admin/synonyms/learned/approve")
def admin_synonyms_approve(original: str):
    """审核通过一条学习到的同义词"""
    from synonym_store import approve_learned
    if not original or not original.strip():
        raise HTTPException(status_code=400, detail="original 不能为空")
    ok = approve_learned(original.strip())
    if not ok:
        raise HTTPException(status_code=404, detail="未找到该同义词")
    _reload_synonym_runtime()
    return {"ok": True, "original": original.strip()}


@app.delete("/admin/synonyms/learned")
def admin_synonyms_delete(original: str):
    """删除一条学习到的同义词"""
    from synonym_store import delete_learned
    if not original or not original.strip():
        raise HTTPException(status_code=400, detail="original 不能为空")
    ok = delete_learned(original.strip())
    if not ok:
        raise HTTPException(status_code=404, detail="未找到该同义词")
    _reload_synonym_runtime()
    return {"ok": True, "deleted": original.strip()}


class SynonymAddRequest(BaseModel):
    original: str
    mapped_to: str


@app.post("/admin/synonyms/learned/add")
def admin_synonyms_add(req: SynonymAddRequest):
    """手动添加一条同义词映射"""
    from synonym_store import add_manual
    result = add_manual(req.original, req.mapped_to)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "添加失败"))
    _reload_synonym_runtime()
    return result


class SynonymEditRequest(BaseModel):
    original: str
    mapped_to: str


@app.put("/admin/synonyms/learned")
def admin_synonyms_edit(req: SynonymEditRequest):
    """编辑已有同义词的映射目标"""
    from synonym_store import update_learned
    result = update_learned(req.original, req.mapped_to)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "编辑失败"))
    _reload_synonym_runtime()
    return result


class SynonymBatchRequest(BaseModel):
    terms: List[str]


@app.post("/admin/synonyms/learned/batch-approve")
def admin_synonyms_batch_approve(req: SynonymBatchRequest):
    """批量审核通过多条同义词"""
    from synonym_store import batch_approve
    result = batch_approve(req.terms)
    _reload_synonym_runtime()
    return result


@app.post("/admin/synonyms/learned/batch-delete")
def admin_synonyms_batch_delete(req: SynonymBatchRequest):
    """批量删除多条同义词"""
    from synonym_store import batch_delete
    result = batch_delete(req.terms)
    _reload_synonym_runtime()
    return result


@app.get("/admin/logs/qa")
def admin_logs_qa(limit: int = 20):
    return {"items": get_recent_qa(limit=min(max(1, limit), 100))}


@app.get("/admin/logs/miss")
def admin_logs_miss(limit: int = 20):
    return {"items": get_recent_misses(limit=min(max(1, limit), 100))}


@app.get("/admin/logs/error")
def admin_logs_error(limit: int = 20):
    return {"items": get_recent_errors(limit=min(max(1, limit), 100))}


# ===== 知识库文件管理接口 =====

# 安全校验：产品名只允许字母数字下划线横线
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-\u4e00-\u9fff]+$")
# 允许的文件扩展名
_ALLOWED_EXTENSIONS = {".txt", ".json"}


def _validate_product_name(name: str) -> str:
    """校验并清理产品名"""
    name = name.strip()
    if not name or not _SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="非法产品名称：只允许字母、数字、下划线、横线、中文")
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="非法产品名称")
    # 路径遍历防护
    product_dir = (KNOWLEDGE_DIR / name).resolve()
    if not product_dir.is_relative_to(KNOWLEDGE_DIR.resolve()):
        raise HTTPException(status_code=400, detail="非法产品名称")
    return name


@app.get("/admin/knowledge/{product}")
def admin_knowledge_files(product: str):
    """列出某产品的知识库文件"""
    product = _validate_product_name(product)
    pdir = KNOWLEDGE_DIR / product
    if not pdir.exists():
        raise HTTPException(status_code=404, detail=f"产品 '{product}' 不存在")
    files = []
    for f in sorted(pdir.iterdir()):
        if f.is_file():
            stat = f.stat()
            files.append({
                "name": f.name,
                "size": stat.st_size,
                "modified": int(stat.st_mtime),
                "editable": f.suffix in _ALLOWED_EXTENSIONS,
            })
    return {"product": product, "files": files}


@app.get("/admin/knowledge/{product}/{filename}")
def admin_knowledge_read(product: str, filename: str):
    """读取知识库文件内容"""
    product = _validate_product_name(product)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    fpath = KNOWLEDGE_DIR / product / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    # 路径遍历二次防护
    if not fpath.resolve().is_relative_to(KNOWLEDGE_DIR.resolve()):
        raise HTTPException(status_code=400, detail="非法路径")
    try:
        content = fpath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = fpath.read_text(encoding="utf-8-sig", errors="replace")
    return {"product": product, "filename": filename, "content": content,
            "size": len(content)}


class KnowledgeWriteRequest(BaseModel):
    content: str = Field(..., min_length=0)


@app.put("/admin/knowledge/{product}/{filename}")
def admin_knowledge_write(product: str, filename: str, req: KnowledgeWriteRequest):
    """写入/更新知识库文件内容"""
    product = _validate_product_name(product)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不允许的文件类型: {suffix}")
    pdir = KNOWLEDGE_DIR / product
    pdir.mkdir(parents=True, exist_ok=True)
    fpath = pdir / filename
    # 路径遍历防护
    if not fpath.resolve().is_relative_to((KNOWLEDGE_DIR / product).resolve()):
        raise HTTPException(status_code=400, detail="非法路径")
    # 原子写入
    tmp = fpath.with_suffix(fpath.suffix + ".tmp")
    try:
        tmp.write_text(req.content, encoding="utf-8")
        os.replace(str(tmp), str(fpath))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="写入失败，请查看服务器日志")
    return {"ok": True, "product": product, "filename": filename,
            "size": len(req.content)}


@app.delete("/admin/knowledge/{product}/{filename}")
def admin_knowledge_delete(product: str, filename: str):
    """删除知识库文件"""
    product = _validate_product_name(product)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    fpath = KNOWLEDGE_DIR / product / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    if not fpath.resolve().is_relative_to(KNOWLEDGE_DIR.resolve()):
        raise HTTPException(status_code=400, detail="非法路径")
    fpath.unlink()
    return {"ok": True, "deleted": filename}


class CreateProductRequest(BaseModel):
    product: str = Field(..., min_length=1, max_length=50)


@app.post("/admin/knowledge/create_product")
def admin_create_product(req: CreateProductRequest):
    """创建新产品目录"""
    product = _validate_product_name(req.product)
    pdir = KNOWLEDGE_DIR / product
    if pdir.exists():
        raise HTTPException(status_code=409, detail=f"产品 '{product}' 已存在")
    pdir.mkdir(parents=True, exist_ok=True)
    # 创建空的 main.txt
    (pdir / "main.txt").write_text("", encoding="utf-8")
    return {"ok": True, "product": product}


@app.delete("/admin/knowledge/{product}")
def admin_delete_product(product: str):
    """删除产品目录（含所有文件）"""
    product = _validate_product_name(product)
    pdir = KNOWLEDGE_DIR / product
    if not pdir.exists():
        raise HTTPException(status_code=404, detail=f"产品 '{product}' 不存在")
    import shutil
    shutil.rmtree(pdir)
    # 清理对应的索引
    from rag_runtime_config import STORE_ROOT
    store_dir = STORE_ROOT / product
    if store_dir.exists():
        shutil.rmtree(store_dir)
    invalidate_store_cache(product)
    invalidate_media_cache(product)
    return {"ok": True, "deleted": product}


# ===== 文件上传接口 =====

@app.post("/admin/upload")
async def admin_upload(request: "Request"):
    """通用文件上传：支持上传 txt/json 文件到指定产品目录。
    Form fields: product (str), files (UploadFile[])
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="需要 multipart/form-data 格式")
    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail="解析表单失败")
    try:
        product = str(form.get("product", "")).strip()
        if not product:
            raise HTTPException(status_code=400, detail="缺少 product 字段")
        product = _validate_product_name(product)
        pdir = KNOWLEDGE_DIR / product
        pdir.mkdir(parents=True, exist_ok=True)
        uploaded = []
        errors = []
        for key in form:
            if key == "product":
                continue
            item = form[key]
            # UploadFile 对象
            if hasattr(item, "filename") and hasattr(item, "read"):
                fname = item.filename or ""
                if ".." in fname or "/" in fname or "\\" in fname:
                    errors.append({"file": fname, "error": "非法文件名"})
                    continue
                suffix = Path(fname).suffix.lower()
                if suffix not in _ALLOWED_EXTENSIONS:
                    errors.append({"file": fname, "error": f"不允许的文件类型: {suffix}"})
                    continue
                try:
                    content = await item.read()
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        text = content.decode("utf-8-sig")
                    except Exception:
                        text = content.decode("gbk", errors="replace")
                fpath = pdir / fname
                # 原子写入
                tmp = fpath.with_suffix(fpath.suffix + ".tmp")
                try:
                    tmp.write_text(text, encoding="utf-8")
                    os.replace(str(tmp), str(fpath))
                except Exception:
                    tmp.unlink(missing_ok=True)
                    raise
                uploaded.append({"file": fname, "size": len(text)})
        return {"ok": True, "product": product, "uploaded": uploaded, "errors": errors}
    finally:
        await form.close()


# ===== 批量上传：ZIP 包解压 =====

@app.post("/admin/upload_zip")
async def admin_upload_zip(request: "Request"):
    """上传 ZIP 包，自动解压到知识库。
    ZIP 内部结构：product_name/main.txt, product_name/faq.txt 等
    """
    import zipfile
    import io
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="需要 multipart/form-data 格式")
    form = await request.form()
    try:
        results = []
        for key in form:
            item = form[key]
            if not hasattr(item, "read"):
                continue
            fname = getattr(item, "filename", "") or ""
            if not fname.lower().endswith(".zip"):
                results.append({"file": fname, "error": "只支持 .zip 文件"})
                continue
            data = await item.read()
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        # 安全校验路径
                        parts = Path(info.filename).parts
                        if len(parts) < 2:
                            continue
                        product_name = parts[0]
                        file_name = parts[-1]
                        if ".." in info.filename:
                            continue
                        suffix = Path(file_name).suffix.lower()
                        if suffix not in _ALLOWED_EXTENSIONS:
                            continue
                        try:
                            product_name = _validate_product_name(product_name)
                        except Exception:
                            continue
                        pdir = KNOWLEDGE_DIR / product_name
                        pdir.mkdir(parents=True, exist_ok=True)
                        content = zf.read(info.filename)
                        try:
                            text = content.decode("utf-8")
                        except UnicodeDecodeError:
                            text = content.decode("utf-8-sig", errors="replace")
                        # 原子写入
                        dest = pdir / file_name
                        tmp = dest.with_suffix(dest.suffix + ".tmp")
                        try:
                            tmp.write_text(text, encoding="utf-8")
                            os.replace(str(tmp), str(dest))
                        except Exception:
                            tmp.unlink(missing_ok=True)
                            raise
                        results.append({"product": product_name, "file": file_name,
                                        "size": len(text)})
            except zipfile.BadZipFile:
                results.append({"file": fname, "error": "无效的 ZIP 文件"})
        return {"ok": True, "results": results}
    finally:
        await form.close()


# ===== 运行时配置接口 =====

@app.get("/admin/config")
def admin_get_config():
    """获取所有可调参数"""
    from rag_runtime_config import get_tunable_config
    return get_tunable_config()


class ConfigUpdateRequest(BaseModel):
    updates: Dict[str, Any]


@app.post("/admin/config")
def admin_update_config(req: ConfigUpdateRequest):
    """热更新运行时参数"""
    from rag_runtime_config import update_tunable_config
    changed = update_tunable_config(req.updates)
    return {"ok": True, "changed": changed}


@app.get("/admin/config/model")
def admin_get_model_config():
    """获取当前模型配置"""
    from rag_runtime_config import get_model_config
    return get_model_config()


class ModelSwitchRequest(BaseModel):
    provider: str = Field(..., min_length=1)
    model: str = ""
    api_base: str = ""
    api_key: str = ""


@app.post("/admin/config/model")
def admin_switch_model(req: ModelSwitchRequest):
    """切换模型提供商"""
    from rag_runtime_config import switch_model_provider
    result = switch_model_provider(req.provider, req.model, req.api_base, req.api_key)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ===== 多 LLM 提供商配置接口 =====

def _get_knowledge_model_name() -> str:
    """获取知识库 LLM 模型名，用于 API 响应"""
    try:
        from llm_client import get_model as _gm, is_enabled as _ie
        if _ie("knowledge"):
            m = _gm("knowledge")
            if m:
                return m
    except ImportError:
        pass
    from rag_runtime_config import OPENAI_MODEL
    return OPENAI_MODEL


@app.get("/admin/llm/configs")
def admin_get_llm_configs():
    """获取所有用途的 LLM 配置（chat / knowledge）"""
    try:
        from llm_client import get_all_llm_configs
        from rag_runtime_config import MODEL_PRESETS
        return {
            "ok": True,
            "configs": get_all_llm_configs(),
            "presets": MODEL_PRESETS,
        }
    except ImportError:
        from rag_runtime_config import get_model_config, MODEL_PRESETS
        cfg = get_model_config()
        return {
            "ok": True,
            "configs": {"chat": cfg, "knowledge": cfg},
            "presets": MODEL_PRESETS,
        }


class MultiLLMUpdateRequest(BaseModel):
    purpose: str = Field(..., pattern="^(chat|knowledge)$")
    provider: str = ""
    model: str = ""
    api_base: Optional[str] = None   # None=不更新, ""=清空
    api_key: str = ""
    enabled: Optional[bool] = None
    model_format: str = ""           # "standard" or "litellm" (provider:model_id)


@app.post("/admin/llm/configs")
def admin_update_llm_config(req: MultiLLMUpdateRequest):
    """更新指定用途的 LLM 配置"""
    from llm_client import update_llm_config
    result = update_llm_config(
        req.purpose,
        provider=req.provider,
        model=req.model,
        api_base=req.api_base,
        api_key=req.api_key,
        enabled=req.enabled,
        model_format=req.model_format,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/admin/llm/test")
def admin_llm_test_purpose(purpose: str = "chat"):
    """测试指定用途的 LLM 连接"""
    from llm_client import get_client, get_model, is_enabled, get_llm_config
    if purpose not in ("chat", "knowledge"):
        raise HTTPException(status_code=400, detail="purpose 必须为 chat 或 knowledge")

    if not is_enabled(purpose):
        return {"ok": False, "error": f"{purpose} LLM 未启用"}

    cfg = get_llm_config(purpose)
    if not cfg["api_key_set"]:
        return {"ok": False, "error": f"{purpose} LLM 未设置 API Key"}

    t0 = time.monotonic()
    try:
        client = get_client(purpose)
        if client is None:
            return {"ok": False, "error": "LLM client 创建失败"}
        model_name = get_model(purpose)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "请回复OK"}],
            max_tokens=10,
            temperature=0,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        reply = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        return {
            "ok": True,
            "purpose": purpose,
            "provider": cfg["provider"],
            "model": model_name,
            "api_base": cfg["api_base"] or "(default)",
            "reply": reply,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {"ok": False, "error": str(e), "latency_ms": latency_ms}


# ===== 服务器 / 域名配置接口 =====

@app.get("/admin/config/server")
def admin_get_server_config():
    """获取服务器和域名访问配置"""
    from rag_runtime_config import get_server_config
    return get_server_config()


class ServerConfigRequest(BaseModel):
    updates: Dict[str, Any]


@app.post("/admin/config/server")
def admin_update_server_config(req: ServerConfigRequest):
    """更新服务器和域名配置"""
    from rag_runtime_config import update_server_config
    changed = update_server_config(req.updates)
    return {"ok": True, "changed": changed}


class NginxGenRequest(BaseModel):
    domain: str = Field(..., min_length=1)
    port: int = 0
    ssl: bool = False
    cert_path: str = ""
    key_path: str = ""


@app.post("/admin/config/nginx")
def admin_generate_nginx(req: NginxGenRequest):
    """生成 nginx 反向代理配置"""
    from rag_runtime_config import generate_nginx_config
    config = generate_nginx_config(req.domain, req.port, req.ssl, req.cert_path, req.key_path)
    return {"ok": True, "config": config}


# ===== BGE-M3 嵌入模型控制接口 =====

@app.get("/admin/service/embedding")
def admin_embedding_status():
    """获取 BGE-M3 嵌入模型状态"""
    from rag_runtime_config import get_embedding_status
    return get_embedding_status()


@app.post("/admin/service/embedding/start")
def admin_embedding_start():
    """加载 BGE-M3 嵌入模型"""
    from rag_runtime_config import start_embedding_model
    result = start_embedding_model()
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "启动失败"))
    return result


@app.post("/admin/service/embedding/stop")
def admin_embedding_stop():
    """卸载 BGE-M3 嵌入模型"""
    from rag_runtime_config import stop_embedding_model
    result = stop_embedding_model()
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "停止失败"))
    return result


# ===== LLM 服务控制接口 =====

@app.get("/admin/service/llm")
def admin_llm_status():
    """获取 LLM 服务状态"""
    from rag_runtime_config import get_llm_status
    return get_llm_status()


class LLMStartRequest(BaseModel):
    api_key: str = ""


@app.post("/admin/service/llm/start")
def admin_llm_start(req: LLMStartRequest):
    """启动 LLM 服务"""
    from rag_runtime_config import start_llm_service
    result = start_llm_service(req.api_key)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "启动失败"))
    return result


@app.post("/admin/service/llm/stop")
def admin_llm_stop():
    """停止 LLM 服务"""
    from rag_runtime_config import stop_llm_service
    return stop_llm_service()


@app.post("/admin/service/llm/test")
def admin_llm_test():
    """测试 LLM API 连接（发送一次简短请求验证连通性）"""
    from rag_runtime_config import USE_OPENAI, OPENAI_MODEL, OPENAI_API_BASE
    if not USE_OPENAI:
        return {"ok": False, "error": "LLM 未启用"}
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return {"ok": False, "error": "未设置 OPENAI_API_KEY"}
    t0 = time.monotonic()
    try:
        from openai import OpenAI
        kwargs = {"api_key": key}
        if OPENAI_API_BASE:
            kwargs["base_url"] = OPENAI_API_BASE
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "请回复OK"}],
            max_tokens=10,
            temperature=0,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        reply = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        return {
            "ok": True,
            "model": OPENAI_MODEL,
            "api_base": OPENAI_API_BASE or "(default)",
            "reply": reply,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {"ok": False, "error": str(e), "latency_ms": latency_ms}


# ===== 系统状态接口 =====

@app.get("/admin/stats")
def admin_stats():
    """获取系统统计信息"""
    from rag_runtime_config import STORE_ROOT
    products_info = []
    if KNOWLEDGE_DIR.exists():
        shared_names = _SHARED_DIR_NAMES
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if not p.is_dir() or p.name in shared_names:
                continue
            info = {"name": p.name, "files": [], "total_size": 0}
            for f in p.iterdir():
                if f.is_file():
                    size = f.stat().st_size
                    info["files"].append({"name": f.name, "size": size})
                    info["total_size"] += size
            # 索引状态
            store = STORE_ROOT / p.name
            info["index_exists"] = (store / "index.faiss").exists()
            info["docs_count"] = 0
            docs_path = store / "docs.jsonl"
            if docs_path.exists():
                with open(docs_path, "r", encoding="utf-8") as f:
                    info["docs_count"] = sum(1 for _ in f)
            products_info.append(info)
    # 共享知识
    shared_store = STORE_ROOT / "_shared"
    shared_docs = 0
    if (shared_store / "docs.jsonl").exists():
        with open(shared_store / "docs.jsonl", "r", encoding="utf-8") as f:
            shared_docs = sum(1 for _ in f)
    return {
        "products": products_info,
        "shared_docs": shared_docs,
        "shared_indexed": (shared_store / "index.faiss").exists(),
    }


# ===== 知识库导入接口（LLM 自动整理） =====

# 导入锁，防止并发导入
_import_locks: Dict[str, threading.Lock] = {}
_import_locks_guard = threading.Lock()


class ImportKnowledgeRequest(BaseModel):
    """通过文本内容导入知识库"""
    type: str = Field(..., description="知识类型: product/procedure/equipment/material/anatomy/indication/complication/course/script")
    id: str = Field(default="", description="实体ID（目录名），product/procedure/equipment/material 必填")
    content: str = Field(..., min_length=10, description="原始文档文本内容")
    build: bool = Field(default=True, description="导入后自动构建索引")
    dry_run: bool = Field(default=False, description="仅预览，不写入文件")


@app.post("/admin/import_knowledge")
def admin_import_knowledge(req: ImportKnowledgeRequest):
    """通过 LLM 自动将原始文档整理为结构化知识库文件。

    用法：POST JSON 请求体，content 字段放原始文档文本。
    LLM 会自动整理为 main.txt / faq.txt / alias.txt 等结构化文件。
    """
    from import_knowledge import (
        _ENTITY_TYPES, _get_openai_client, _generate_knowledge,
        _write_knowledge_files,
    )

    entity_type = req.type.strip()
    entity_id = req.id.strip()

    # 校验类型
    if entity_type not in _ENTITY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的类型: {entity_type}，可选: {', '.join(_ENTITY_TYPES.keys())}")

    # 校验 ID
    _, is_single = _ENTITY_TYPES[entity_type]
    if not is_single and not entity_id:
        raise HTTPException(
            status_code=400,
            detail=f"类型 '{entity_type}' 需要提供 id 参数")

    # 安全校验 ID
    if entity_id:
        if ".." in entity_id or "/" in entity_id or "\\" in entity_id:
            raise HTTPException(status_code=400, detail="非法 ID")
        if not _SAFE_NAME_RE.match(entity_id):
            raise HTTPException(status_code=400, detail="ID 只允许字母、数字、下划线、横线、中文")

    raw_text = req.content.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="文档内容不能为空")

    # 获取导入锁
    lock_key = f"{entity_type}:{entity_id or '_single'}"
    with _import_locks_guard:
        if lock_key not in _import_locks:
            _import_locks[lock_key] = threading.Lock()
        lock = _import_locks[lock_key]

    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="该知识正在导入中，请稍后重试")

    try:
        # 检查 LLM 是否可用（优先检查知识库专用 LLM，回退到全局）
        try:
            from llm_client import is_enabled as _llm_is_enabled
            knowledge_llm_ok = _llm_is_enabled("knowledge")
        except ImportError:
            knowledge_llm_ok = False
        if not knowledge_llm_ok:
            from rag_runtime_config import USE_OPENAI
            if not USE_OPENAI:
                raise HTTPException(
                    status_code=400,
                    detail="知识库 LLM 未启用，请先在管理后台「模型切换」中配置知识库整理用的 LLM")
        from rag_runtime_config import OPENAI_MODEL

        # 调用 LLM 整理
        client = _get_openai_client()
        result = _generate_knowledge(client, raw_text, entity_type, entity_id)

        # 写入文件
        out_dir = _write_knowledge_files(result, entity_type, entity_id,
                                          dry_run=req.dry_run)

        # 构建索引
        built_index = False
        if req.build and not req.dry_run:
            try:
                if entity_type == "product":
                    from build_faiss import build_for_product
                    build_for_product(entity_id)
                    invalidate_store_cache(entity_id)
                else:
                    from build_faiss import build_shared
                    build_shared()
                    invalidate_store_cache("_shared")
                # 清除健康检查缓存
                with _health_lock:
                    global _health_cache
                    _health_cache = {}
                built_index = True
            except Exception as e:
                log_error("import_build_index", repr(e),
                          meta={"type": entity_type, "id": entity_id})

        # 构建响应
        response = {
            "ok": True,
            "type": entity_type,
            "id": entity_id,
            "dry_run": req.dry_run,
            "output_dir": str(out_dir),
            "built_index": built_index,
            "model": _get_knowledge_model_name(),
            "files_generated": {},
        }
        if result.get("main_txt"):
            response["files_generated"]["main.txt"] = len(result["main_txt"])
        if result.get("faq_txt"):
            response["files_generated"]["faq.txt"] = len(result["faq_txt"])
        if result.get("alias_txt"):
            response["files_generated"]["alias.txt"] = len(result["alias_txt"])

        # 别名注册提示
        aliases = result.get("product_aliases") or result.get("entity_aliases") or []
        name = result.get("product_name") or result.get("entity_name") or entity_id
        if aliases:
            alias_config_key = {
                "product": "PRODUCT_ALIASES",
                "procedure": "PROCEDURE_ALIASES",
                "equipment": "EQUIPMENT_ALIASES",
                "material": "MATERIAL_ALIASES",
            }.get(entity_type)
            if alias_config_key:
                response["alias_hint"] = {
                    "config_key": alias_config_key,
                    "entity_id": entity_id,
                    "name": name,
                    "aliases": aliases,
                }

        # dry_run 时返回预览内容
        if req.dry_run:
            response["preview"] = {}
            if result.get("main_txt"):
                txt = result["main_txt"]
                response["preview"]["main_txt"] = txt[:3000] + (f"\n...(省略 {len(txt)-3000} 字)" if len(txt) > 3000 else "")
            if result.get("faq_txt"):
                txt = result["faq_txt"]
                response["preview"]["faq_txt"] = txt[:2000] + (f"\n...(省略 {len(txt)-2000} 字)" if len(txt) > 2000 else "")
            if result.get("alias_txt"):
                response["preview"]["alias_txt"] = result["alias_txt"]

        return response

    except HTTPException:
        raise
    except Exception as e:
        log_error("admin_import_knowledge", repr(e),
                  meta={"type": entity_type, "id": entity_id})
        raise HTTPException(status_code=500, detail="导入失败，请查看服务器日志")
    finally:
        lock.release()


class RefineKnowledgeRequest(BaseModel):
    """修订知识库内容"""
    type: str = Field(..., description="知识类型")
    id: str = Field(default="", description="实体ID")
    feedback: str = Field(..., min_length=2, description="用户修改意见")
    current: dict = Field(..., description="当前整理结果，包含 main_txt/faq_txt/alias_txt")
    raw_text: str = Field(default="", description="可选，原始文档供 LLM 参考")


class CommitKnowledgeRequest(BaseModel):
    """正式写入知识库"""
    type: str = Field(..., description="知识类型")
    id: str = Field(default="", description="实体ID")
    content: dict = Field(..., description="最终确认的内容，包含 main_txt/faq_txt/alias_txt")
    build: bool = Field(default=True, description="是否自动构建索引")


@app.post("/admin/import_knowledge/refine")
def admin_import_knowledge_refine(req: RefineKnowledgeRequest):
    """根据用户反馈修订知识库内容（LLM 修订）。"""
    from import_knowledge import (
        _ENTITY_TYPES, _get_openai_client, refine_knowledge,
    )

    entity_type = req.type.strip()
    entity_id = req.id.strip()

    if entity_type not in _ENTITY_TYPES:
        raise HTTPException(status_code=400,
            detail=f"不支持的类型: {entity_type}")

    _, is_single = _ENTITY_TYPES[entity_type]
    if not is_single and not entity_id:
        raise HTTPException(status_code=400,
            detail=f"类型 '{entity_type}' 需要提供 id 参数")

    if entity_id:
        if ".." in entity_id or "/" in entity_id or "\\" in entity_id:
            raise HTTPException(status_code=400, detail="非法 ID")
        if not _SAFE_NAME_RE.match(entity_id):
            raise HTTPException(status_code=400, detail="ID 只允许字母、数字、下划线、横线、中文")

    lock_key = f"refine:{entity_type}:{entity_id or '_single'}"
    with _import_locks_guard:
        if lock_key not in _import_locks:
            _import_locks[lock_key] = threading.Lock()
        lock = _import_locks[lock_key]

    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="正在修订中，请稍后重试")

    try:
        # 检查知识库 LLM 是否可用
        try:
            from llm_client import is_enabled as _llm_is_enabled
            knowledge_llm_ok = _llm_is_enabled("knowledge")
        except ImportError:
            knowledge_llm_ok = False
        if not knowledge_llm_ok:
            from rag_runtime_config import USE_OPENAI
            if not USE_OPENAI:
                raise HTTPException(status_code=400,
                    detail="知识库 LLM 未启用，请先在管理后台配置知识库整理用的 LLM")

        client = _get_openai_client()
        result = refine_knowledge(
            client,
            current=req.current,
            feedback=req.feedback,
            raw_text=req.raw_text,
            entity_type=entity_type,
        )

        response = {
            "ok": True,
            "type": entity_type,
            "id": entity_id,
            "model": _get_knowledge_model_name(),
            "preview": {},
            "files_generated": {},
        }

        if result.get("main_txt"):
            response["preview"]["main_txt"] = result["main_txt"]
            response["files_generated"]["main.txt"] = len(result["main_txt"])
        if result.get("faq_txt"):
            response["preview"]["faq_txt"] = result["faq_txt"]
            response["files_generated"]["faq.txt"] = len(result["faq_txt"])
        if result.get("alias_txt"):
            response["preview"]["alias_txt"] = result["alias_txt"]
            response["files_generated"]["alias.txt"] = len(result["alias_txt"])

        return response

    except HTTPException:
        raise
    except Exception as e:
        log_error("admin_import_refine", repr(e),
                  meta={"type": entity_type, "id": entity_id})
        raise HTTPException(status_code=500, detail="修订失败，请查看服务器日志")
    finally:
        lock.release()


@app.post("/admin/import_knowledge/commit")
def admin_import_knowledge_commit(req: CommitKnowledgeRequest):
    """将审阅确认的内容正式写入知识库文件并建索引。"""
    from import_knowledge import (
        _ENTITY_TYPES, _write_knowledge_files,
    )

    entity_type = req.type.strip()
    entity_id = req.id.strip()

    if entity_type not in _ENTITY_TYPES:
        raise HTTPException(status_code=400,
            detail=f"不支持的类型: {entity_type}")

    _, is_single = _ENTITY_TYPES[entity_type]
    if not is_single and not entity_id:
        raise HTTPException(status_code=400,
            detail=f"类型 '{entity_type}' 需要提供 id 参数")

    if entity_id:
        if ".." in entity_id or "/" in entity_id or "\\" in entity_id:
            raise HTTPException(status_code=400, detail="非法 ID")
        if not _SAFE_NAME_RE.match(entity_id):
            raise HTTPException(status_code=400, detail="ID 只允许字母、数字、下划线、横线、中文")

    content = req.content
    if not content.get("main_txt"):
        raise HTTPException(status_code=400, detail="内容不能为空（至少需要 main_txt）")

    lock_key = f"commit:{entity_type}:{entity_id or '_single'}"
    with _import_locks_guard:
        if lock_key not in _import_locks:
            _import_locks[lock_key] = threading.Lock()
        lock = _import_locks[lock_key]

    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="正在写入中，请稍后重试")

    try:
        out_dir = _write_knowledge_files(content, entity_type, entity_id,
                                          dry_run=False)

        built_index = False
        if req.build:
            try:
                if entity_type == "product":
                    from build_faiss import build_for_product
                    build_for_product(entity_id)
                    invalidate_store_cache(entity_id)
                else:
                    from build_faiss import build_shared
                    build_shared()
                    invalidate_store_cache("_shared")
                with _health_lock:
                    global _health_cache
                    _health_cache = {}
                built_index = True
            except Exception as e:
                log_error("commit_build_index", repr(e),
                          meta={"type": entity_type, "id": entity_id})

        return {
            "ok": True,
            "type": entity_type,
            "id": entity_id,
            "output_dir": str(out_dir),
            "built_index": built_index,
        }

    except HTTPException:
        raise
    except Exception as e:
        log_error("admin_import_commit", repr(e),
                  meta={"type": entity_type, "id": entity_id})
        raise HTTPException(status_code=500, detail="写入失败，请查看服务器日志")
    finally:
        lock.release()


@app.post("/admin/import_knowledge_file")
async def admin_import_knowledge_file(request: "Request"):
    """通过文件上传导入知识库（LLM 自动整理）。

    Form fields:
        - type: 知识类型 (product/procedure/equipment/material/...)
        - id: 实体ID（product/procedure/equipment/material 必填）
        - build: "1" 导入后自动建索引（默认 "1"）
        - dry_run: "1" 仅预览（默认 "0"）
        - file: 上传的文件（支持 .txt .md .pdf）
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="需要 multipart/form-data 格式")

    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail="解析表单失败")

    try:
        entity_type = str(form.get("type", "")).strip()
        entity_id = str(form.get("id", "")).strip()
        build = str(form.get("build", "1")).strip().lower() in ("1", "true", "yes")
        dry_run = str(form.get("dry_run", "0")).strip().lower() in ("1", "true", "yes")

        if not entity_type:
            raise HTTPException(status_code=400, detail="缺少 type 字段")

        # 读取上传的文件
        file_item = form.get("file")
        if not file_item or not hasattr(file_item, "read"):
            raise HTTPException(status_code=400, detail="缺少 file 文件")

        fname = getattr(file_item, "filename", "") or "upload.txt"
        suffix = Path(fname).suffix.lower()

        file_data = await file_item.read()

        if suffix == ".pdf":
            try:
                import pdfplumber
                import io
                text_parts = []
                with pdfplumber.open(io.BytesIO(file_data)) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text_parts.append(t)
                raw_text = "\n\n".join(text_parts)
            except ImportError:
                raise HTTPException(status_code=400, detail="服务器未安装 pdfplumber，无法处理 PDF")
        else:
            # txt / md 等文本文件
            for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312"):
                try:
                    raw_text = file_data.decode(enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                raw_text = file_data.decode("utf-8", errors="replace")

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="上传的文件内容为空")

        # 构造请求，复用 JSON 接口逻辑
        import_req = ImportKnowledgeRequest(
            type=entity_type,
            id=entity_id,
            content=raw_text,
            build=build,
            dry_run=dry_run,
        )
        return admin_import_knowledge(import_req)
    finally:
        await form.close()
