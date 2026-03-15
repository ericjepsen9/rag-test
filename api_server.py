import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI, HTTPException
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
    """启动时预热嵌入模型，避免首次查询延迟"""
    if os.environ.get("SKIP_WARMUP"):
        return
    try:
        from rag_answer import embed_query
        embed_query("预热查询")
        print("[INFO] 嵌入模型预热完成")
    except Exception as e:
        print(f"[WARN] 模型预热失败: {e}")

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
    question = _sanitize_input(conv[-1].content[:MAX_QUESTION_LEN])
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
        raise HTTPException(status_code=500, detail=repr(e))
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
        raise HTTPException(status_code=500, detail=repr(e))


@app.get("/admin/logs/qa")
def admin_logs_qa(limit: int = 20):
    return {"items": get_recent_qa(limit=min(max(1, limit), 100))}


@app.get("/admin/logs/miss")
def admin_logs_miss(limit: int = 20):
    return {"items": get_recent_misses(limit=min(max(1, limit), 100))}


@app.get("/admin/logs/error")
def admin_logs_error(limit: int = 20):
    return {"items": get_recent_errors(limit=min(max(1, limit), 100))}
