import os
import time
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from media_router import find_media
from rag_answer import answer_question, invalidate_store_cache
from rag_logger import log_error, get_recent_qa, get_recent_misses, get_recent_errors
from rag_runtime_config import KNOWLEDGE_DIR, SHARED_ENTITY_DIRS

BASE_DIR = Path(__file__).resolve().parent
ADMIN_PAGE = BASE_DIR / "admin_page.html"
CHAT_PAGE = BASE_DIR / "web" / "chat.html"

app = FastAPI(title="Medical Aesthetics RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
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


# ===== 问答接口 =====

@app.get("/health")
def health():
    from rag_runtime_config import STORE_ROOT
    shared_names = set(SHARED_ENTITY_DIRS.values())
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
    return {
        "status": "ok" if all_indexed else "degraded",
        "knowledge_exists": KNOWLEDGE_DIR.exists(),
        "products": products,
        "shared_knowledge_indexed": shared_indexed,
    }


@app.get("/chat")
def chat_page():
    if not CHAT_PAGE.exists():
        raise HTTPException(status_code=404, detail="chat.html 不存在")
    return FileResponse(CHAT_PAGE, media_type="text/html")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    t0 = time.monotonic()
    rw = None
    try:
        history = [{"role": h.role, "content": h.content} for h in req.history[-6:]]
        # 防御：限制历史总字符数，避免过大 payload 占用内存
        total_chars = sum(len(h.get("content", "")) for h in history)
        if total_chars > MAX_HISTORY_TOTAL_CHARS:
            # 从最早的消息开始裁剪，保留最近的对话
            while history and total_chars > MAX_HISTORY_TOTAL_CHARS:
                total_chars -= len(history[0].get("content", ""))
                history.pop(0)
        # rewrite 只做一次，传给 answer_question 复用
        from query_rewrite import rewrite_query
        rw = rewrite_query(question, history=history)
        resolved_q = rw["original"]  # 上下文补全后的问题

        answer = answer_question(question, req.mode, history=history, rewrite=rw)
        latency_ms = int((time.monotonic() - t0) * 1000)
        # 用产品+路由精准匹配媒体，关键词兜底
        from rag_answer import detect_route, detect_product
        product_id = detect_product(resolved_q)
        route = detect_route(resolved_q)
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
                error_meta["route"] = detect_route(rw.get("original", question))
                error_meta["product"] = detect_product(rw.get("original", question))
        except Exception:
            pass
        log_error("api_ask", repr(e), meta=error_meta)
        return AskResponse(
            ok=False,
            answer="接口执行异常，请稍后重试",
        )


# ===== 管理接口 =====

@app.get("/admin")
def admin_page():
    if not ADMIN_PAGE.exists():
        raise HTTPException(status_code=404, detail="admin_page.html 不存在")
    return FileResponse(ADMIN_PAGE)


@app.get("/admin/products")
def admin_products():
    products = []
    if KNOWLEDGE_DIR.exists():
        for p in sorted(KNOWLEDGE_DIR.iterdir()):
            if not p.is_dir():
                continue
            products.append({
                "product": p.name,
                "files": sorted([x.name for x in p.iterdir() if x.is_file()]),
            })
    return {"products": products}


@app.post("/admin/rebuild")
def admin_rebuild(req: RebuildRequest):
    from build_faiss import build_for_product
    try:
        build_for_product(req.product.strip())
        # 重建后清除缓存，下次请求会加载新索引
        invalidate_store_cache(req.product.strip())
        # 同时清除关联数据缓存
        from relation_engine import invalidate_relations_cache
        invalidate_relations_cache()
        return {"ok": True, "product": req.product}
    except Exception as e:
        log_error("admin_rebuild", repr(e), meta={"product": req.product})
        raise HTTPException(status_code=500, detail=repr(e))


@app.get("/admin/logs/qa")
def admin_logs_qa(limit: int = 20):
    return {"items": get_recent_qa(limit=limit)}


@app.get("/admin/logs/miss")
def admin_logs_miss(limit: int = 20):
    return {"items": get_recent_misses(limit=limit)}


@app.get("/admin/logs/error")
def admin_logs_error(limit: int = 20):
    return {"items": get_recent_errors(limit=limit)}
