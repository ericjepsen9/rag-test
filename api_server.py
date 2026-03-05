from pathlib import Path
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from media_router import find_media
from rag_answer import answer_question, _store_cache
from rag_logger import log_error, get_recent_qa, get_recent_misses, get_recent_errors
from rag_runtime_config import KNOWLEDGE_DIR

BASE_DIR = Path(__file__).resolve().parent
ADMIN_PAGE = BASE_DIR / "admin_page.html"
CHAT_PAGE = BASE_DIR / "web" / "chat.html"

app = FastAPI(title="Medical Aesthetics RAG API", version="1.0.0")

MAX_QUESTION_LEN = 500


# ===== 数据模型 =====

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LEN)
    mode: Literal["brief", "full"] = "brief"
    debug: bool = False


class MediaItem(BaseModel):
    title: str
    type: str
    url: str = ""


class AskResponse(BaseModel):
    ok: bool
    answer: str
    media: List[MediaItem] = []
    debug: Optional[Dict[str, Any]] = None


class RebuildRequest(BaseModel):
    product: str = Field(..., min_length=1, max_length=50)
    timeout_sec: int = Field(default=120, ge=10, le=600)


# ===== 问答接口 =====

@app.get("/health")
def health():
    return {
        "status": "ok",
        "base_dir": str(BASE_DIR),
        "knowledge_exists": KNOWLEDGE_DIR.exists(),
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

    try:
        answer = answer_question(question, req.mode)
        media = [MediaItem(**m) for m in find_media(question)]
        debug = None
        if req.debug:
            debug = {"question": question, "mode": req.mode}
        return AskResponse(
            ok=True,
            answer=answer,
            media=media,
            debug=debug,
        )
    except Exception as e:
        log_error("api_ask", repr(e), meta={"question": question[:200]})
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
        _store_cache.pop(req.product.strip(), None)
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
