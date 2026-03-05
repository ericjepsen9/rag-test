import traceback
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from media_router import find_media
from rag_answer import answer_question
from rag_logger import log_error

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Medical Aesthetics RAG API", version="1.0.0")

# ===== 常量 =====
MAX_QUESTION_LEN = 500


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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "base_dir": str(BASE_DIR),
    }


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
