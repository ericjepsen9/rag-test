import traceback
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List

from fastapi import FastAPI
from pydantic import BaseModel

from media_router import find_media
from rag_answer import answer_question

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Medical Aesthetics RAG API", version="1.0.0")


class AskRequest(BaseModel):
    question: str
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
    stderr: str = ""
    debug: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "base_dir": str(BASE_DIR),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        answer = answer_question(req.question, req.mode)
        media = [MediaItem(**m) for m in find_media(req.question)]
        debug = None
        if req.debug:
            debug = {"question": req.question, "mode": req.mode}
        return AskResponse(
            ok=True,
            answer=answer,
            media=media,
            debug=debug,
        )
    except Exception as e:
        return AskResponse(
            ok=False,
            answer="接口执行异常",
            stderr=traceback.format_exc()[:2000],
        )
