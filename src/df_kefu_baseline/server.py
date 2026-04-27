from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .answer import AnswerEngine


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    images: list[str] = Field(default_factory=list)
    session_id: str | None = None
    stream: bool = False


app = FastAPI(title="DataFountain 1165 Customer Service Agent")
engine = AnswerEngine(use_llm=bool(os.getenv("OPENAI_API_KEY")))


@app.post("/chat")
def chat(payload: ChatRequest, authorization: str | None = Header(default=None)):
    expected_token = os.getenv("KAFU_API_TOKEN", "").strip()
    if expected_token:
        if authorization != f"Bearer {expected_token}":
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    session_id = payload.session_id or f"kf_session_{uuid.uuid4().hex}"
    answer = engine.answer(payload.question)
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "answer": answer,
            "session_id": session_id,
            "timestamp": int(time.time()),
        },
    }

