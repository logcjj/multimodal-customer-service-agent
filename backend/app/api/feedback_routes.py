from __future__ import annotations

from fastapi import APIRouter, Request, status

from app.observability.traces import FeedbackCreate


router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
def create_feedback(payload: FeedbackCreate, request: Request) -> dict[str, object]:
    return request.app.state.trace_store.add_feedback(payload)

