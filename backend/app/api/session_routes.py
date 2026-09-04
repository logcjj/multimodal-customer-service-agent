from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from app.contracts.models import SessionMemoryView


router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("/{session_id}", response_model=SessionMemoryView)
def get_session(
    session_id: str,
    request: Request,
    user_id: str | None = Query(default=None, max_length=120),
) -> SessionMemoryView:
    memory = request.app.state.session_memory.load(session_id, user_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return memory


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    request: Request,
    user_id: str | None = Query(default=None, max_length=120),
) -> Response:
    request.app.state.session_memory.delete(session_id, user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
