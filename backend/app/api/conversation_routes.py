from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from app.contracts.models import (
    ConversationCreate,
    ConversationDetail,
    ConversationSummary,
    ConversationUpdate,
)


router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    request: Request,
    user_id: str = Query(min_length=1, max_length=120),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=100),
) -> list[ConversationSummary]:
    return request.app.state.conversations.list(
        user_id,
        offset=offset,
        limit=limit,
    )


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreate,
    request: Request,
    user_id: str = Query(min_length=1, max_length=120),
) -> ConversationSummary:
    try:
        return request.app.state.conversations.create(
            user_id,
            conversation_id=payload.id,
            title=payload.title,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="对话不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    request: Request,
    user_id: str = Query(min_length=1, max_length=120),
) -> ConversationDetail:
    conversation = request.app.state.conversations.get(conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conversation


@router.patch("/{conversation_id}", response_model=ConversationSummary)
def rename_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    request: Request,
    user_id: str = Query(min_length=1, max_length=120),
) -> ConversationSummary:
    try:
        conversation = request.app.state.conversations.rename(
            conversation_id,
            user_id,
            payload.title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    request: Request,
    user_id: str = Query(min_length=1, max_length=120),
) -> Response:
    request.app.state.conversations.delete(conversation_id, user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
