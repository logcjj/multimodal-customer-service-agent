from __future__ import annotations

from fastapi import APIRouter, Request, status

from app.knowledge.contracts import (
    RetrievalProfileCreate,
    RetrievalProfileUpdate,
    RetrievalProfileView,
    RetrievalTestRequest,
)


router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.post("/test")
def retrieval_test(payload: RetrievalTestRequest, request: Request) -> dict[str, object]:
    return request.app.state.knowledge_service.retrieval_test(payload)


@router.post("/profiles", response_model=RetrievalProfileView, status_code=status.HTTP_201_CREATED)
def create_profile(payload: RetrievalProfileCreate, request: Request) -> RetrievalProfileView:
    return request.app.state.knowledge_service.create_retrieval_profile(payload)


@router.get("/profiles", response_model=list[RetrievalProfileView])
def list_profiles(request: Request) -> list[RetrievalProfileView]:
    return request.app.state.knowledge_service.list_retrieval_profiles()


@router.patch("/profiles/{profile_id}", response_model=RetrievalProfileView)
def update_profile(
    profile_id: str,
    payload: RetrievalProfileUpdate,
    request: Request,
) -> RetrievalProfileView:
    return request.app.state.knowledge_service.update_retrieval_profile(profile_id, payload)
