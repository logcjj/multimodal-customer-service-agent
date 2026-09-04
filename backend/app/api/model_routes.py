from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from app.contracts.models import (
    ModelConfiguration,
    ModelConfigurationCreate,
    ModelConfigurationUpdate,
)
from app.models.service import PROVIDER_CATALOG, ModelService


router = APIRouter(prefix="/api", tags=["models"])


def get_service(request: Request) -> ModelService:
    return request.app.state.model_service


@router.get("/providers")
def list_providers() -> list[dict[str, object]]:
    return PROVIDER_CATALOG


@router.get("/models", response_model=list[ModelConfiguration])
def list_models(request: Request) -> list[ModelConfiguration]:
    return get_service(request).list_models()


@router.post("/models", response_model=ModelConfiguration, status_code=status.HTTP_201_CREATED)
def create_model(payload: ModelConfigurationCreate, request: Request) -> ModelConfiguration:
    return get_service(request).create_model(payload)


@router.patch("/models/{model_id}", response_model=ModelConfiguration)
def update_model(
    model_id: str,
    payload: ModelConfigurationUpdate,
    request: Request,
) -> ModelConfiguration:
    return get_service(request).update_model(model_id, payload)


@router.post("/models/{model_id}/default", response_model=ModelConfiguration)
def set_default(model_id: str, request: Request) -> ModelConfiguration:
    return get_service(request).set_default(model_id)


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: str, request: Request) -> Response:
    get_service(request).delete_model(model_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/models/{model_id}/test")
async def test_model(model_id: str, request: Request) -> dict[str, object]:
    return await get_service(request).test_model(model_id)
