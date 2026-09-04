from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.knowledge.contracts import ImageChunkView, IndexActivationRequest, IndexBuildView
from app.knowledge.index_bundle import IndexManifest


router = APIRouter(prefix="/api", tags=["index-builds"])


@router.post(
    "/datasets/{dataset_id}/index-builds",
    response_model=IndexBuildView,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_index_build(dataset_id: str, request: Request) -> IndexBuildView:
    return request.app.state.knowledge_service.start_index_build(dataset_id)


@router.get("/index-builds/{build_id}", response_model=IndexBuildView)
def get_index_build(build_id: str, request: Request) -> IndexBuildView:
    return request.app.state.knowledge_service.index_build_view(build_id)


@router.get("/datasets/{dataset_id}/index-manifest", response_model=IndexManifest)
def get_index_manifest(dataset_id: str, request: Request) -> IndexManifest:
    return request.app.state.knowledge_service.active_index_manifest(dataset_id)


@router.get(
    "/datasets/{dataset_id}/index-manifests/{index_version}",
    response_model=IndexManifest,
)
def get_versioned_index_manifest(
    dataset_id: str,
    index_version: str,
    request: Request,
) -> IndexManifest:
    return request.app.state.knowledge_service.index_manifest(dataset_id, index_version)


@router.post(
    "/datasets/{dataset_id}/index-manifests/{index_version}/activate",
    response_model=IndexManifest,
)
def activate_previous_index_manifest(
    dataset_id: str,
    index_version: str,
    request: Request,
    payload: IndexActivationRequest | None = None,
) -> IndexManifest:
    service = request.app.state.knowledge_service
    record = service.repository.get_index_manifest(dataset_id, index_version)
    if record.published_at is not None:
        return service.activate_previous_index_manifest(dataset_id, index_version)
    dataset = service.repository.get_dataset(dataset_id)
    activation = payload or IndexActivationRequest()
    if dataset.is_system:
        gate = request.app.state.evaluation_service.assess_release_gate(
            dataset_id=dataset_id,
            candidate_version=dataset.published_version or index_version,
            run_id=activation.evaluation_run_id,
            frozen_score=activation.frozen_score,
        )
        if gate["status"] != "approved":
            raise HTTPException(status_code=409, detail=gate)
    return service.activate_candidate_index_manifest(dataset_id, index_version)


@router.get("/index-runtime")
def get_index_runtime(request: Request) -> dict[str, object]:
    service = request.app.state.knowledge_service
    service.preload_active_bundles()
    return {
        "mode": request.app.state.settings.offline_index_mode,
        "datasets": service.offline_index_status(),
    }


@router.get("/datasets/{dataset_id}/image-chunks", response_model=list[ImageChunkView])
def list_image_chunks(
    dataset_id: str,
    request: Request,
    document_id: str | None = Query(default=None),
    query: str | None = Query(default=None, max_length=500),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[ImageChunkView]:
    return request.app.state.knowledge_service.image_chunks(
        dataset_id,
        document_id=document_id,
        query=query,
        limit=limit,
        offset=offset,
    )


@router.get("/image-chunks/{image_chunk_id}", response_model=ImageChunkView)
def get_image_chunk(image_chunk_id: str, request: Request) -> ImageChunkView:
    return request.app.state.knowledge_service.image_chunk(image_chunk_id)
