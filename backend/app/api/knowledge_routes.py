from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from app.knowledge.contracts import (
    ChildChunkView,
    ChunkCollectionView,
    ChunkUpdate,
    DatasetCreate,
    DatasetUpdate,
    DatasetView,
    DocumentLinkCreate,
    DocumentView,
    FileAssetView,
    ParsingJobView,
    PublishRequest,
    VectorMapView,
)
from app.evaluation.service import PublishGateError


router = APIRouter(prefix="/api", tags=["knowledge"])


@router.post("/files", response_model=FileAssetView, status_code=status.HTTP_201_CREATED)
async def upload_file(request: Request, file: UploadFile = File(...)) -> FileAssetView:
    content = await file.read()
    return request.app.state.knowledge_service.create_file(
        file.filename or "upload.bin",
        file.content_type,
        content,
    )


@router.get("/files", response_model=list[FileAssetView])
def list_files(request: Request) -> list[FileAssetView]:
    return request.app.state.knowledge_service.list_files()


@router.get("/files/{file_id}/content")
def get_file_content(file_id: str, request: Request) -> FileResponse:
    path, original_name, mime_type = request.app.state.knowledge_service.file_content(file_id)
    encoded_name = quote(original_name, safe="")
    return FileResponse(
        path,
        media_type=mime_type,
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{encoded_name}"},
    )


@router.get("/assets/{asset_id}")
def get_asset(asset_id: str, request: Request) -> FileResponse:
    path = request.app.state.knowledge_service.asset_path(asset_id)
    return FileResponse(path)


@router.post("/datasets", response_model=DatasetView, status_code=status.HTTP_201_CREATED)
def create_dataset(payload: DatasetCreate, request: Request) -> DatasetView:
    return request.app.state.knowledge_service.create_dataset(payload)


@router.get("/datasets", response_model=list[DatasetView])
def list_datasets(request: Request) -> list[DatasetView]:
    return request.app.state.knowledge_service.list_datasets()


@router.get("/datasets/{dataset_id}", response_model=DatasetView)
def get_dataset(dataset_id: str, request: Request) -> DatasetView:
    return request.app.state.knowledge_service.dataset_view(dataset_id)


@router.get("/datasets/{dataset_id}/vector-map", response_model=VectorMapView)
def get_vector_map(dataset_id: str, request: Request) -> VectorMapView:
    return request.app.state.knowledge_service.vector_map(dataset_id)


@router.post(
    "/datasets/{dataset_id}/vector-map/rebuild",
    response_model=VectorMapView,
    status_code=status.HTTP_202_ACCEPTED,
)
def rebuild_vector_map(dataset_id: str, request: Request) -> VectorMapView:
    payload = request.app.state.knowledge_service.rebuild_vector_map(dataset_id)
    if payload.get("status") in {"no_published_version", "no_embeddings"}:
        detail = payload.get("message") or "当前知识库无法接受向量图重建任务。"
        raise HTTPException(status_code=409, detail=detail)
    return payload


@router.patch("/datasets/{dataset_id}", response_model=DatasetView)
def update_dataset(dataset_id: str, payload: DatasetUpdate, request: Request) -> DatasetView:
    return request.app.state.knowledge_service.update_dataset(dataset_id, payload)


@router.post(
    "/datasets/{dataset_id}/documents",
    response_model=DocumentView,
    status_code=status.HTTP_201_CREATED,
)
def link_document(dataset_id: str, payload: DocumentLinkCreate, request: Request) -> DocumentView:
    return request.app.state.knowledge_service.link_document(dataset_id, payload)


@router.get("/datasets/{dataset_id}/documents", response_model=list[DocumentView])
def list_documents(dataset_id: str, request: Request) -> list[DocumentView]:
    return request.app.state.knowledge_service.list_documents(dataset_id)


@router.post("/documents/{document_id}/parse", response_model=ParsingJobView)
def parse_document(document_id: str, request: Request) -> ParsingJobView:
    return request.app.state.knowledge_service.parse_document(document_id)


@router.get("/parsing-jobs/{job_id}", response_model=ParsingJobView)
def get_job(job_id: str, request: Request) -> ParsingJobView:
    return request.app.state.knowledge_service.job_view(job_id)


@router.post("/datasets/{dataset_id}/publish", response_model=DatasetView)
def publish_dataset(dataset_id: str, payload: PublishRequest, request: Request) -> DatasetView:
    try:
        request.app.state.evaluation_service.require_publish_approval(
            dataset_id,
            payload.index_version,
            payload.evaluation_run_id,
        )
    except PublishGateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return request.app.state.knowledge_service.publish(dataset_id, payload.index_version)


@router.get("/documents/{document_id}/chunks", response_model=ChunkCollectionView)
def list_chunks(document_id: str, request: Request) -> ChunkCollectionView:
    return request.app.state.knowledge_service.chunks(document_id)


@router.patch("/chunks/{chunk_id}", response_model=ChildChunkView)
def update_chunk(chunk_id: str, payload: ChunkUpdate, request: Request) -> ChildChunkView:
    return request.app.state.knowledge_service.edit_child(
        chunk_id,
        **payload.model_dump(exclude_none=True),
    )
