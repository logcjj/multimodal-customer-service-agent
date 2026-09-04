from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field


ParserProfile = Literal["general", "manual", "qa", "table", "picture"]


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    parser_profile: ParserProfile = "manual"
    visibility: Literal["private", "team"] = "private"


class DatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    parser_profile: ParserProfile | None = None
    visibility: Literal["private", "team"] | None = None
    retrieval_profile_id: str | None = None


class DatasetView(BaseModel):
    id: str
    name: str
    description: str
    parser_profile: str
    visibility: str
    published_version: str | None
    status: str
    is_system: bool
    retrieval_profile_id: str | None = None
    document_count: int = 0
    parent_count: int = 0
    child_count: int = 0
    asset_count: int = 0
    failed_job_count: int = 0
    created_at: datetime
    updated_at: datetime


class FileAssetView(BaseModel):
    id: str
    original_name: str
    content_hash: str
    mime_type: str
    size_bytes: int
    storage_path: str
    status: str
    created_at: datetime


class DocumentLinkCreate(BaseModel):
    file_id: str
    parser_profile: ParserProfile | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class DocumentView(BaseModel):
    id: str
    dataset_id: str
    file_id: str
    original_name: str
    mime_type: str
    parser_profile: str
    enabled: bool
    active_version: str | None
    published_version: str | None
    latest_job_state: str | None = None
    latest_job_progress: int | None = None
    created_at: datetime
    updated_at: datetime


class ParsingJobView(BaseModel):
    id: str
    document_ref_id: str
    state: str
    stage: str
    progress: int
    parser_version: str
    index_version: str | None
    error_code: str | None
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None


class IndexBuildView(BaseModel):
    id: str
    dataset_id: str
    state: str
    stage: str
    progress: int
    index_version: str | None
    error_code: str | None
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class IndexActivationRequest(BaseModel):
    evaluation_run_id: str | None = None
    frozen_score: float | None = Field(default=None, ge=0, le=1)


class PublishRequest(BaseModel):
    index_version: str
    evaluation_run_id: str | None = None


class ChunkUpdate(BaseModel):
    text: str | None = Field(default=None, min_length=1)
    keywords: list[str] | None = None
    questions: list[str] | None = None
    tags: list[str] | None = None
    enabled: bool | None = None


class ParentChunkView(BaseModel):
    id: str
    document_id: str
    index_version: str
    title: str
    heading_path: list[str]
    text: str
    page_start: int
    page_end: int
    token_count: int
    enabled: bool
    edited: bool


class ChildChunkView(BaseModel):
    id: str
    parent_id: str
    document_id: str
    index_version: str
    title: str
    text: str
    page_start: int
    page_end: int
    token_count: int
    keywords: list[str]
    questions: list[str]
    tags: list[str]
    asset_ids: list[str]
    enabled: bool
    edited: bool


class ImageChunkView(BaseModel):
    id: str
    dataset_id: str
    document_id: str
    index_version: str
    asset_id: str
    asset_url: str
    image_id: str
    manual_name: str
    chapter_title: str
    page_number: int
    caption: str
    ocr_text: str
    visible_text: list[str]
    visual_summary: str
    visual_meaning: str
    retrieval_text: str
    search_terms: list[str]
    applicable_questions: list[str]
    issue_signals: list[str]
    related_parent_ids: list[str]
    related_child_ids: list[str]
    confidence: float
    content_hash: str
    embedding_dimension: int
    enabled: bool


class ChunkCollectionView(BaseModel):
    parents: list[ParentChunkView]
    children: list[ChildChunkView]


class VectorMapPointView(BaseModel):
    child_id: str
    dataset_id: str
    document_id: str
    document_name: str
    title: str
    excerpt: str
    page_start: int
    page_end: int
    product: str | None
    x: float
    y: float


class VectorMapErrorView(BaseModel):
    code: str
    message: str


class VectorMapReadyView(BaseModel):
    status: Literal["ready"]
    message: str | None = None
    meta: dict[str, object]
    points: list[VectorMapPointView] = Field(default_factory=list)


class VectorMapBuildingView(BaseModel):
    status: Literal["building"]
    message: str | None = None
    meta: dict[str, object] | None = None
    points: list[VectorMapPointView] = Field(default_factory=list)


class VectorMapStaleView(BaseModel):
    status: Literal["stale"]
    message: str | None = None
    meta: dict[str, object] | None = None
    points: list[VectorMapPointView] = Field(default_factory=list)


class VectorMapFailedView(BaseModel):
    status: Literal["failed"]
    message: str | None = None
    error: VectorMapErrorView
    meta: dict[str, object] | None = None
    points: list[VectorMapPointView] = Field(default_factory=list)


class VectorMapMissingView(BaseModel):
    status: Literal["missing"]
    message: str | None = None
    meta: dict[str, object] | None = None
    points: list[VectorMapPointView] = Field(default_factory=list)


class VectorMapNoPublishedVersionView(BaseModel):
    status: Literal["no_published_version"]
    message: str | None = None
    meta: dict[str, object] | None = None
    points: list[VectorMapPointView] = Field(default_factory=list)


class VectorMapNoEmbeddingsView(BaseModel):
    status: Literal["no_embeddings"]
    message: str | None = None
    meta: dict[str, object] | None = None
    points: list[VectorMapPointView] = Field(default_factory=list)


VectorMapView = Annotated[
    VectorMapReadyView
    | VectorMapBuildingView
    | VectorMapStaleView
    | VectorMapFailedView
    | VectorMapMissingView
    | VectorMapNoPublishedVersionView
    | VectorMapNoEmbeddingsView,
    Field(discriminator="status"),
]


class RetrievalTestRequest(BaseModel):
    dataset_ids: list[str] = Field(min_length=1, max_length=20)
    query: str = Field(min_length=1, max_length=4000)
    top_n: int = Field(default=5, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0)
    use_rerank: bool = True
    profile_id: str | None = None


class RetrievalProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    lexical_top_k: int = Field(default=20, ge=1, le=100)
    dense_top_k: int = Field(default=20, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=500)
    rerank_top_k: int = Field(default=12, ge=1, le=100)
    final_top_n: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.012, ge=0, le=10)
    min_coverage: float = Field(default=0.1, ge=0, le=1)
    parent_strategy: Literal["best_plus_coverage"] = "best_plus_coverage"
    empty_response: str = Field(default="当前知识库中没有足够证据，请补充产品型号或问题细节。", max_length=500)


class RetrievalProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    lexical_top_k: int | None = Field(default=None, ge=1, le=100)
    dense_top_k: int | None = Field(default=None, ge=1, le=100)
    rrf_k: int | None = Field(default=None, ge=1, le=500)
    rerank_top_k: int | None = Field(default=None, ge=1, le=100)
    final_top_n: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0, le=10)
    min_coverage: float | None = Field(default=None, ge=0, le=1)
    empty_response: str | None = Field(default=None, max_length=500)


class RetrievalProfileView(BaseModel):
    id: str
    name: str
    lexical_top_k: int
    dense_top_k: int
    rrf_k: int
    rerank_top_k: int
    final_top_n: int
    min_score: float
    min_coverage: float
    parent_strategy: str
    empty_response: str
    created_at: datetime
    updated_at: datetime
