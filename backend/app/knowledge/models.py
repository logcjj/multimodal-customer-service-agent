from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(UTC)


class FileAssetRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    original_name: str
    content_hash: str = Field(index=True)
    mime_type: str
    size_bytes: int
    storage_path: str
    status: str = "ready"
    created_at: datetime = Field(default_factory=utc_now)


class DatasetRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str = Field(index=True)
    description: str = ""
    default_parser_profile: str = "manual"
    embedding_model_id: str | None = None
    retrieval_profile_id: str | None = None
    visibility: str = "private"
    published_version: str | None = Field(default=None, index=True)
    status: str = "ready"
    is_system: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DocumentRefRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset_id: str = Field(index=True)
    file_id: str = Field(index=True)
    parser_profile: str = "manual"
    metadata_json: str = "{}"
    enabled: bool = True
    active_version: str | None = None
    published_version: str | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ParsingJobRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    document_ref_id: str = Field(index=True)
    state: str = "queued"
    stage: str = "extract"
    progress: int = 0
    parser_version: str = "v3.1-parser-1"
    config_hash: str = ""
    index_version: str | None = Field(default=None, index=True)
    error_code: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)


class IndexBuildRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset_id: str = Field(index=True)
    state: str = "queued"
    stage: str = "snapshot"
    progress: int = 0
    index_version: str | None = Field(default=None, index=True)
    bundle_path: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)


class IndexManifestRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset_id: str = Field(index=True)
    index_version: str = Field(index=True)
    bundle_path: str
    manifest_json: str
    validation_status: str = "valid"
    active: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    published_at: datetime | None = None


class ParentChunkRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset_id: str = Field(index=True)
    document_id: str = Field(index=True)
    index_version: str = Field(index=True)
    local_id: str
    title: str
    heading_path_json: str = "[]"
    text: str
    page_start: int = 1
    page_end: int = 1
    bbox_json: str | None = None
    token_count: int = 0
    content_hash: str
    enabled: bool = True
    edited: bool = False
    product: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ChildChunkRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    parent_id: str = Field(index=True)
    dataset_id: str = Field(index=True)
    document_id: str = Field(index=True)
    index_version: str = Field(index=True)
    local_id: str
    title: str
    text: str
    normalized_text: str
    page_start: int = 1
    page_end: int = 1
    bbox_json: str | None = None
    token_count: int = 0
    keywords_json: str = "[]"
    questions_json: str = "[]"
    tags_json: str = "[]"
    asset_ids_json: str = "[]"
    content_hash: str
    enabled: bool = True
    edited: bool = False
    product: str | None = None
    embedding_json: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeAssetRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset_id: str = Field(index=True)
    document_id: str = Field(index=True)
    index_version: str = Field(index=True)
    asset_type: str
    page_number: int
    bbox_json: str | None = None
    storage_path: str
    caption: str = ""
    ocr_text: str = ""
    related_parent_ids_json: str = "[]"
    related_child_ids_json: str = "[]"
    created_at: datetime = Field(default_factory=utc_now)


class ImageChunkRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset_id: str = Field(index=True)
    document_id: str = Field(index=True)
    index_version: str = Field(index=True)
    asset_id: str = Field(index=True)
    image_id: str
    manual_name: str
    chapter_title: str = ""
    page_number: int = 1
    caption: str = ""
    ocr_text: str = ""
    visible_text_json: str = "[]"
    visual_summary: str = ""
    visual_meaning: str = ""
    retrieval_text: str = ""
    search_terms_json: str = "[]"
    applicable_questions_json: str = "[]"
    issue_signals_json: str = "[]"
    related_parent_ids_json: str = "[]"
    related_child_ids_json: str = "[]"
    confidence: float = 0.0
    content_hash: str
    embedding_json: str | None = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=utc_now)


class RetrievalProfileRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    lexical_top_k: int = 20
    dense_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 12
    final_top_n: int = 5
    min_score: float = 0.012
    min_coverage: float = 0.1
    parent_strategy: str = "best_plus_coverage"
    empty_response: str = "当前知识库中没有足够证据，请补充产品型号或问题细节。"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
