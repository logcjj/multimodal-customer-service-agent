from __future__ import annotations

import hashlib
import json
import math
import threading
from concurrent.futures import CancelledError, Future
from contextvars import ContextVar
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue

from fastapi import HTTPException

from app.config.runtime import RuntimeSettings
from app.knowledge.bundle_runtime import BundleRuntimeSnapshot, load_bundle_runtime
from app.knowledge.contracts import (
    ChildChunkView,
    ChunkCollectionView,
    DatasetCreate,
    DatasetUpdate,
    DatasetView,
    DocumentLinkCreate,
    DocumentView,
    FileAssetView,
    IndexBuildView,
    ImageChunkView,
    ParentChunkView,
    ParsingJobView,
    RetrievalTestRequest,
    RetrievalProfileCreate,
    RetrievalProfileUpdate,
    RetrievalProfileView,
)
from app.knowledge.hybrid import IndexedChild, PublishedHybridRetriever
from app.knowledge.image_retrieval import (
    IndexedImageChunk,
    PublishedImageRetriever,
    merge_image_evidence,
)
from app.knowledge.index_bundle import (
    IncrementalStats,
    IndexBundle,
    IndexBundleWriter,
    IndexManifest,
    SourceManifest,
    plan_incremental_build,
)
from app.knowledge.ingestion import IngestionService
from app.knowledge.repository import KnowledgeRepository
from app.knowledge.storage import ContentAddressedStorage
from app.knowledge.vector_map import VectorMapService, VectorSource
from app.storage.database import Database


class _VectorMapExecutorShutdown(RuntimeError):
    pass


class _DaemonBoundedExecutor:
    _STOP = object()

    def __init__(self, name: str, max_workers: int = 2) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._name = name
        self._queue = Queue()
        self._lock = threading.Lock()
        self._shutdown = False
        self._threads = [
            threading.Thread(target=self._run, name=f"{self._name}-{index + 1}", daemon=True)
            for index in range(max_workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future: Future = Future()
        with self._lock:
            if self._shutdown:
                future.set_exception(_VectorMapExecutorShutdown("向量图构建执行器已关闭，不能提交新任务。"))
                return future
            self._queue.put((future, fn, args, kwargs))
        return future

    def shutdown(self, *, cancel_pending: bool = True, wait: bool = True) -> None:
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                if cancel_pending:
                    self._cancel_pending_unlocked()
                for _ in self._threads:
                    self._queue.put(self._STOP)
            threads = list(self._threads)
        if wait:
            for thread in threads:
                thread.join()

    def _cancel_pending_unlocked(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                return
            try:
                if item is not self._STOP:
                    future = item[0]
                    future.cancel()
            finally:
                self._queue.task_done()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                future, fn, args, kwargs = item
                try:
                    if future.set_running_or_notify_cancel():
                        future.set_result(fn(*args, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)
            finally:
                self._queue.task_done()


def _vector_map_failed_payload(code: str, message: str) -> dict[str, object]:
    return {"status": "failed", "error": {"code": code, "message": message}}


class KnowledgeService:
    def __init__(self, database: Database, settings: RuntimeSettings | None = None) -> None:
        self.database = database
        self.settings = settings or RuntimeSettings.from_env()
        self.repository = KnowledgeRepository(database)
        self.storage = ContentAddressedStorage(database.data_dir)
        self.ingestion = IngestionService(self.repository, self.storage, database.data_dir)
        self.vector_maps = VectorMapService(database.data_dir)
        self.index_bundle_writer = IndexBundleWriter(database.data_dir / "index-bundles")
        self.embed_override = None
        self.rerank_override = None
        self.embedding_model_provider = None
        self.embedding_configured_provider = None
        self._retriever_cache: dict[tuple[object, ...], PublishedHybridRetriever] = {}
        self._image_retriever_cache: dict[tuple[object, ...], PublishedImageRetriever] = {}
        self._bundle_runtime_cache: dict[tuple[str, str], BundleRuntimeSnapshot] = {}
        self._bundle_runtime_lock = threading.RLock()
        self._offline_index_status: dict[str, dict[str, object]] = {}
        self._vector_map_executor = _DaemonBoundedExecutor("vector-map", max_workers=2)
        self._vector_map_futures: dict[tuple[str, str, str], Future[dict[str, object]]] = {}
        self._vector_map_lock = threading.RLock()
        self._vector_map_stale: dict[str, dict[str, object]] = {}
        self._vector_map_failures: dict[tuple[str, str, str], dict[str, object]] = {}
        self._vector_map_dataset_epochs: dict[str, int] = {}
        self._index_build_executor = _DaemonBoundedExecutor("index-bundle", max_workers=1)
        self._index_build_futures: dict[str, Future[object]] = {}
        self._index_build_lock = threading.RLock()

    def create_file(self, original_name: str, mime_type: str | None, content: bytes) -> FileAssetView:
        stored = self.storage.store(original_name, mime_type, content)
        record = self.repository.create_file(
            original_name=Path(original_name).name,
            content_hash=stored.content_hash,
            mime_type=stored.mime_type,
            size_bytes=stored.size_bytes,
            storage_path=stored.relative_path,
        )
        return FileAssetView.model_validate(record, from_attributes=True)

    def list_files(self) -> list[FileAssetView]:
        return [FileAssetView.model_validate(item, from_attributes=True) for item in self.repository.list_files()]

    def create_dataset(self, payload: DatasetCreate) -> DatasetView:
        record = self.repository.create_dataset(
            payload.name,
            description=payload.description,
            parser_profile=payload.parser_profile,
            visibility=payload.visibility,
        )
        return self.dataset_view(record.id)

    def list_datasets(self) -> list[DatasetView]:
        return [self.dataset_view(item.id) for item in self.repository.list_datasets()]

    def dataset_view(self, dataset_id: str) -> DatasetView:
        record = self.repository.get_dataset(dataset_id)
        metrics = self.repository.dataset_metrics(dataset_id)
        return DatasetView(
            id=record.id,
            name=record.name,
            description=record.description,
            parser_profile=record.default_parser_profile,
            visibility=record.visibility,
            published_version=record.published_version,
            status=record.status,
            is_system=record.is_system,
            retrieval_profile_id=record.retrieval_profile_id,
            created_at=record.created_at,
            updated_at=record.updated_at,
            **metrics,
        )

    def update_dataset(self, dataset_id: str, payload: DatasetUpdate) -> DatasetView:
        values = payload.model_dump(exclude_unset=True)
        profile_id = values.get("retrieval_profile_id")
        if profile_id:
            self.repository.get_retrieval_profile(str(profile_id))
        self.repository.update_dataset(dataset_id, **values)
        self._retriever_cache.clear()
        self._image_retriever_cache.clear()
        return self.dataset_view(dataset_id)

    def create_retrieval_profile(self, payload: RetrievalProfileCreate) -> RetrievalProfileView:
        record = self.repository.create_retrieval_profile(**payload.model_dump())
        return RetrievalProfileView.model_validate(record, from_attributes=True)

    def list_retrieval_profiles(self) -> list[RetrievalProfileView]:
        return [
            RetrievalProfileView.model_validate(item, from_attributes=True)
            for item in self.repository.list_retrieval_profiles()
        ]

    def update_retrieval_profile(
        self,
        profile_id: str,
        payload: RetrievalProfileUpdate,
    ) -> RetrievalProfileView:
        record = self.repository.update_retrieval_profile(
            profile_id,
            **payload.model_dump(exclude_unset=True),
        )
        self._retriever_cache.clear()
        return RetrievalProfileView.model_validate(record, from_attributes=True)

    def link_document(self, dataset_id: str, payload: DocumentLinkCreate) -> DocumentView:
        record = self.repository.link_file(
            dataset_id,
            payload.file_id,
            payload.parser_profile,
            payload.metadata,
        )
        return self.document_view(record.id)

    def list_documents(self, dataset_id: str) -> list[DocumentView]:
        return [self.document_view(item.id) for item in self.repository.list_document_refs(dataset_id=dataset_id)]

    def document_view(self, document_id: str) -> DocumentView:
        record = self.repository.get_document(document_id)
        file = self.repository.get_file(record.file_id)
        job = self.repository.latest_job(document_id)
        return DocumentView(
            id=record.id,
            dataset_id=record.dataset_id,
            file_id=record.file_id,
            original_name=file.original_name,
            mime_type=file.mime_type,
            parser_profile=record.parser_profile,
            enabled=record.enabled,
            active_version=record.active_version,
            published_version=record.published_version,
            latest_job_state=job.state if job else None,
            latest_job_progress=job.progress if job else None,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def parse_document(self, document_id: str) -> ParsingJobView:
        job = self.ingestion.parse_document(document_id)
        document = self.repository.get_document(document_id)
        dataset = self.repository.get_dataset(document.dataset_id)
        if job.index_version and self._embedding_is_configured(dataset):
            if self.embed_override is None:
                self._fail_embedding_job(job.id, "Embedding 模型已配置，但当前不可调用。")
            self.repository.update_job(
                job.id,
                state="running",
                stage="embedding",
                progress=80,
                error_code=None,
                error_message=None,
                finished_at=None,
            )
            children = self.repository.list_children(document_id=document_id, index_version=job.index_version)
            missing = [item for item in children if self._decode_embedding(item.embedding_json) is None]
            vectors: dict[str, list[float]] = {}
            try:
                for start in range(0, len(missing), 10):
                    batch = missing[start : start + 10]
                    returned = self.embed_override([item.text for item in batch])
                    if len(returned) != len(batch):
                        self._fail_embedding_job(job.id, "Embedding 返回数量不完整，候选版本未完成向量化。")
                    decoded = [self._decode_embedding(json.dumps(vector)) for vector in returned]
                    if any(vector is None for vector in decoded):
                        self._fail_embedding_job(job.id, "Embedding 返回了空向量或无效向量。")
                    vectors.update(
                        {
                            item.id: vector
                            for item, vector in zip(batch, decoded, strict=True)
                            if vector is not None
                        }
                    )
                    self.repository.update_job(
                        job.id,
                        stage="embedding",
                        progress=min(98, 80 + round((start + len(batch)) / max(1, len(missing)) * 18)),
                    )
            except HTTPException:
                raise
            except Exception:
                self._fail_embedding_job(job.id, "Embedding 调用失败，候选版本未完成向量化。")
            if vectors:
                self.repository.save_embeddings(document_id, job.index_version, vectors)
            embedded = self.repository.list_children(document_id=document_id, index_version=job.index_version)
            decoded_vectors = [self._decode_embedding(item.embedding_json) for item in embedded]
            dimensions = {len(vector) for vector in decoded_vectors if vector is not None}
            if any(vector is None for vector in decoded_vectors) or len(dimensions) > 1:
                self._fail_embedding_job(job.id, "候选版本仍有 Chunk 缺少有效 Embedding。")
            job = self.repository.update_job(
                job.id,
                state="succeeded",
                stage="reviewed",
                progress=100,
                error_code=None,
                error_message=None,
                finished_at=datetime.now(UTC),
            )
        self._retriever_cache.clear()
        self._image_retriever_cache.clear()
        return self.job_view(job.id)

    def _fail_embedding_job(self, job_id: str, message: str) -> None:
        self.repository.update_job(
            job_id,
            state="failed",
            stage="embedding",
            progress=100,
            error_code="EmbeddingIncomplete",
            error_message=message,
            finished_at=datetime.now(UTC),
        )
        raise HTTPException(status_code=502, detail=message)

    def job_view(self, job_id: str) -> ParsingJobView:
        record = self.repository.get_job(job_id)
        return ParsingJobView.model_validate(record, from_attributes=True)

    def start_index_build(self, dataset_id: str) -> IndexBuildView:
        dataset = self.repository.get_dataset(dataset_id)
        if not dataset.published_version:
            raise HTTPException(status_code=409, detail="知识库尚未发布，不能构建运行索引包。")
        job = self.repository.create_index_build(dataset_id, dataset.published_version)
        future = self._index_build_executor.submit(self._build_index_bundle, job.id)
        with self._index_build_lock:
            self._index_build_futures[job.id] = future
        return self.index_build_view(job.id)

    def index_build_view(self, build_id: str) -> IndexBuildView:
        return IndexBuildView.model_validate(
            self.repository.get_index_build(build_id),
            from_attributes=True,
        )

    def active_index_manifest(self, dataset_id: str) -> IndexManifest:
        self.repository.get_dataset(dataset_id)
        record = self.repository.active_index_manifest(dataset_id)
        if record is None:
            raise HTTPException(status_code=404, detail="知识库尚未生成活动索引 Manifest。")
        bundle = IndexBundle.load(self.database.data_dir / record.bundle_path)
        report = bundle.validate()
        if not report.valid:
            raise HTTPException(status_code=409, detail="活动索引包完整性校验失败。")
        return bundle.manifest

    def index_manifest(self, dataset_id: str, index_version: str) -> IndexManifest:
        self.repository.get_dataset(dataset_id)
        record = self.repository.get_index_manifest(dataset_id, index_version)
        bundle = IndexBundle.load(self.database.data_dir / record.bundle_path)
        report = bundle.validate()
        if not report.valid:
            raise HTTPException(status_code=409, detail="索引包完整性校验失败。")
        return bundle.manifest

    def activate_previous_index_manifest(
        self,
        dataset_id: str,
        index_version: str,
    ) -> IndexManifest:
        self.repository.get_dataset(dataset_id)
        record = self.repository.get_index_manifest(dataset_id, index_version)
        bundle = IndexBundle.load(self.database.data_dir / record.bundle_path)
        report = bundle.validate()
        if not report.valid:
            raise HTTPException(status_code=409, detail="目标索引包完整性校验失败，不能回滚。")
        try:
            snapshot = load_bundle_runtime(bundle.root)
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail="目标索引包无法加载为运行索引，不能回滚。",
            ) from exc
        self.repository.activate_index_manifest(dataset_id, index_version)
        self._retriever_cache.clear()
        self._image_retriever_cache.clear()
        self._cache_bundle_snapshot(dataset_id, snapshot)
        return bundle.manifest

    def activate_candidate_index_manifest(
        self,
        dataset_id: str,
        index_version: str,
    ) -> IndexManifest:
        self.repository.get_dataset(dataset_id)
        record = self.repository.get_index_manifest(dataset_id, index_version)
        bundle = IndexBundle.load(self.database.data_dir / record.bundle_path)
        try:
            snapshot = load_bundle_runtime(bundle.root)
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail="候选索引包无法加载为运行索引，不能激活。",
            ) from exc
        self.repository.activate_index_manifest(
            dataset_id,
            index_version,
            require_previously_active=False,
        )
        self._retriever_cache.clear()
        self._image_retriever_cache.clear()
        self._cache_bundle_snapshot(dataset_id, snapshot)
        return bundle.manifest

    def _build_index_bundle(self, build_id: str) -> None:
        job = self.repository.get_index_build(build_id)
        self.repository.update_index_build(
            build_id,
            state="running",
            stage="snapshot",
            progress=5,
            started_at=datetime.now(UTC),
            error_code=None,
            error_message=None,
        )
        try:
            dataset = self.repository.get_dataset(job.dataset_id)
            if not dataset.published_version:
                raise ValueError("dataset-not-published")
            documents = [
                item
                for item in self.repository.list_document_refs(dataset_id=job.dataset_id)
                if item.enabled and item.published_version
            ]
            sources: list[SourceManifest] = []
            for document in documents:
                file = self.repository.get_file(document.file_id)
                parse_job = self.repository.latest_job(document.id)
                sources.append(
                    SourceManifest(
                        document_id=document.id,
                        file_id=file.id,
                        source_name=file.original_name,
                        source_sha256=file.content_hash,
                        mime_type=file.mime_type,
                        size_bytes=file.size_bytes,
                        parser_fingerprint=(
                            parse_job.config_hash
                            if parse_job and parse_job.config_hash
                            else f"{document.parser_profile}:{document.published_version}"
                        ),
                        document_version=document.published_version,
                    )
                )
            self.repository.update_index_build(build_id, stage="materializing", progress=25)
            parents = self.repository.list_dataset_parents(job.dataset_id, published_only=True)
            children = self.repository.list_children(dataset_ids=[job.dataset_id], published_only=True)
            assets = self.repository.list_dataset_assets(job.dataset_id, published_only=True)
            image_chunks = self.repository.list_image_chunks(
                dataset_id=job.dataset_id,
                published_only=True,
            )
            parent_rows = [
                {
                    "id": item.id,
                    "chunk_type": "parent",
                    "dataset_id": item.dataset_id,
                    "document_id": item.document_id,
                    "document_version": item.index_version,
                    "title": item.title,
                    "heading_path": json.loads(item.heading_path_json),
                    "text": item.text,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "product": item.product,
                    "content_hash": item.content_hash,
                }
                for item in parents
            ]
            child_rows = [
                {
                    "id": item.id,
                    "chunk_type": "child",
                    "parent_id": item.parent_id,
                    "dataset_id": item.dataset_id,
                    "document_id": item.document_id,
                    "document_version": item.index_version,
                    "title": item.title,
                    "text": item.text,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "keywords": json.loads(item.keywords_json),
                    "questions": json.loads(item.questions_json),
                    "tags": json.loads(item.tags_json),
                    "asset_ids": json.loads(item.asset_ids_json),
                    "product": item.product,
                    "content_hash": item.content_hash,
                }
                for item in children
            ]
            text_rows = [*parent_rows, *child_rows]
            asset_rows = [
                {
                    "id": item.id,
                    "dataset_id": item.dataset_id,
                    "document_id": item.document_id,
                    "document_version": item.index_version,
                    "asset_type": item.asset_type,
                    "page_number": item.page_number,
                    "storage_path": item.storage_path,
                    "caption": item.caption,
                    "ocr_text": item.ocr_text,
                    "related_parent_ids": json.loads(item.related_parent_ids_json),
                    "related_child_ids": json.loads(item.related_child_ids_json),
                }
                for item in assets
            ]
            image_rows = [
                {
                    "id": item.id,
                    "dataset_id": item.dataset_id,
                    "document_id": item.document_id,
                    "document_version": item.index_version,
                    "asset_id": item.asset_id,
                    "image_id": item.image_id,
                    "manual_name": item.manual_name,
                    "chapter_title": item.chapter_title,
                    "page_number": item.page_number,
                    "caption": item.caption,
                    "ocr_text": item.ocr_text,
                    "visible_text": json.loads(item.visible_text_json),
                    "visual_summary": item.visual_summary,
                    "visual_meaning": item.visual_meaning,
                    "retrieval_text": item.retrieval_text,
                    "search_terms": json.loads(item.search_terms_json),
                    "applicable_questions": json.loads(item.applicable_questions_json),
                    "issue_signals": json.loads(item.issue_signals_json),
                    "related_parent_ids": json.loads(item.related_parent_ids_json),
                    "related_child_ids": json.loads(item.related_child_ids_json),
                    "confidence": item.confidence,
                    "content_hash": item.content_hash,
                }
                for item in image_chunks
            ]
            text_vectors = {
                item.id: vector
                for item in children
                if (vector := self._decode_embedding(item.embedding_json)) is not None
            }
            image_caption_vectors = {
                item.id: vector
                for item in image_chunks
                if (vector := self._decode_embedding(item.embedding_json)) is not None
            }
            dimensions = {
                len(vector)
                for vector in [*text_vectors.values(), *image_caption_vectors.values()]
            }
            if len(dimensions) > 1:
                raise ValueError("embedding-dimension-mismatch")
            vector_dimension = next(iter(dimensions), 0)
            embedding_model = (
                self._embedding_model_for_dataset(dataset)
                if text_vectors or image_caption_vectors
                else None
            )
            active = self.repository.active_index_manifest(job.dataset_id)
            previous_manifest = (
                IndexManifest.model_validate_json(active.manifest_json) if active is not None else None
            )
            incremental_plan = plan_incremental_build(
                previous_manifest.sources if previous_manifest else [],
                sources,
            )
            incremental = IncrementalStats(
                reused=len(incremental_plan.reuse),
                added=len(incremental_plan.add),
                updated=len(incremental_plan.update),
                deleted=len(incremental_plan.delete),
            )
            content_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "bundle_schema": "aka-index-bundle-v2",
                        "parser_version": IngestionService.parser_version,
                        "embedding_model": embedding_model,
                        "vector_dimension": vector_dimension,
                        "sources": [
                            [item.document_id, item.source_sha256, item.document_version]
                            for item in sorted(sources, key=lambda value: value.document_id)
                        ],
                        "parents": sorted([item.content_hash for item in parents]),
                        "children": sorted([item.content_hash for item in children]),
                        "images": sorted([item.content_hash for item in image_chunks]),
                        "assets": sorted([item.id for item in assets]),
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:12]
            bundle_version = f"{dataset.published_version}-{content_fingerprint}"
            self.repository.update_index_build(build_id, stage="writing", progress=60)
            bundle = self.index_bundle_writer.write(
                dataset_id=job.dataset_id,
                index_version=bundle_version,
                parent_index_version=previous_manifest.index_version if previous_manifest else None,
                parser_version=IngestionService.parser_version,
                embedding_model=embedding_model,
                vector_dimension=vector_dimension,
                sources=sources,
                text_chunks=text_rows,
                image_chunks=image_rows,
                assets=asset_rows,
                text_vectors=text_vectors,
                image_caption_vectors=image_caption_vectors,
                incremental=incremental,
                evaluation_status="not_run",
                approval_status="awaiting_approval",
            )
            self.repository.update_index_build(build_id, stage="validating", progress=85)
            report = bundle.validate()
            if not report.valid:
                raise ValueError("index-bundle-validation-failed")
            relative_bundle_path = str(bundle.root.relative_to(self.database.data_dir))
            activate_bundle = self._index_bundle_can_activate_without_gate(
                dataset,
                active,
                bundle,
            )
            self.repository.save_index_manifest(
                dataset_id=job.dataset_id,
                index_version=bundle.manifest.index_version,
                bundle_path=relative_bundle_path,
                manifest_json=bundle.manifest.model_dump_json(),
                validation_status="valid",
                active=activate_bundle,
            )
            if activate_bundle:
                self._retriever_cache.clear()
                self._image_retriever_cache.clear()
                self._load_active_bundle(job.dataset_id, refresh=True)
            self.repository.update_index_build(
                build_id,
                state="succeeded",
                stage="validated" if activate_bundle else "awaiting_approval",
                progress=100,
                index_version=bundle.manifest.index_version,
                bundle_path=relative_bundle_path,
                finished_at=datetime.now(UTC),
            )
        except Exception as exc:
            self.repository.update_index_build(
                build_id,
                state="failed",
                stage="failed",
                progress=100,
                error_code=exc.__class__.__name__,
                error_message=str(exc)[:500],
                finished_at=datetime.now(UTC),
            )
        finally:
            with self._index_build_lock:
                self._index_build_futures.pop(build_id, None)

    def _index_bundle_can_activate_without_gate(
        self,
        dataset,
        active_record,
        candidate: IndexBundle,
    ) -> bool:
        if active_record is None or not dataset.is_system:
            return True
        try:
            active_bundle = IndexBundle.load(self.database.data_dir / active_record.bundle_path)
            if not active_bundle.validate().valid:
                return False
        except Exception:
            return False
        behavior_artifacts = (
            "text_chunks.jsonl",
            "image_chunks.jsonl",
            "assets.jsonl",
            "text_vectors.npz",
            "image_caption_vectors.npz",
        )
        return all(
            active_bundle.manifest.artifacts.get(name) is not None
            and candidate.manifest.artifacts.get(name) is not None
            and active_bundle.manifest.artifacts[name].sha256
            == candidate.manifest.artifacts[name].sha256
            for name in behavior_artifacts
        )

    def publish(self, dataset_id: str, index_version: str) -> DatasetView:
        dataset = self.repository.get_dataset(dataset_id)
        previous_version = dataset.published_version
        if self._embedding_is_configured(dataset):
            candidates = self.repository.list_children(
                dataset_ids=[dataset_id],
                index_version=index_version,
            )
            decoded = [self._decode_embedding(item.embedding_json) for item in candidates]
            dimensions = {len(vector) for vector in decoded if vector is not None}
            if candidates and (any(vector is None for vector in decoded) or len(dimensions) > 1):
                raise HTTPException(
                    status_code=409,
                    detail="候选版本存在缺失或无效的 Embedding，请重新解析后再发布。",
                )
        self.repository.publish_dataset(dataset_id, index_version)
        self._retriever_cache.clear()
        self._image_retriever_cache.clear()
        self._invalidate_vector_maps_for_publish(dataset_id, previous_version, index_version)
        return self.dataset_view(dataset_id)

    def shutdown(self) -> None:
        with self._vector_map_lock:
            futures = list(self._vector_map_futures.values())
            self._vector_map_futures.clear()
        for future in futures:
            future.cancel()
        self._vector_map_executor.shutdown(cancel_pending=True)
        with self._index_build_lock:
            index_futures = list(self._index_build_futures.values())
            self._index_build_futures.clear()
        for future in index_futures:
            future.cancel()
        self._index_build_executor.shutdown(cancel_pending=True)

    def chunks(self, document_id: str) -> ChunkCollectionView:
        document = self.repository.get_document(document_id)
        parents = self.repository.list_parents(document_id, document.active_version)
        children = self.repository.list_children(document_id=document_id, index_version=document.active_version)
        return ChunkCollectionView(
            parents=[self._parent_view(item) for item in parents],
            children=[self._child_view(item) for item in children],
        )

    def image_chunks(
        self,
        dataset_id: str,
        *,
        document_id: str | None = None,
        query: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ImageChunkView]:
        self.repository.get_dataset(dataset_id)
        records = self.repository.list_image_chunks(
            dataset_id=dataset_id,
            document_id=document_id,
            published_only=True,
        )
        if query and query.strip():
            tokens = set(query.lower().split())
            records = [
                item
                for item in records
                if any(
                    token in (
                        f"{item.manual_name} {item.chapter_title} {item.caption} "
                        f"{item.ocr_text} {item.retrieval_text} {item.search_terms_json}"
                    ).lower()
                    for token in tokens
                )
            ]
        return [self._image_chunk_view(item) for item in records[offset : offset + limit]]

    def image_chunk(self, image_chunk_id: str) -> ImageChunkView:
        return self._image_chunk_view(self.repository.get_image_chunk(image_chunk_id))

    def edit_child(self, child_id: str, **changes: object) -> ChildChunkView:
        self._retriever_cache.clear()
        self._image_retriever_cache.clear()
        return self._child_view(self.repository.edit_child(child_id, **changes))

    def retriever(
        self,
        dataset_ids: list[str] | None = None,
        profile_id: str | None = None,
    ) -> PublishedHybridRetriever:
        selected = [item for item in self.repository.list_datasets() if not dataset_ids or item.id in dataset_ids]
        selected_profile_ids = {item.retrieval_profile_id for item in selected if item.retrieval_profile_id}
        resolved_profile_id = profile_id or (next(iter(selected_profile_ids)) if len(selected_profile_ids) == 1 else None)
        profile = self.repository.get_retrieval_profile(resolved_profile_id) if resolved_profile_id else None
        bundle_snapshots, database_dataset_ids, source_identity = self._runtime_bundle_sources(
            selected
        )
        cache_key = (
            *source_identity,
            ("profile", resolved_profile_id, profile.updated_at.isoformat() if profile else None),
        )
        cached = self._retriever_cache.get(cache_key)
        if cached is not None:
            return cached
        documents = [
            child
            for snapshot in bundle_snapshots
            for child in snapshot.children
        ]
        if database_dataset_ids:
            documents.extend(self._indexed_children(database_dataset_ids))
        retriever = PublishedHybridRetriever(
            documents,
            embed=self.embed_override,
            rerank=self.rerank_override,
            min_score=profile.min_score if profile else 0.012,
            rrf_k=profile.rrf_k if profile else 60,
            lexical_top_k=profile.lexical_top_k if profile else 20,
            dense_top_k=profile.dense_top_k if profile else 20,
            rerank_top_k=profile.rerank_top_k if profile else 12,
        )
        self._retriever_cache[cache_key] = retriever
        return retriever

    def candidate_retriever(
        self,
        dataset_ids: list[str],
        candidate_version: str,
    ) -> PublishedHybridRetriever:
        published = self.repository.list_children(dataset_ids=dataset_ids, published_only=True)
        candidate = self.repository.list_children(
            dataset_ids=dataset_ids,
            index_version=candidate_version,
        )
        candidate_document_ids = {item.document_id for item in candidate}
        records = [item for item in published if item.document_id not in candidate_document_ids]
        records.extend(candidate)
        return PublishedHybridRetriever(
            self._indexed_records(records),
            embed=self.embed_override,
            rerank=self.rerank_override,
        )

    def image_retriever(
        self,
        dataset_ids: list[str] | None = None,
    ) -> PublishedImageRetriever:
        selected = [
            item for item in self.repository.list_datasets() if not dataset_ids or item.id in dataset_ids
        ]
        bundle_snapshots, database_dataset_ids, source_identity = self._runtime_bundle_sources(
            selected
        )
        cache_key = source_identity
        cached = self._image_retriever_cache.get(cache_key)
        if cached is not None:
            return cached
        images = [
            image
            for snapshot in bundle_snapshots
            for image in snapshot.images
        ]
        records = self.repository.list_image_chunks(published_only=True) if database_dataset_ids else []
        allowed = set(database_dataset_ids)
        images.extend(self._indexed_images([item for item in records if item.dataset_id in allowed]))
        retriever = PublishedImageRetriever(
            images,
            embed=(
                self.embed_override
                if self.settings.is_enabled(self.settings.caption_embedding)
                else None
            ),
            rerank=self.rerank_override,
        )
        self._image_retriever_cache[cache_key] = retriever
        return retriever

    def preload_active_bundles(self) -> dict[str, dict[str, object]]:
        if self.settings.offline_index_mode == "off":
            return {}
        for dataset in self.repository.list_datasets():
            if dataset.published_version:
                self._load_active_bundle(dataset.id)
        return self.offline_index_status()

    def offline_index_status(self) -> dict[str, dict[str, object]]:
        with self._bundle_runtime_lock:
            return {
                dataset_id: dict(payload)
                for dataset_id, payload in self._offline_index_status.items()
            }

    def _runtime_bundle_sources(
        self,
        selected,
    ) -> tuple[list[BundleRuntimeSnapshot], list[str], tuple[tuple[object, ...], ...]]:
        snapshots: list[BundleRuntimeSnapshot] = []
        database_dataset_ids: list[str] = []
        identities: list[tuple[object, ...]] = []
        mode = self.settings.offline_index_mode
        for dataset in selected:
            snapshot = self._load_active_bundle(dataset.id) if mode != "off" else None
            if snapshot is not None and mode == "on":
                snapshots.append(snapshot)
                identities.append((dataset.id, "bundle", snapshot.manifest.index_version))
            else:
                database_dataset_ids.append(dataset.id)
                identities.append((dataset.id, "database", dataset.published_version))
        return snapshots, database_dataset_ids, tuple(sorted(identities))

    def _load_active_bundle(
        self,
        dataset_id: str,
        *,
        refresh: bool = False,
    ) -> BundleRuntimeSnapshot | None:
        record = self.repository.active_index_manifest(dataset_id)
        if record is None:
            with self._bundle_runtime_lock:
                self._offline_index_status[dataset_id] = {
                    "status": "missing",
                    "mode": self.settings.offline_index_mode,
                    "message": "知识库尚无活动离线 Index Bundle，当前使用数据库发布版本。",
                }
            return None
        key = (dataset_id, record.index_version)
        with self._bundle_runtime_lock:
            if not refresh and key in self._bundle_runtime_cache:
                return self._bundle_runtime_cache[key]
        try:
            snapshot = load_bundle_runtime(self.database.data_dir / record.bundle_path)
            if snapshot.manifest.dataset_id != dataset_id:
                raise ValueError("offline index dataset id does not match active manifest")
        except Exception as exc:
            with self._bundle_runtime_lock:
                self._offline_index_status[dataset_id] = {
                    "status": "failed",
                    "mode": self.settings.offline_index_mode,
                    "index_version": record.index_version,
                    "error_code": "index_load_failed",
                    "message": str(exc)[:300],
                }
            return None
        self._cache_bundle_snapshot(dataset_id, snapshot)
        return snapshot

    def _cache_bundle_snapshot(
        self,
        dataset_id: str,
        snapshot: BundleRuntimeSnapshot,
    ) -> None:
        key = (dataset_id, snapshot.manifest.index_version)
        with self._bundle_runtime_lock:
            stale_keys = [item for item in self._bundle_runtime_cache if item[0] == dataset_id]
            for stale_key in stale_keys:
                self._bundle_runtime_cache.pop(stale_key, None)
            self._bundle_runtime_cache[key] = snapshot
            self._offline_index_status[dataset_id] = {
                "status": "ready",
                "mode": self.settings.offline_index_mode,
                "index_version": snapshot.manifest.index_version,
                "approval_status": snapshot.manifest.approval_status,
                "child_chunks": len(snapshot.children),
                "image_chunks": len(snapshot.images),
            }

    def asset_path(self, asset_id: str) -> Path:
        asset = self.repository.get_asset(asset_id)
        path = (self.database.data_dir / asset.storage_path).resolve()
        if self.database.data_dir.resolve() not in path.parents or not path.exists():
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="证据资产文件不存在")
        return path

    def file_content(self, file_id: str) -> tuple[Path, str, str]:
        record = self.repository.get_file(file_id)
        return (
            self.storage.resolve(record.storage_path),
            record.original_name,
            record.mime_type,
        )

    def retrieval_test(self, payload: RetrievalTestRequest) -> dict[str, object]:
        retriever = self.retriever(payload.dataset_ids, payload.profile_id)
        query_vector: list[float] | None = None
        query_vector_supplied = False
        if (
            len(payload.dataset_ids) == 1
            and self.embed_override is not None
            and any(item.embedding for item in retriever.documents)
        ):
            try:
                returned = self.embed_override([payload.query])
            except Exception:
                returned = []
            query_vector = list(returned[0]) if len(returned) == 1 and returned[0] else []
            query_vector_supplied = True

        result = retriever.explain(
            payload.query,
            dataset_ids=payload.dataset_ids,
            top_n=payload.top_n,
            min_score=payload.min_score,
            use_rerank=payload.use_rerank,
            query_vector=query_vector if query_vector_supplied else None,
        )
        response = asdict(result)
        response["visualization"] = self._retrieval_visualization(
            payload.dataset_ids[0] if len(payload.dataset_ids) == 1 else None,
            query_vector if query_vector_supplied and query_vector else None,
            result.stages,
        )
        return response

    def _retrieval_visualization(
        self,
        dataset_id: str | None,
        query_vector: list[float] | None,
        stages: dict[str, list[dict[str, object]]],
    ) -> dict[str, object] | None:
        if dataset_id is None:
            return None
        try:
            dataset = self.repository.get_dataset(dataset_id)
            published_version = dataset.published_version
            if not published_version:
                return None
            embedding_model = self._embedding_model_for_dataset(dataset)
            state = self.vector_maps.status(dataset_id, published_version, embedding_model)
            if state.get("status") != "ready":
                return None
            transformed_query = None
            status = "ready"
            message = None
            if query_vector:
                transformed = self.vector_maps.transform_query(
                    dataset_id,
                    published_version,
                    embedding_model,
                    query_vector,
                )
                if (
                    isinstance(transformed, dict)
                    and isinstance(transformed.get("x"), int | float)
                    and isinstance(transformed.get("y"), int | float)
                ):
                    transformed_query = {
                        "x": float(transformed["x"]),
                        "y": float(transformed["y"]),
                    }
                else:
                    status = "query_transform_failed"
                    message = "查询向量投影失败，仍显示 RRF 命中。"
            else:
                status = "unavailable"
                message = "查询向量不可用，仍显示 BM25/RRF 命中。"
            meta = state.get("meta") if isinstance(state.get("meta"), dict) else {}
            digest = str(meta.get("content_digest") or "unknown")
            return {
                "projection_version": f"{published_version}:{digest[:12]}",
                "query": transformed_query,
                "dense_top10": self._visualization_hits(stages.get("dense", [])),
                "rerank_top10": self._visualization_hits(stages.get("rerank", [])),
                "rrf_top10": self._visualization_hits(stages.get("rrf", [])),
                "status": status,
                "message": message,
            }
        except Exception:
            return None

    @staticmethod
    def _visualization_hits(items: list[dict[str, object]]) -> list[dict[str, object]]:
        hits: list[dict[str, object]] = []
        for rank, item in enumerate(items[:10], start=1):
            child_id = item.get("id")
            if not isinstance(child_id, str) or not child_id:
                continue
            score = item.get("score", 0.0)
            hits.append(
                {
                    "child_id": child_id,
                    "rank": rank,
                    "score": float(score) if isinstance(score, int | float) else 0.0,
                }
            )
        return hits

    def vector_map(self, dataset_id: str) -> dict[str, object]:
        dataset = self.repository.get_dataset(dataset_id)
        published_version = dataset.published_version
        if not published_version:
            return {
                "status": "no_published_version",
                "message": "知识库尚未发布，无法生成向量图。",
            }

        embedding_model = self._embedding_model_for_dataset(dataset)
        key = (dataset_id, published_version, embedding_model)
        if self._vector_map_future_active(key):
            return self._vector_map_progress_payload(dataset_id, published_version)

        failure = self._vector_map_failure(key)
        if failure is not None:
            return failure

        current = self.vector_maps.status(dataset_id, published_version, embedding_model)
        if current.get("status") == "ready":
            payload = self.vector_maps.load(dataset_id, published_version, embedding_model)
            if payload.get("status") == "ready":
                self._clear_vector_map_stale(dataset_id, published_version)
            return payload
        if current.get("status") == "building":
            return self._vector_map_progress_payload(dataset_id, published_version)
        if current.get("status") == "failed":
            return current
        if current.get("status") not in {"missing", None}:
            return current

        if not self.repository.has_vector_map_sources(dataset_id, published_version):
            return {
                "status": "no_embeddings",
                "message": "当前知识库尚未完成向量化。",
            }
        self._ensure_vector_map_build(key)
        failure = self._vector_map_failure(key)
        if failure is not None:
            return failure
        return self._vector_map_progress_payload(dataset_id, published_version)

    def rebuild_vector_map(self, dataset_id: str) -> dict[str, object]:
        dataset = self.repository.get_dataset(dataset_id)
        published_version = dataset.published_version
        if not published_version:
            return {
                "status": "no_published_version",
                "message": "知识库尚未发布，无法重建向量图。",
            }

        embedding_model = self._embedding_model_for_dataset(dataset)
        key = (dataset_id, published_version, embedding_model)
        if self._vector_map_future_active(key):
            return self._vector_map_progress_payload(dataset_id, published_version)
        current = self.vector_maps.status(dataset_id, published_version, embedding_model)
        if current.get("status") == "building":
            return self._vector_map_progress_payload(dataset_id, published_version)
        if not self.repository.has_vector_map_sources(dataset_id, published_version):
            return {
                "status": "no_embeddings",
                "message": "当前知识库尚未完成向量化。",
            }
        invalidated = self.vector_maps.invalidate(
            dataset_id,
            published_version,
            embedding_model,
            remove_cache=False,
        )
        if invalidated.get("status") == "failed":
            return invalidated
        self._clear_vector_map_failure(key)
        self._ensure_vector_map_build(key)
        failure = self._vector_map_failure(key)
        if failure is not None:
            return failure
        return self._vector_map_progress_payload(dataset_id, published_version)

    def _indexed_children(self, dataset_ids: list[str] | None) -> list[IndexedChild]:
        records = self.repository.list_children(dataset_ids=dataset_ids, published_only=True)
        return self._indexed_records(records)

    def _vector_map_sources(self, dataset_id: str, published_version: str) -> list[VectorSource]:
        records = self.repository.list_children(
            dataset_ids=[dataset_id],
            published_only=True,
        )
        documents = {}
        files = {}
        sources: list[VectorSource] = []
        for record in records:
            embedding = self._decode_embedding(record.embedding_json)
            if embedding is None:
                continue
            document = documents.get(record.document_id)
            if document is None:
                document = self.repository.get_document(record.document_id)
                documents[record.document_id] = document
            file = files.get(document.file_id)
            if file is None:
                file = self.repository.get_file(document.file_id)
                files[document.file_id] = file
            sources.append(
                VectorSource(
                    child_id=record.id,
                    dataset_id=record.dataset_id,
                    document_id=record.document_id,
                    document_name=file.original_name,
                    title=record.title,
                    excerpt=record.text[:180],
                    page_start=record.page_start,
                    page_end=record.page_end,
                    product=record.product,
                    embedding=embedding,
                    content_hash=record.content_hash,
                )
            )
        return sources

    @staticmethod
    def _decode_embedding(raw: str | None) -> list[float] | None:
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(value, list) or not value:
            return None
        vector: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int | float):
                return None
            number = float(item)
            if not math.isfinite(number):
                return None
            vector.append(number)
        return vector

    def _embedding_model_for_dataset(self, dataset) -> str:
        if dataset.embedding_model_id:
            return dataset.embedding_model_id
        provider = self.embedding_model_provider
        if provider:
            try:
                value = provider()
            except Exception:
                value = None
            if value:
                return str(value)
        return "embedding-unconfigured"

    def _embedding_is_configured(self, dataset) -> bool:
        if dataset.embedding_model_id:
            return True
        provider = self.embedding_configured_provider
        if provider is None:
            return False
        try:
            return bool(provider())
        except Exception:
            return False

    def _ensure_vector_map_build(
        self,
        key: tuple[str, str, str],
        sources: list[VectorSource] | None = None,
    ) -> bool:
        dataset_id = key[0]
        with self._vector_map_lock:
            current = self._vector_map_futures.get(key)
            if current is not None and not current.done():
                return False
            dataset_epoch = self._vector_map_dataset_epochs.get(dataset_id, 0)
            prepared_sources = tuple(sources) if sources is not None else None
            future = self._vector_map_executor.submit(self._build_vector_map, key, prepared_sources, dataset_epoch)
            self._vector_map_futures[key] = future
            future.add_done_callback(lambda done, build_key=key: self._finish_vector_map_build(build_key, done))
        return not future.done()

    def _build_vector_map(
        self,
        key: tuple[str, str, str],
        sources: tuple[VectorSource, ...] | None,
        dataset_epoch: int,
    ) -> dict[str, object]:
        dataset_id, published_version, embedding_model = key
        try:
            source_values = (
                list(sources)
                if sources is not None
                else self._vector_map_sources(dataset_id, published_version)
            )
            if not source_values:
                return {
                    "status": "no_embeddings",
                    "message": "当前知识库尚未完成向量化。",
                }
            payload = self.vector_maps.build(dataset_id, published_version, embedding_model, source_values)
        except BaseException as exc:
            return self.vector_maps.record_failure(
                dataset_id,
                published_version,
                embedding_model,
                "projection_failed",
                str(exc),
            )
        if payload.get("status") == "ready":
            with self._vector_map_lock:
                current_epoch = self._vector_map_dataset_epochs.get(dataset_id, 0)
            if current_epoch == dataset_epoch:
                self._clear_vector_map_stale(dataset_id, published_version)
        return payload

    def _finish_vector_map_build(
        self,
        key: tuple[str, str, str],
        future: Future[dict[str, object]],
    ) -> None:
        try:
            payload = future.result()
        except CancelledError:
            return
        except _VectorMapExecutorShutdown as exc:
            payload = self.vector_maps.record_failure(
                key[0],
                key[1],
                key[2],
                "executor_shutdown",
                str(exc),
            )
        except BaseException as exc:
            payload = self.vector_maps.record_failure(
                key[0],
                key[1],
                key[2],
                "projection_failed",
                str(exc),
            )

        status = payload.get("status")
        with self._vector_map_lock:
            if self._vector_map_futures.get(key) is not future:
                return
            self._vector_map_futures.pop(key, None)
            if status == "ready":
                self._vector_map_failures.pop(key, None)
            elif status == "failed":
                error = payload.get("error")
                code = error.get("code") if isinstance(error, dict) else None
                if code == "build_cancelled":
                    return
                self._vector_map_failures[key] = json.loads(json.dumps(payload, ensure_ascii=False))
        if status == "ready":
            self._clear_vector_map_stale(key[0], key[1])

    def _vector_map_future_active(self, key: tuple[str, str, str]) -> bool:
        with self._vector_map_lock:
            future = self._vector_map_futures.get(key)
        if future is None:
            return False
        if not future.done():
            return True
        self._finish_vector_map_build(key, future)
        return False

    def _invalidate_vector_maps_for_publish(
        self,
        dataset_id: str,
        previous_version: str | None,
        current_version: str,
    ) -> None:
        dataset = self.repository.get_dataset(dataset_id)
        embedding_model = self._embedding_model_for_dataset(dataset)
        with self._vector_map_lock:
            self._vector_map_dataset_epochs[dataset_id] = self._vector_map_dataset_epochs.get(dataset_id, 0) + 1
            for key in [key for key in self._vector_map_failures if key[0] == dataset_id]:
                self._vector_map_failures.pop(key, None)
        self._cancel_vector_map_futures_for_dataset(dataset_id)
        if previous_version:
            previous_payload = self.vector_maps.load(dataset_id, previous_version, embedding_model)
            self._mark_vector_map_stale(
                dataset_id,
                previous_version,
                current_version,
                embedding_model,
                previous_payload if previous_payload.get("status") == "ready" else None,
            )
        self.vector_maps.invalidate(
            dataset_id,
            current_version,
            embedding_model,
            remove_cache=False,
        )

    def _cancel_vector_map_futures_for_dataset(self, dataset_id: str) -> None:
        with self._vector_map_lock:
            items = [
                (key, future)
                for key, future in list(self._vector_map_futures.items())
                if key[0] == dataset_id and not future.done()
            ]
            for key, _ in items:
                self._vector_map_futures.pop(key, None)
        for _, future in items:
            future.cancel()

    def _vector_map_failure(self, key: tuple[str, str, str]) -> dict[str, object] | None:
        with self._vector_map_lock:
            failure = self._vector_map_failures.get(key)
        if failure is None:
            return None
        return json.loads(json.dumps(failure, ensure_ascii=False))

    def _clear_vector_map_failure(self, key: tuple[str, str, str]) -> None:
        with self._vector_map_lock:
            self._vector_map_failures.pop(key, None)

    def _mark_vector_map_stale(
        self,
        dataset_id: str,
        previous_version: str,
        current_version: str,
        embedding_model: str,
        previous_payload: dict[str, object] | None = None,
    ) -> None:
        with self._vector_map_lock:
            self._vector_map_stale[dataset_id] = {
                "dataset_id": dataset_id,
                "previous_published_version": previous_version,
                "current_published_version": current_version,
                "embedding_model": embedding_model,
                "previous_payload": previous_payload,
            }

    def _clear_vector_map_stale(self, dataset_id: str, published_version: str) -> None:
        with self._vector_map_lock:
            stale = self._vector_map_stale.get(dataset_id)
            if stale and stale.get("current_published_version") == published_version:
                self._vector_map_stale.pop(dataset_id, None)

    def _vector_map_progress_payload(
        self,
        dataset_id: str,
        published_version: str,
    ) -> dict[str, object]:
        with self._vector_map_lock:
            stale = self._vector_map_stale.get(dataset_id)
        if stale and stale.get("current_published_version") == published_version:
            previous_payload = stale.get("previous_payload")
            if not isinstance(previous_payload, dict):
                previous_version = stale.get("previous_published_version")
                embedding_model = stale.get("embedding_model")
                if isinstance(previous_version, str) and isinstance(embedding_model, str):
                    loaded = self.vector_maps.load(dataset_id, previous_version, embedding_model)
                    previous_payload = loaded if loaded.get("status") == "ready" else None
            if isinstance(previous_payload, dict):
                previous_meta = previous_payload.get("meta")
                meta = dict(previous_meta) if isinstance(previous_meta, dict) else {}
                meta.update(
                    {
                        "stale": True,
                        "target_published_version": published_version,
                    }
                )
                points = previous_payload.get("points")
                return {
                    "status": "stale",
                    "message": "正在为当前发布版本重建，暂时显示上一版向量图。",
                    "meta": meta,
                    "points": points if isinstance(points, list) else [],
                }
            return {
                "status": "stale",
                "message": "向量图已过期，正在为当前发布版本重建。",
                "meta": {
                    "dataset_id": dataset_id,
                    "previous_published_version": stale.get("previous_published_version"),
                    "target_published_version": published_version,
                    "stale": True,
                },
            }
        return {"status": "building", "message": "向量图正在生成。"}

    def _indexed_records(self, records) -> list[IndexedChild]:
        parent_cache = {}
        document_cache = {}
        file_cache = {}
        values = []
        for record in records:
            parent = parent_cache.get(record.parent_id)
            if parent is None:
                parent = self.repository.get_parent(record.parent_id)
                parent_cache[record.parent_id] = parent
            document = document_cache.get(record.document_id)
            if document is None:
                document = self.repository.get_document(record.document_id)
                document_cache[record.document_id] = document
            file = file_cache.get(document.file_id)
            if file is None:
                file = self.repository.get_file(document.file_id)
                file_cache[document.file_id] = file
            values.append(
                IndexedChild(
                    child_id=record.id,
                    parent_id=record.parent_id,
                    dataset_id=record.dataset_id,
                    document_id=record.document_id,
                    document_version=record.index_version,
                    file_id=file.id,
                    document_name=file.original_name,
                    document_mime_type=file.mime_type,
                    title=record.title,
                    text=record.text,
                    parent_text=parent.text,
                    product=record.product,
                    page_start=record.page_start,
                    page_end=record.page_end,
                    asset_ids=json.loads(record.asset_ids_json),
                    keywords=json.loads(record.keywords_json),
                    embedding=json.loads(record.embedding_json) if record.embedding_json else None,
                )
            )
        return values

    def _indexed_images(self, records) -> list[IndexedImageChunk]:
        document_cache = {}
        file_cache = {}
        values: list[IndexedImageChunk] = []
        for record in records:
            document = document_cache.get(record.document_id)
            if document is None:
                document = self.repository.get_document(record.document_id)
                document_cache[record.document_id] = document
            file = file_cache.get(document.file_id)
            if file is None:
                file = self.repository.get_file(document.file_id)
                file_cache[document.file_id] = file
            metadata = json.loads(document.metadata_json)
            values.append(
                IndexedImageChunk(
                    image_chunk_id=record.id,
                    dataset_id=record.dataset_id,
                    document_id=record.document_id,
                    document_version=record.index_version,
                    file_id=file.id,
                    document_name=file.original_name,
                    document_mime_type=file.mime_type,
                    asset_id=record.asset_id,
                    image_id=record.image_id,
                    manual_name=record.manual_name,
                    chapter_title=record.chapter_title,
                    page_number=record.page_number,
                    caption=record.caption,
                    ocr_text=record.ocr_text,
                    retrieval_text=record.retrieval_text,
                    product=str(metadata.get("product")) if metadata.get("product") else None,
                    confidence=record.confidence,
                    related_parent_ids=json.loads(record.related_parent_ids_json),
                    related_child_ids=json.loads(record.related_child_ids_json),
                    embedding=self._decode_embedding(record.embedding_json),
                )
            )
        return values

    @staticmethod
    def _parent_view(record) -> ParentChunkView:
        return ParentChunkView(
            id=record.id,
            document_id=record.document_id,
            index_version=record.index_version,
            title=record.title,
            heading_path=json.loads(record.heading_path_json),
            text=record.text,
            page_start=record.page_start,
            page_end=record.page_end,
            token_count=record.token_count,
            enabled=record.enabled,
            edited=record.edited,
        )

    @staticmethod
    def _child_view(record) -> ChildChunkView:
        return ChildChunkView(
            id=record.id,
            parent_id=record.parent_id,
            document_id=record.document_id,
            index_version=record.index_version,
            title=record.title,
            text=record.text,
            page_start=record.page_start,
            page_end=record.page_end,
            token_count=record.token_count,
            keywords=json.loads(record.keywords_json),
            questions=json.loads(record.questions_json),
            tags=json.loads(record.tags_json),
            asset_ids=json.loads(record.asset_ids_json),
            enabled=record.enabled,
            edited=record.edited,
        )

    @classmethod
    def _image_chunk_view(cls, record) -> ImageChunkView:
        embedding = cls._decode_embedding(record.embedding_json)
        return ImageChunkView(
            id=record.id,
            dataset_id=record.dataset_id,
            document_id=record.document_id,
            index_version=record.index_version,
            asset_id=record.asset_id,
            asset_url=f"/api/assets/{record.asset_id}",
            image_id=record.image_id,
            manual_name=record.manual_name,
            chapter_title=record.chapter_title,
            page_number=record.page_number,
            caption=record.caption,
            ocr_text=record.ocr_text,
            visible_text=json.loads(record.visible_text_json),
            visual_summary=record.visual_summary,
            visual_meaning=record.visual_meaning,
            retrieval_text=record.retrieval_text,
            search_terms=json.loads(record.search_terms_json),
            applicable_questions=json.loads(record.applicable_questions_json),
            issue_signals=json.loads(record.issue_signals_json),
            related_parent_ids=json.loads(record.related_parent_ids_json),
            related_child_ids=json.loads(record.related_child_ids_json),
            confidence=record.confidence,
            content_hash=record.content_hash,
            embedding_dimension=len(embedding) if embedding else 0,
            enabled=record.enabled,
        )


class LiveKnowledgeRetriever:
    """Loads the current published versions for every request.

    This avoids holding an obsolete in-memory index after a Dataset publish.
    """

    def __init__(self, service: KnowledgeService) -> None:
        self.service = service
        self._last_explanation = ContextVar(
            f"live_knowledge_last_explanation_{id(self)}",
            default=None,
        )
        self._last_image_explanation = ContextVar(
            f"live_knowledge_last_image_explanation_{id(self)}",
            default=None,
        )
        self._last_shadow_image_evidence = ContextVar(
            f"live_knowledge_last_shadow_image_evidence_{id(self)}",
            default=None,
        )
        self._last_text_retriever = ContextVar(
            f"live_knowledge_last_text_retriever_{id(self)}",
            default=None,
        )

    @property
    def last_explanation(self):
        return self._last_explanation.get()

    @last_explanation.setter
    def last_explanation(self, value) -> None:
        self._last_explanation.set(value)

    @property
    def last_image_explanation(self):
        return self._last_image_explanation.get()

    @last_image_explanation.setter
    def last_image_explanation(self, value) -> None:
        self._last_image_explanation.set(value)

    @property
    def last_shadow_image_evidence(self):
        return self._last_shadow_image_evidence.get() or []

    @last_shadow_image_evidence.setter
    def last_shadow_image_evidence(self, value) -> None:
        self._last_shadow_image_evidence.set(value)

    @property
    def last_text_retriever(self):
        """Return the concrete text retriever used by the current request."""

        return self._last_text_retriever.get()

    @last_text_retriever.setter
    def last_text_retriever(self, value) -> None:
        self._last_text_retriever.set(value)

    def search(self, query: str, products: list[str] | None = None, top_k: int = 5):
        retriever = self.service.retriever()
        self.last_text_retriever = retriever
        text_evidence = retriever.search(query, products=products, top_k=top_k)
        self.last_explanation = retriever.last_explanation
        self.last_image_explanation = None
        self.last_shadow_image_evidence = []
        mode = self.service.settings.image_chunk_retrieval
        if mode == "off":
            return text_evidence
        image_retriever = self.service.image_retriever()
        image_evidence, image_explanation = image_retriever.search(query, top_k=top_k)
        merged, shadow = merge_image_evidence(
            text_evidence,
            image_evidence,
            mode=mode,
            query=query,
        )
        self.last_image_explanation = image_explanation
        self.last_shadow_image_evidence = shadow
        return merged

    def explain(self, query: str, **kwargs):
        retriever = self.service.retriever(kwargs.get("dataset_ids"))
        self.last_text_retriever = retriever
        self.last_explanation = retriever.explain(query, **kwargs)
        return self.last_explanation
