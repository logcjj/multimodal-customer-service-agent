from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException
from sqlmodel import select

from app.knowledge.models import (
    ChildChunkRecord,
    DatasetRecord,
    DocumentRefRecord,
    FileAssetRecord,
    IndexBuildRecord,
    IndexManifestRecord,
    ImageChunkRecord,
    KnowledgeAssetRecord,
    ParentChunkRecord,
    ParsingJobRecord,
    RetrievalProfileRecord,
)
from app.storage.database import Database


class KnowledgeRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def create_file(
        self,
        original_name: str,
        content_hash: str,
        mime_type: str,
        size_bytes: int,
        storage_path: str,
    ) -> FileAssetRecord:
        with self.database.session() as session:
            existing = session.exec(
                select(FileAssetRecord).where(FileAssetRecord.content_hash == content_hash)
            ).first()
            if existing:
                session.expunge(existing)
                return existing
            record = FileAssetRecord(
                id=str(uuid4()),
                original_name=original_name,
                content_hash=content_hash,
                mime_type=mime_type,
                size_bytes=size_bytes,
                storage_path=storage_path,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def list_files(self) -> list[FileAssetRecord]:
        with self.database.session() as session:
            records = session.exec(select(FileAssetRecord).order_by(FileAssetRecord.created_at.desc())).all()
            for record in records:
                session.expunge(record)
            return list(records)

    def get_file(self, file_id: str) -> FileAssetRecord:
        with self.database.session() as session:
            record = session.get(FileAssetRecord, file_id)
            if record is None:
                raise HTTPException(status_code=404, detail="文件不存在")
            session.expunge(record)
            return record

    def create_dataset(
        self,
        name: str,
        *,
        description: str = "",
        parser_profile: str = "manual",
        visibility: str = "private",
        dataset_id: str | None = None,
        is_system: bool = False,
    ) -> DatasetRecord:
        with self.database.session() as session:
            record = DatasetRecord(
                id=dataset_id or str(uuid4()),
                name=name,
                description=description,
                default_parser_profile=parser_profile,
                visibility=visibility,
                is_system=is_system,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def list_datasets(self) -> list[DatasetRecord]:
        with self.database.session() as session:
            records = session.exec(select(DatasetRecord).order_by(DatasetRecord.created_at)).all()
            for record in records:
                session.expunge(record)
            return list(records)

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        with self.database.session() as session:
            record = session.get(DatasetRecord, dataset_id)
            if record is None:
                raise HTTPException(status_code=404, detail="知识库不存在")
            session.expunge(record)
            return record

    def update_dataset(self, dataset_id: str, **values: object) -> DatasetRecord:
        with self.database.session() as session:
            record = session.get(DatasetRecord, dataset_id)
            if record is None:
                raise HTTPException(status_code=404, detail="知识库不存在")
            for key, value in values.items():
                if key == "parser_profile":
                    key = "default_parser_profile"
                setattr(record, key, value)
            record.updated_at = datetime.now(UTC)
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def create_retrieval_profile(self, **values: object) -> RetrievalProfileRecord:
        with self.database.session() as session:
            record = RetrievalProfileRecord(id=str(uuid4()), **values)
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def list_retrieval_profiles(self) -> list[RetrievalProfileRecord]:
        with self.database.session() as session:
            records = session.exec(
                select(RetrievalProfileRecord).order_by(RetrievalProfileRecord.created_at)
            ).all()
            for record in records:
                session.expunge(record)
            return list(records)

    def get_retrieval_profile(self, profile_id: str) -> RetrievalProfileRecord:
        with self.database.session() as session:
            record = session.get(RetrievalProfileRecord, profile_id)
            if record is None:
                raise HTTPException(status_code=404, detail="检索配置不存在")
            session.expunge(record)
            return record

    def update_retrieval_profile(self, profile_id: str, **values: object) -> RetrievalProfileRecord:
        with self.database.session() as session:
            record = session.get(RetrievalProfileRecord, profile_id)
            if record is None:
                raise HTTPException(status_code=404, detail="检索配置不存在")
            for key, value in values.items():
                setattr(record, key, value)
            record.updated_at = datetime.now(UTC)
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def link_file(
        self,
        dataset_id: str,
        file_id: str,
        parser_profile: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> DocumentRefRecord:
        dataset = self.get_dataset(dataset_id)
        self.get_file(file_id)
        with self.database.session() as session:
            existing = session.exec(
                select(DocumentRefRecord).where(
                    DocumentRefRecord.dataset_id == dataset_id,
                    DocumentRefRecord.file_id == file_id,
                )
            ).first()
            if existing:
                session.expunge(existing)
                return existing
            record = DocumentRefRecord(
                id=str(uuid4()),
                dataset_id=dataset_id,
                file_id=file_id,
                parser_profile=parser_profile or dataset.default_parser_profile,
                metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def list_document_refs(
        self,
        *,
        dataset_id: str | None = None,
        file_id: str | None = None,
    ) -> list[DocumentRefRecord]:
        with self.database.session() as session:
            statement = select(DocumentRefRecord)
            if dataset_id:
                statement = statement.where(DocumentRefRecord.dataset_id == dataset_id)
            if file_id:
                statement = statement.where(DocumentRefRecord.file_id == file_id)
            records = session.exec(statement.order_by(DocumentRefRecord.created_at)).all()
            for record in records:
                session.expunge(record)
            return list(records)

    def get_document(self, document_id: str) -> DocumentRefRecord:
        with self.database.session() as session:
            record = session.get(DocumentRefRecord, document_id)
            if record is None:
                raise HTTPException(status_code=404, detail="知识库文档不存在")
            session.expunge(record)
            return record

    def create_job(self, document_id: str, config_hash: str) -> ParsingJobRecord:
        with self.database.session() as session:
            existing = session.exec(
                select(ParsingJobRecord).where(
                    ParsingJobRecord.document_ref_id == document_id,
                    ParsingJobRecord.config_hash == config_hash,
                    ParsingJobRecord.state.in_(["queued", "running", "succeeded"]),
                )
            ).first()
            if existing:
                session.expunge(existing)
                return existing
            record = ParsingJobRecord(id=str(uuid4()), document_ref_id=document_id, config_hash=config_hash)
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def get_job(self, job_id: str) -> ParsingJobRecord:
        with self.database.session() as session:
            record = session.get(ParsingJobRecord, job_id)
            if record is None:
                raise HTTPException(status_code=404, detail="解析任务不存在")
            session.expunge(record)
            return record

    def latest_job(self, document_id: str) -> ParsingJobRecord | None:
        with self.database.session() as session:
            record = session.exec(
                select(ParsingJobRecord)
                .where(ParsingJobRecord.document_ref_id == document_id)
                .order_by(ParsingJobRecord.created_at.desc())
            ).first()
            if record:
                session.expunge(record)
            return record

    def update_job(self, job_id: str, **values: object) -> ParsingJobRecord:
        with self.database.session() as session:
            record = session.get(ParsingJobRecord, job_id)
            if record is None:
                raise HTTPException(status_code=404, detail="解析任务不存在")
            for key, value in values.items():
                setattr(record, key, value)
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def create_index_build(self, dataset_id: str, index_version: str) -> IndexBuildRecord:
        self.get_dataset(dataset_id)
        with self.database.session() as session:
            record = IndexBuildRecord(
                id=str(uuid4()),
                dataset_id=dataset_id,
                index_version=index_version,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def get_index_build(self, build_id: str) -> IndexBuildRecord:
        with self.database.session() as session:
            record = session.get(IndexBuildRecord, build_id)
            if record is None:
                raise HTTPException(status_code=404, detail="索引构建任务不存在")
            session.expunge(record)
            return record

    def update_index_build(self, build_id: str, **values: object) -> IndexBuildRecord:
        with self.database.session() as session:
            record = session.get(IndexBuildRecord, build_id)
            if record is None:
                raise HTTPException(status_code=404, detail="索引构建任务不存在")
            for key, value in values.items():
                setattr(record, key, value)
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def save_index_manifest(
        self,
        *,
        dataset_id: str,
        index_version: str,
        bundle_path: str,
        manifest_json: str,
        validation_status: str,
        active: bool,
    ) -> IndexManifestRecord:
        with self.database.session() as session:
            if active:
                for item in session.exec(
                    select(IndexManifestRecord).where(IndexManifestRecord.dataset_id == dataset_id)
                ).all():
                    item.active = False
                    session.add(item)
            record = session.exec(
                select(IndexManifestRecord).where(
                    IndexManifestRecord.dataset_id == dataset_id,
                    IndexManifestRecord.index_version == index_version,
                )
            ).first()
            if record is None:
                record = IndexManifestRecord(
                    id=str(uuid4()),
                    dataset_id=dataset_id,
                    index_version=index_version,
                    bundle_path=bundle_path,
                    manifest_json=manifest_json,
                )
            record.bundle_path = bundle_path
            record.manifest_json = manifest_json
            record.validation_status = validation_status
            record.active = active
            record.published_at = datetime.now(UTC) if active else None
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def active_index_manifest(self, dataset_id: str) -> IndexManifestRecord | None:
        with self.database.session() as session:
            record = session.exec(
                select(IndexManifestRecord)
                .where(
                    IndexManifestRecord.dataset_id == dataset_id,
                    IndexManifestRecord.active == True,  # noqa: E712
                )
                .order_by(IndexManifestRecord.published_at.desc())
            ).first()
            if record is not None:
                session.expunge(record)
            return record

    def get_index_manifest(
        self,
        dataset_id: str,
        index_version: str,
    ) -> IndexManifestRecord:
        with self.database.session() as session:
            record = session.exec(
                select(IndexManifestRecord).where(
                    IndexManifestRecord.dataset_id == dataset_id,
                    IndexManifestRecord.index_version == index_version,
                )
            ).first()
            if record is None:
                raise HTTPException(status_code=404, detail="索引 Manifest 不存在")
            session.expunge(record)
            return record

    def activate_index_manifest(
        self,
        dataset_id: str,
        index_version: str,
        *,
        require_previously_active: bool = True,
    ) -> IndexManifestRecord:
        with self.database.session() as session:
            target = session.exec(
                select(IndexManifestRecord).where(
                    IndexManifestRecord.dataset_id == dataset_id,
                    IndexManifestRecord.index_version == index_version,
                )
            ).first()
            if target is None:
                raise HTTPException(status_code=404, detail="索引 Manifest 不存在")
            if require_previously_active and target.published_at is None:
                raise HTTPException(
                    status_code=409,
                    detail="该索引版本从未通过运行发布，不能绕过评测门禁直接激活。",
                )
            records = session.exec(
                select(IndexManifestRecord).where(IndexManifestRecord.dataset_id == dataset_id)
            ).all()
            for record in records:
                record.active = record.id == target.id
                session.add(record)
            target.published_at = datetime.now(UTC)
            session.add(target)
            session.commit()
            session.refresh(target)
            session.expunge(target)
            return target

    def replace_chunks(
        self,
        *,
        document_id: str,
        dataset_id: str,
        index_version: str,
        parents: list[dict[str, object]],
        children: list[dict[str, object]],
    ) -> tuple[list[ParentChunkRecord], list[ChildChunkRecord]]:
        parent_records: list[ParentChunkRecord] = []
        child_records: list[ChildChunkRecord] = []
        local_to_id: dict[str, str] = {}
        with self.database.session() as session:
            for item in parents:
                record_id = str(uuid4())
                local_to_id[str(item["local_id"])] = record_id
                record = ParentChunkRecord(
                    id=record_id,
                    dataset_id=dataset_id,
                    document_id=document_id,
                    index_version=index_version,
                    local_id=str(item["local_id"]),
                    title=str(item["title"]),
                    heading_path_json=json.dumps(item.get("heading_path", []), ensure_ascii=False),
                    text=str(item["text"]),
                    page_start=int(item.get("page_start", 1)),
                    page_end=int(item.get("page_end", 1)),
                    token_count=int(item.get("token_count", 0)),
                    content_hash=hashlib.sha256(str(item["text"]).encode()).hexdigest(),
                    product=str(item["product"]) if item.get("product") else None,
                )
                session.add(record)
                parent_records.append(record)
            session.flush()
            for item in children:
                text = str(item["text"])
                record = ChildChunkRecord(
                    id=str(uuid4()),
                    parent_id=local_to_id[str(item["parent_local_id"])] ,
                    dataset_id=dataset_id,
                    document_id=document_id,
                    index_version=index_version,
                    local_id=str(item["local_id"]),
                    title=str(item["title"]),
                    text=text,
                    normalized_text=str(item.get("normalized_text", text.lower())),
                    page_start=int(item.get("page_start", 1)),
                    page_end=int(item.get("page_end", 1)),
                    token_count=int(item.get("token_count", 0)),
                    keywords_json=json.dumps(item.get("keywords", []), ensure_ascii=False),
                    questions_json=json.dumps(item.get("questions", []), ensure_ascii=False),
                    tags_json=json.dumps(item.get("tags", []), ensure_ascii=False),
                    asset_ids_json=json.dumps(item.get("asset_ids", []), ensure_ascii=False),
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    product=str(item["product"]) if item.get("product") else None,
                    embedding_json=json.dumps(item["embedding"]) if item.get("embedding") is not None else None,
                )
                session.add(record)
                child_records.append(record)
            document = session.get(DocumentRefRecord, document_id)
            if document:
                document.active_version = index_version
                document.updated_at = datetime.now(UTC)
                session.add(document)
            session.commit()
            for record in [*parent_records, *child_records]:
                session.refresh(record)
                session.expunge(record)
        return parent_records, child_records

    def publish_dataset(self, dataset_id: str, index_version: str) -> DatasetRecord:
        with self.database.session() as session:
            dataset = session.get(DatasetRecord, dataset_id)
            if dataset is None:
                raise HTTPException(status_code=404, detail="知识库不存在")
            chunks = session.exec(
                select(ChildChunkRecord).where(
                    ChildChunkRecord.dataset_id == dataset_id,
                    ChildChunkRecord.index_version == index_version,
                )
            ).first()
            if chunks is None:
                raise HTTPException(status_code=409, detail="该索引版本没有可发布的 Chunk")
            documents = session.exec(
                select(DocumentRefRecord).where(
                    DocumentRefRecord.dataset_id == dataset_id,
                    DocumentRefRecord.active_version == index_version,
                )
            ).all()
            if not documents:
                raise HTTPException(status_code=409, detail="该索引版本未关联到可发布文档")
            for document in documents:
                document.published_version = index_version
                document.updated_at = datetime.now(UTC)
                session.add(document)
            dataset.published_version = index_version
            dataset.updated_at = datetime.now(UTC)
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
            session.expunge(dataset)
            return dataset

    def list_parents(self, document_id: str, index_version: str | None = None) -> list[ParentChunkRecord]:
        with self.database.session() as session:
            statement = select(ParentChunkRecord).where(ParentChunkRecord.document_id == document_id)
            if index_version:
                statement = statement.where(ParentChunkRecord.index_version == index_version)
            records = session.exec(statement.order_by(ParentChunkRecord.created_at)).all()
            for record in records:
                session.expunge(record)
            return list(records)

    def list_dataset_parents(
        self,
        dataset_id: str,
        *,
        published_only: bool = False,
    ) -> list[ParentChunkRecord]:
        with self.database.session() as session:
            records = list(
                session.exec(
                    select(ParentChunkRecord)
                    .where(
                        ParentChunkRecord.dataset_id == dataset_id,
                        ParentChunkRecord.enabled == True,  # noqa: E712
                    )
                    .order_by(ParentChunkRecord.created_at)
                ).all()
            )
            if published_only:
                published_versions = {
                    item.id: item.published_version
                    for item in session.exec(
                        select(DocumentRefRecord).where(
                            DocumentRefRecord.dataset_id == dataset_id,
                            DocumentRefRecord.enabled == True,  # noqa: E712
                            DocumentRefRecord.published_version != None,  # noqa: E711
                        )
                    ).all()
                }
                records = [
                    item
                    for item in records
                    if published_versions.get(item.document_id) == item.index_version
                ]
            for record in records:
                session.expunge(record)
            return records

    def get_parent(self, parent_id: str) -> ParentChunkRecord:
        with self.database.session() as session:
            record = session.get(ParentChunkRecord, parent_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Parent Chunk 不存在")
            session.expunge(record)
            return record

    def create_asset(
        self,
        *,
        asset_id: str,
        dataset_id: str,
        document_id: str,
        index_version: str,
        asset_type: str,
        page_number: int,
        storage_path: str,
        bbox: tuple[float, float, float, float] | None = None,
        caption: str = "",
        ocr_text: str = "",
    ) -> KnowledgeAssetRecord:
        with self.database.session() as session:
            existing = session.get(KnowledgeAssetRecord, asset_id)
            if existing:
                session.expunge(existing)
                return existing
            record = KnowledgeAssetRecord(
                id=asset_id,
                dataset_id=dataset_id,
                document_id=document_id,
                index_version=index_version,
                asset_type=asset_type,
                page_number=page_number,
                bbox_json=json.dumps(bbox) if bbox else None,
                storage_path=storage_path,
                caption=caption,
                ocr_text=ocr_text,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def create_assets_bulk(self, items: list[dict[str, object]]) -> int:
        if not items:
            return 0
        inserted = 0
        with self.database.session() as session:
            existing_ids = {
                item
                for item in session.exec(
                    select(KnowledgeAssetRecord.id).where(
                        KnowledgeAssetRecord.id.in_([str(value["id"]) for value in items])
                    )
                ).all()
            }
            for item in items:
                asset_id = str(item["id"])
                if asset_id in existing_ids:
                    continue
                session.add(
                    KnowledgeAssetRecord(
                        id=asset_id,
                        dataset_id=str(item["dataset_id"]),
                        document_id=str(item["document_id"]),
                        index_version=str(item["index_version"]),
                        asset_type=str(item.get("asset_type", "image")),
                        page_number=int(item.get("page_number", 1)),
                        storage_path=str(item["storage_path"]),
                        caption=str(item.get("caption", "")),
                        ocr_text=str(item.get("ocr_text", "")),
                    )
                )
                inserted += 1
            session.commit()
        return inserted

    def get_asset(self, asset_id: str) -> KnowledgeAssetRecord:
        with self.database.session() as session:
            record = session.get(KnowledgeAssetRecord, asset_id)
            if record is None:
                raise HTTPException(status_code=404, detail="证据资产不存在")
            session.expunge(record)
            return record

    def replace_image_chunks(
        self,
        *,
        document_id: str,
        dataset_id: str,
        index_version: str,
        items: list[dict[str, object]],
    ) -> list[ImageChunkRecord]:
        records: list[ImageChunkRecord] = []
        with self.database.session() as session:
            existing = session.exec(
                select(ImageChunkRecord).where(
                    ImageChunkRecord.document_id == document_id,
                    ImageChunkRecord.index_version == index_version,
                )
            ).all()
            for item in existing:
                session.delete(item)
            session.flush()
            for item in items:
                embedding = item.get("embedding")
                record = ImageChunkRecord(
                    id=str(item["id"]),
                    dataset_id=dataset_id,
                    document_id=document_id,
                    index_version=index_version,
                    asset_id=str(item["asset_id"]),
                    image_id=str(item.get("image_id", item["asset_id"])),
                    manual_name=str(item.get("manual_name", "")),
                    chapter_title=str(item.get("chapter_title", "")),
                    page_number=max(1, int(item.get("page_number", 1))),
                    caption=str(item.get("caption", "")),
                    ocr_text=str(item.get("ocr_text", "")),
                    visible_text_json=json.dumps(item.get("visible_text", []), ensure_ascii=False),
                    visual_summary=str(item.get("visual_summary", "")),
                    visual_meaning=str(item.get("visual_meaning", "")),
                    retrieval_text=str(item.get("retrieval_text", "")),
                    search_terms_json=json.dumps(item.get("search_terms", []), ensure_ascii=False),
                    applicable_questions_json=json.dumps(
                        item.get("applicable_questions", []), ensure_ascii=False
                    ),
                    issue_signals_json=json.dumps(item.get("issue_signals", []), ensure_ascii=False),
                    related_parent_ids_json=json.dumps(
                        item.get("related_parent_ids", []), ensure_ascii=False
                    ),
                    related_child_ids_json=json.dumps(
                        item.get("related_child_ids", []), ensure_ascii=False
                    ),
                    confidence=float(item.get("confidence", 0.0)),
                    content_hash=str(item["content_hash"]),
                    embedding_json=(
                        json.dumps(embedding, ensure_ascii=False) if embedding is not None else None
                    ),
                    enabled=bool(item.get("enabled", True)),
                )
                session.add(record)
                records.append(record)
            session.commit()
            for record in records:
                session.refresh(record)
                session.expunge(record)
        return records

    def get_image_chunk(self, image_chunk_id: str) -> ImageChunkRecord:
        with self.database.session() as session:
            record = session.get(ImageChunkRecord, image_chunk_id)
            if record is None:
                raise HTTPException(status_code=404, detail="ImageChunk 不存在")
            session.expunge(record)
            return record

    def list_image_chunks(
        self,
        *,
        dataset_id: str | None = None,
        document_id: str | None = None,
        index_version: str | None = None,
        published_only: bool = False,
    ) -> list[ImageChunkRecord]:
        with self.database.session() as session:
            statement = select(ImageChunkRecord).where(ImageChunkRecord.enabled == True)  # noqa: E712
            if dataset_id:
                statement = statement.where(ImageChunkRecord.dataset_id == dataset_id)
            if document_id:
                statement = statement.where(ImageChunkRecord.document_id == document_id)
            if index_version:
                statement = statement.where(ImageChunkRecord.index_version == index_version)
            records = list(session.exec(statement.order_by(ImageChunkRecord.created_at)).all())
            if published_only:
                published_versions = {
                    item.id: item.published_version
                    for item in session.exec(
                        select(DocumentRefRecord).where(
                            DocumentRefRecord.enabled == True,  # noqa: E712
                            DocumentRefRecord.published_version != None,  # noqa: E711
                        )
                    ).all()
                }
                records = [
                    item
                    for item in records
                    if published_versions.get(item.document_id) == item.index_version
                ]
            for record in records:
                session.expunge(record)
            return records

    def save_image_embeddings(
        self,
        vectors: dict[str, list[float]],
    ) -> int:
        updated = 0
        if not vectors:
            return updated
        with self.database.session() as session:
            records = session.exec(
                select(ImageChunkRecord).where(ImageChunkRecord.id.in_(list(vectors)))
            ).all()
            for record in records:
                vector = vectors.get(record.id)
                if vector is None:
                    continue
                record.embedding_json = json.dumps(vector)
                session.add(record)
                updated += 1
            session.commit()
        return updated

    def list_assets(self, document_id: str, index_version: str | None = None) -> list[KnowledgeAssetRecord]:
        with self.database.session() as session:
            statement = select(KnowledgeAssetRecord).where(KnowledgeAssetRecord.document_id == document_id)
            if index_version:
                statement = statement.where(KnowledgeAssetRecord.index_version == index_version)
            records = session.exec(statement.order_by(KnowledgeAssetRecord.page_number)).all()
            for record in records:
                session.expunge(record)
            return list(records)

    def list_dataset_assets(
        self,
        dataset_id: str,
        *,
        published_only: bool = False,
    ) -> list[KnowledgeAssetRecord]:
        with self.database.session() as session:
            records = list(
                session.exec(
                    select(KnowledgeAssetRecord)
                    .where(KnowledgeAssetRecord.dataset_id == dataset_id)
                    .order_by(KnowledgeAssetRecord.page_number)
                ).all()
            )
            if published_only:
                published_versions = {
                    item.id: item.published_version
                    for item in session.exec(
                        select(DocumentRefRecord).where(
                            DocumentRefRecord.dataset_id == dataset_id,
                            DocumentRefRecord.enabled == True,  # noqa: E712
                            DocumentRefRecord.published_version != None,  # noqa: E711
                        )
                    ).all()
                }
                records = [
                    item
                    for item in records
                    if published_versions.get(item.document_id) == item.index_version
                ]
            for record in records:
                session.expunge(record)
            return records

    def list_children(
        self,
        *,
        document_id: str | None = None,
        dataset_ids: list[str] | None = None,
        index_version: str | None = None,
        published_only: bool = False,
    ) -> list[ChildChunkRecord]:
        with self.database.session() as session:
            statement = select(ChildChunkRecord).where(ChildChunkRecord.enabled == True)  # noqa: E712
            if document_id:
                statement = statement.where(ChildChunkRecord.document_id == document_id)
            if dataset_ids:
                statement = statement.where(ChildChunkRecord.dataset_id.in_(dataset_ids))
            if index_version:
                statement = statement.where(ChildChunkRecord.index_version == index_version)
            records = session.exec(statement.order_by(ChildChunkRecord.created_at)).all()
            if published_only:
                published_datasets = {
                    item.id
                    for item in session.exec(
                        select(DatasetRecord).where(DatasetRecord.published_version != None)  # noqa: E711
                    ).all()
                }
                published_documents = {
                    item.id: item.published_version
                    for item in session.exec(
                        select(DocumentRefRecord).where(
                            DocumentRefRecord.enabled == True,  # noqa: E712
                            DocumentRefRecord.published_version != None,  # noqa: E711
                        )
                    ).all()
                    if item.dataset_id in published_datasets
                }
                records = [
                    item
                    for item in records
                    if published_documents.get(item.document_id) == item.index_version
                ]
            for record in records:
                session.expunge(record)
            return list(records)

    def has_vector_map_sources(self, dataset_id: str, published_version: str) -> bool:
        with self.database.session() as session:
            child_id = session.exec(
                select(ChildChunkRecord.id)
                .join(DocumentRefRecord, DocumentRefRecord.id == ChildChunkRecord.document_id)
                .join(DatasetRecord, DatasetRecord.id == ChildChunkRecord.dataset_id)
                .where(
                    ChildChunkRecord.dataset_id == dataset_id,
                    ChildChunkRecord.enabled == True,  # noqa: E712
                    ChildChunkRecord.embedding_json != None,  # noqa: E711
                    DocumentRefRecord.dataset_id == dataset_id,
                    DocumentRefRecord.enabled == True,  # noqa: E712
                    DocumentRefRecord.published_version == ChildChunkRecord.index_version,
                    DatasetRecord.published_version == published_version,
                )
                .limit(1)
            ).first()
            return child_id is not None

    def get_child(self, child_id: str) -> ChildChunkRecord:
        with self.database.session() as session:
            record = session.get(ChildChunkRecord, child_id)
            if record is None:
                raise HTTPException(status_code=404, detail="Chunk 不存在")
            session.expunge(record)
            return record

    def save_embeddings(
        self,
        document_id: str,
        index_version: str,
        vectors: dict[str, list[float]],
    ) -> int:
        updated = 0
        with self.database.session() as session:
            records = session.exec(
                select(ChildChunkRecord).where(
                    ChildChunkRecord.document_id == document_id,
                    ChildChunkRecord.index_version == index_version,
                )
            ).all()
            for record in records:
                vector = vectors.get(record.id)
                if vector is None:
                    continue
                record.embedding_json = json.dumps(vector)
                session.add(record)
                updated += 1
            session.commit()
        return updated

    def edit_child(
        self,
        child_id: str,
        *,
        text: str | None = None,
        keywords: list[str] | None = None,
        questions: list[str] | None = None,
        tags: list[str] | None = None,
        enabled: bool | None = None,
    ) -> ChildChunkRecord:
        source = self.get_child(child_id)
        new_text = text if text is not None else source.text
        draft_version = f"draft-{uuid4()}"
        with self.database.session() as session:
            source_parents = list(
                session.exec(
                    select(ParentChunkRecord).where(
                        ParentChunkRecord.document_id == source.document_id,
                        ParentChunkRecord.index_version == source.index_version,
                    )
                ).all()
            )
            source_children = list(
                session.exec(
                    select(ChildChunkRecord).where(
                        ChildChunkRecord.document_id == source.document_id,
                        ChildChunkRecord.index_version == source.index_version,
                    )
                ).all()
            )
            parent_id_map: dict[str, str] = {}
            for parent in source_parents:
                cloned_text = parent.text
                is_target_parent = parent.id == source.parent_id
                if is_target_parent and source.text in cloned_text:
                    cloned_text = cloned_text.replace(source.text, new_text, 1)
                elif is_target_parent:
                    cloned_text = new_text
                cloned = ParentChunkRecord(
                    id=str(uuid4()),
                    dataset_id=parent.dataset_id,
                    document_id=parent.document_id,
                    index_version=draft_version,
                    local_id=parent.local_id,
                    title=parent.title,
                    heading_path_json=parent.heading_path_json,
                    text=cloned_text,
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    bbox_json=parent.bbox_json,
                    token_count=len(cloned_text),
                    content_hash=hashlib.sha256(cloned_text.encode()).hexdigest(),
                    enabled=parent.enabled,
                    edited=parent.edited or is_target_parent,
                    product=parent.product,
                )
                parent_id_map[parent.id] = cloned.id
                session.add(cloned)

            record = None
            for child in source_children:
                is_target = child.id == source.id
                cloned_text = new_text if is_target else child.text
                cloned = ChildChunkRecord(
                    id=str(uuid4()),
                    parent_id=parent_id_map[child.parent_id],
                    dataset_id=child.dataset_id,
                    document_id=child.document_id,
                    index_version=draft_version,
                    local_id=child.local_id,
                    title=child.title,
                    text=cloned_text,
                    normalized_text=cloned_text.lower(),
                    page_start=child.page_start,
                    page_end=child.page_end,
                    bbox_json=child.bbox_json,
                    token_count=len(cloned_text) if is_target else child.token_count,
                    keywords_json=(
                        json.dumps(keywords, ensure_ascii=False)
                        if is_target and keywords is not None
                        else child.keywords_json
                    ),
                    questions_json=(
                        json.dumps(questions, ensure_ascii=False)
                        if is_target and questions is not None
                        else child.questions_json
                    ),
                    tags_json=(
                        json.dumps(tags, ensure_ascii=False)
                        if is_target and tags is not None
                        else child.tags_json
                    ),
                    asset_ids_json=child.asset_ids_json,
                    content_hash=hashlib.sha256(cloned_text.encode()).hexdigest(),
                    enabled=(enabled if enabled is not None else child.enabled) if is_target else child.enabled,
                    edited=child.edited or is_target,
                    product=child.product,
                    embedding_json=child.embedding_json if not is_target or cloned_text == child.text else None,
                )
                session.add(cloned)
                if is_target:
                    record = cloned
            if record is None:
                raise HTTPException(status_code=409, detail="源 Chunk 版本不完整，无法创建编辑草稿")
            document = session.get(DocumentRefRecord, source.document_id)
            if document:
                document.active_version = draft_version
                document.updated_at = datetime.now(UTC)
                session.add(document)
            session.commit()
            session.refresh(record)
            session.expunge(record)
            return record

    def dataset_metrics(self, dataset_id: str) -> dict[str, int]:
        with self.database.session() as session:
            dataset = session.get(DatasetRecord, dataset_id)
            if dataset is None:
                raise HTTPException(status_code=404, detail="知识库不存在")
            documents = list(
                session.exec(
                    select(DocumentRefRecord).where(DocumentRefRecord.dataset_id == dataset_id)
                ).all()
            )
            document_count = len(documents)
            parent_count = 0
            child_count = 0
            asset_count = 0
            published_versions = {
                item.id: item.published_version
                for item in documents
                if item.enabled and item.published_version
            }
            if dataset.published_version and published_versions:
                parent_count = sum(
                    1
                    for item in session.exec(
                        select(ParentChunkRecord).where(ParentChunkRecord.dataset_id == dataset_id)
                    ).all()
                    if published_versions.get(item.document_id) == item.index_version and item.enabled
                )
                child_count = sum(
                    1
                    for item in session.exec(
                        select(ChildChunkRecord).where(ChildChunkRecord.dataset_id == dataset_id)
                    ).all()
                    if published_versions.get(item.document_id) == item.index_version and item.enabled
                )
                asset_ids: set[str] = set()
                for item in session.exec(
                    select(ChildChunkRecord).where(ChildChunkRecord.dataset_id == dataset_id)
                ).all():
                    if published_versions.get(item.document_id) == item.index_version and item.enabled:
                        asset_ids.update(json.loads(item.asset_ids_json))
                asset_count = len(asset_ids)
            document_ids = {item.id for item in documents}
            failed_jobs = sum(
                1
                for item in session.exec(
                    select(ParsingJobRecord).where(ParsingJobRecord.state == "failed")
                ).all()
                if item.document_ref_id in document_ids
            )
            return {
                "document_count": int(document_count),
                "parent_count": int(parent_count),
                "child_count": int(child_count),
                "asset_count": int(asset_count),
                "failed_job_count": int(failed_jobs),
            }
