from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.knowledge.chunking import ChunkingService
from app.knowledge.parsers import ParserRegistry
from app.knowledge.repository import KnowledgeRepository
from app.knowledge.storage import ContentAddressedStorage


class IngestionService:
    parser_version = "v3.1-parser-1"

    def __init__(
        self,
        repository: KnowledgeRepository,
        storage: ContentAddressedStorage,
        data_dir: Path,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.parser = ParserRegistry()
        self.chunker = ChunkingService()
        self.asset_root = data_dir / "knowledge-assets"
        self.asset_root.mkdir(parents=True, exist_ok=True)

    def parse_document(self, document_id: str):
        document = self.repository.get_document(document_id)
        file = self.repository.get_file(document.file_id)
        config_hash = hashlib.sha256(
            f"{file.content_hash}:{document.parser_profile}:{self.parser_version}".encode()
        ).hexdigest()
        job = self.repository.create_job(document_id, config_hash)
        if job.state == "succeeded":
            return job
        self.repository.update_job(
            job.id,
            state="running",
            stage="extract",
            progress=5,
            started_at=datetime.now(UTC),
        )
        try:
            path = self.storage.resolve(file.storage_path)
            normalized = self.parser.parse(path, file.mime_type)
            self.repository.update_job(job.id, stage="chunk", progress=45)
            chunks = self.chunker.chunk(normalized, document.parser_profile)
            index_version = f"idx-{config_hash[:12]}"
            asset_map: dict[str, str] = {}
            for asset in normalized.assets:
                digest = hashlib.sha256(asset.content).hexdigest()
                asset_id = f"asset-{digest[:20]}"
                target = self.asset_root / f"{digest}{asset.extension}"
                if not target.exists():
                    target.write_bytes(asset.content)
                asset_map[asset.local_id] = asset_id
                self.repository.create_asset(
                    asset_id=asset_id,
                    dataset_id=document.dataset_id,
                    document_id=document.id,
                    index_version=index_version,
                    asset_type=asset.asset_type,
                    page_number=asset.page_number,
                    storage_path=str(target.relative_to(self.storage.root.parent)),
                    bbox=asset.bbox,
                    caption=asset.caption,
                    ocr_text=asset.ocr_text,
                )
            self.repository.update_job(job.id, stage="index", progress=75)
            self.repository.replace_chunks(
                document_id=document.id,
                dataset_id=document.dataset_id,
                index_version=index_version,
                parents=[item.__dict__ for item in chunks.parents],
                children=[
                    {
                        **item.__dict__,
                        "asset_ids": [asset_map[value] for value in item.asset_ids if value in asset_map],
                    }
                    for item in chunks.children
                ],
            )
            return self.repository.update_job(
                job.id,
                state="succeeded",
                stage="reviewed",
                progress=100,
                index_version=index_version,
                finished_at=datetime.now(UTC),
            )
        except Exception as exc:
            self.repository.update_job(
                job.id,
                state="failed",
                stage="extract",
                progress=100,
                error_code=exc.__class__.__name__,
                error_message=str(exc)[:500],
                finished_at=datetime.now(UTC),
            )
            raise

