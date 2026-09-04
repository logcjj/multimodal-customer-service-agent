from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from app.knowledge.image_retrieval import (
    AssetImportContext,
    ImageInsightImportReport,
    build_image_chunk_rows,
)
from app.knowledge.providers import ModelGateway
from app.knowledge.repository import KnowledgeRepository
from app.models.service import ModelService
from app.storage.database import Database


EmbedOverride = Callable[[list[str]], list[list[float]]]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_insights(
    *,
    dataset_id: str,
    data_dir: Path,
    insights_path: Path,
    embed: EmbedOverride | None = None,
) -> tuple[ImageInsightImportReport, Path]:
    data_dir = data_dir.expanduser().resolve()
    database = Database(data_dir)
    repository = KnowledgeRepository(database)
    dataset = repository.get_dataset(dataset_id)
    if not dataset.published_version:
        raise ValueError("dataset must be published before importing image insights")

    documents = {
        item.id: item
        for item in repository.list_document_refs(dataset_id=dataset_id)
        if item.enabled and item.published_version
    }
    children = repository.list_children(dataset_ids=[dataset_id], published_only=True)
    parent_ids_by_asset: dict[str, set[str]] = {}
    child_ids_by_asset: dict[str, set[str]] = {}
    for child in children:
        for asset_id in json.loads(child.asset_ids_json):
            parent_ids_by_asset.setdefault(asset_id, set()).add(child.parent_id)
            child_ids_by_asset.setdefault(asset_id, set()).add(child.id)

    contexts: list[AssetImportContext] = []
    for asset in repository.list_dataset_assets(dataset_id, published_only=True):
        document = documents.get(asset.document_id)
        if document is None:
            continue
        file = repository.get_file(document.file_id)
        path = (data_dir / asset.storage_path).resolve()
        if data_dir not in path.parents or not path.is_file():
            continue
        contexts.append(
            AssetImportContext(
                asset_id=asset.id,
                dataset_id=dataset_id,
                document_id=asset.document_id,
                index_version=asset.index_version,
                page_number=asset.page_number,
                sha256=_file_sha256(path),
                source_name=path.name,
                manual_name=Path(file.original_name).stem,
                caption=asset.caption,
                ocr_text=asset.ocr_text,
                related_parent_ids=sorted(parent_ids_by_asset.get(asset.id, set())),
                related_child_ids=sorted(child_ids_by_asset.get(asset.id, set())),
            )
        )

    rows_by_document, report = build_image_chunk_rows(insights_path, contexts)
    existing = {
        item.id: item
        for item in repository.list_image_chunks(dataset_id=dataset_id, published_only=True)
    }
    vectors: dict[str, list[float]] = {}
    missing_rows: list[dict[str, object]] = []
    for rows in rows_by_document.values():
        for row in rows:
            previous = existing.get(str(row["id"]))
            if previous and previous.content_hash == row["content_hash"] and previous.embedding_json:
                vector = json.loads(previous.embedding_json)
                if isinstance(vector, list) and vector:
                    vectors[str(row["id"])] = [float(value) for value in vector]
                    row["embedding"] = vectors[str(row["id"])]
                    continue
            missing_rows.append(row)

    if missing_rows and embed is not None:
        for start in range(0, len(missing_rows), 10):
            batch = missing_rows[start : start + 10]
            texts = [str(item.get("embedding_text", ""))[:2500] for item in batch]
            returned = embed(texts)
            if len(returned) != len(batch):
                raise ValueError("image caption embedding returned an incomplete batch")
            for row, vector in zip(batch, returned, strict=True):
                decoded = [float(value) for value in vector]
                if not decoded:
                    raise ValueError("image caption embedding returned an empty vector")
                row["embedding"] = decoded
                vectors[str(row["id"])] = decoded

    for document_id, document in documents.items():
        repository.replace_image_chunks(
            document_id=document_id,
            dataset_id=dataset_id,
            index_version=str(document.published_version),
            items=rows_by_document.get(document_id, []),
        )

    report = replace(report, embedded_rows=len(vectors))
    report_root = data_dir / "index-import-reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / (
        f"{dataset_id}-image-insights-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report, report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    backend_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="导入手册图片洞察并生成独立 ImageChunk")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--insights", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=backend_root / "data")
    parser.add_argument("--skip-embedding", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    embed = None
    if not args.skip_embedding:
        database = Database(data_dir)
        gateway = ModelGateway(ModelService(database))
        embed = gateway.embed
    try:
        report, report_path = import_insights(
            dataset_id=args.dataset_id,
            data_dir=data_dir,
            insights_path=args.insights.expanduser().resolve(),
            embed=embed,
        )
    except Exception as exc:
        print(f"status=failed error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        "status=ready "
        f"imported_rows={report.imported_rows} embedded_rows={report.embedded_rows} "
        f"matched_sha256={report.matched_sha256} matched_stem={report.matched_stem} "
        f"unmatched={report.unmatched_insights} ambiguous={report.ambiguous_insights} "
        f"fallback_assets={report.fallback_assets} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

