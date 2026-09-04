from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.knowledge.hybrid import IndexedChild
from app.knowledge.image_retrieval import IndexedImageChunk
from app.knowledge.index_bundle import IndexBundle, IndexManifest


@dataclass(frozen=True)
class BundleRuntimeSnapshot:
    manifest: IndexManifest
    children: tuple[IndexedChild, ...]
    images: tuple[IndexedImageChunk, ...]


def load_bundle_runtime(bundle_root: str | Path) -> BundleRuntimeSnapshot:
    bundle = IndexBundle.load(bundle_root)
    report = bundle.validate()
    if not report.valid:
        codes = ", ".join(dict.fromkeys(item.code for item in report.errors))
        raise ValueError(f"offline index bundle validation failed: {codes}")

    text_rows = _read_jsonl(bundle.root / "text_chunks.jsonl")
    image_rows = _read_jsonl(bundle.root / "image_chunks.jsonl")
    text_vectors = _read_vectors(bundle.root / "text_vectors.npz")
    image_vectors = _read_vectors(bundle.root / "image_caption_vectors.npz")
    source_by_document = {item.document_id: item for item in bundle.manifest.sources}
    parent_by_id = {
        str(item["id"]): item
        for item in text_rows
        if item.get("chunk_type") == "parent" and item.get("id")
    }

    children: list[IndexedChild] = []
    child_by_id: dict[str, IndexedChild] = {}
    for row in text_rows:
        if row.get("chunk_type") != "child":
            continue
        child_id = _required_string(row, "id")
        parent_id = _required_string(row, "parent_id")
        document_id = _required_string(row, "document_id")
        parent = parent_by_id.get(parent_id)
        source = source_by_document.get(document_id)
        if parent is None or source is None:
            raise ValueError(
                f"offline index child {child_id} is missing its parent or source manifest"
            )
        child = IndexedChild(
            child_id=child_id,
            parent_id=parent_id,
            dataset_id=str(row.get("dataset_id") or bundle.manifest.dataset_id),
            document_id=document_id,
            document_version=str(
                row.get("document_version") or source.document_version or bundle.manifest.index_version
            ),
            file_id=source.file_id,
            document_name=source.source_name,
            document_mime_type=source.mime_type,
            title=str(row.get("title") or parent.get("title") or ""),
            text=str(row.get("text") or ""),
            parent_text=str(parent.get("text") or ""),
            product=_optional_string(row.get("product") or parent.get("product")),
            page_start=_positive_int(row.get("page_start") or parent.get("page_start")),
            page_end=_positive_int(row.get("page_end") or parent.get("page_end")),
            asset_ids=_string_list(row.get("asset_ids")),
            keywords=_string_list(row.get("keywords")),
            embedding=text_vectors.get(child_id),
        )
        children.append(child)
        child_by_id[child_id] = child

    images: list[IndexedImageChunk] = []
    for row in image_rows:
        image_chunk_id = _required_string(row, "id")
        document_id = _required_string(row, "document_id")
        source = source_by_document.get(document_id)
        if source is None:
            raise ValueError(
                f"offline image chunk {image_chunk_id} is missing its source manifest"
            )
        related_parent_ids = _string_list(row.get("related_parent_ids"))
        related_child_ids = _string_list(row.get("related_child_ids"))
        product = _related_product(
            related_parent_ids,
            related_child_ids,
            parent_by_id,
            child_by_id,
        )
        images.append(
            IndexedImageChunk(
                image_chunk_id=image_chunk_id,
                dataset_id=str(row.get("dataset_id") or bundle.manifest.dataset_id),
                document_id=document_id,
                document_version=str(
                    row.get("document_version")
                    or source.document_version
                    or bundle.manifest.index_version
                ),
                file_id=source.file_id,
                document_name=source.source_name,
                document_mime_type=source.mime_type,
                asset_id=_required_string(row, "asset_id"),
                image_id=str(row.get("image_id") or image_chunk_id),
                manual_name=str(row.get("manual_name") or source.source_name),
                chapter_title=str(row.get("chapter_title") or ""),
                page_number=_positive_int(row.get("page_number")),
                caption=str(row.get("caption") or ""),
                ocr_text=str(row.get("ocr_text") or ""),
                retrieval_text=str(
                    row.get("retrieval_text") or row.get("caption") or row.get("ocr_text") or ""
                ),
                product=product,
                confidence=_confidence(row.get("confidence")),
                related_parent_ids=related_parent_ids,
                related_child_ids=related_child_ids,
                embedding=image_vectors.get(image_chunk_id),
            )
        )

    if len(children) != bundle.manifest.counts.get("child_chunks", len(children)):
        raise ValueError("offline index child count does not match manifest")
    if len(images) != bundle.manifest.counts.get("image_chunks", len(images)):
        raise ValueError("offline index image count does not match manifest")
    return BundleRuntimeSnapshot(
        manifest=bundle.manifest,
        children=tuple(children),
        images=tuple(images),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"offline index JSONL row must be an object: {path.name}")
        rows.append(value)
    return rows


def _read_vectors(path: Path) -> dict[str, list[float]]:
    with np.load(path, allow_pickle=False) as payload:
        ids = payload["ids"].tolist()
        vectors = payload["vectors"]
    return {
        str(item_id): [float(value) for value in vectors[index].tolist()]
        for index, item_id in enumerate(ids)
    }


def _required_string(row: dict[str, Any], key: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"offline index row is missing required field: {key}")
    return value


def _optional_string(value: object) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _positive_int(value: object) -> int:
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _related_product(
    parent_ids: list[str],
    child_ids: list[str],
    parent_by_id: dict[str, dict[str, Any]],
    child_by_id: dict[str, IndexedChild],
) -> str | None:
    for parent_id in parent_ids:
        parent = parent_by_id.get(parent_id)
        if parent and (product := _optional_string(parent.get("product"))):
            return product
    for child_id in child_ids:
        child = child_by_id.get(child_id)
        if child and child.product:
            return child.product
    return None
