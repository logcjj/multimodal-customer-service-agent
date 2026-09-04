from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from app.knowledge.repository import KnowledgeRepository
from app.storage.database import Database


DATASET_ID = "v6-manuals"
INDEX_VERSION = "v6-import-v1"


def stable_asset_id(product: str, image_id: str) -> str:
    return "asset-v6-" + hashlib.sha256(f"{product}|{image_id}".encode()).hexdigest()[:24]


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    workspace_root = project_root.parent
    v6_root = Path(os.getenv("V6_ROOT", workspace_root / "V6_submit_code")).resolve()
    source_data = v6_root / "data"
    if not (source_data / "section_chunks.json").exists():
        raise SystemExit(f"V6 data not found: {source_data}")

    database = Database(Path(os.getenv("AKA_DATA_DIR", project_root / "backend" / "data")))
    repository = KnowledgeRepository(database)
    try:
        existing = repository.get_dataset(DATASET_ID)
        print(
            f"V6 knowledge already imported: {existing.published_version or 'draft'}; "
            f"{repository.dataset_metrics(DATASET_ID)['child_count']} children"
        )
        return
    except Exception:
        pass

    sections = json.loads((source_data / "section_chunks.json").read_text(encoding="utf-8"))
    children = json.loads((source_data / "retrieval_chunks.json").read_text(encoding="utf-8"))
    caption_payload = json.loads((source_data / "image_captions_v4_final.json").read_text(encoding="utf-8"))
    captions = caption_payload.get("items", {})
    sections_by_product: dict[str, list[dict[str, object]]] = defaultdict(list)
    children_by_product: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in sections:
        sections_by_product[str(item["product"])].append(item)
    for item in children:
        children_by_product[str(item["product"])].append(item)

    repository.create_dataset(
        "V6 图文说明书知识库",
        description="从 V6 Small-to-Big 索引迁移的 39 类产品说明书、Parent Section、Child Chunk 与图片 Caption。",
        parser_profile="manual",
        dataset_id=DATASET_ID,
        is_system=True,
    )
    image_sources: dict[str, Path] = {}
    for root in (v6_root / "手册" / "插图", v6_root / "手册_v4" / "插图"):
        if root.exists():
            for path in root.iterdir():
                if path.is_file():
                    image_sources.setdefault(path.stem, path)
    asset_root = database.data_dir / "knowledge-assets" / "v6"
    asset_root.mkdir(parents=True, exist_ok=True)

    total_assets = 0
    for product in sorted(sections_by_product):
        product_sections = sorted(sections_by_product[product], key=lambda item: int(item["section_id"]))
        product_children = sorted(children_by_product[product], key=lambda item: int(item["subchunk_id"]))
        file_hash = hashlib.sha256(f"v6:{product}".encode()).hexdigest()
        file = repository.create_file(
            original_name=f"{product}.json",
            content_hash=file_hash,
            mime_type="application/json",
            size_bytes=sum(int(item.get("char_len", 0)) for item in product_sections),
            storage_path=f"source://v6/{product}",
        )
        document = repository.link_file(DATASET_ID, file.id, "manual", {"product": product, "source": "V6"})

        picture_ids = sorted(
            {
                str(picture)
                for item in [*product_sections, *product_children]
                for picture in item.get("pics", [])
            }
        )
        asset_map: dict[str, str] = {}
        asset_rows: list[dict[str, object]] = []
        for picture_id in picture_ids:
            source = image_sources.get(picture_id)
            if source is None:
                continue
            asset_id = stable_asset_id(product, picture_id)
            target = asset_root / f"{asset_id}{source.suffix.lower()}"
            if not target.exists():
                shutil.copy2(source, target)
            caption = captions.get(f"{product}|{picture_id}", {})
            asset_map[picture_id] = asset_id
            page_match = __import__("re").search(r"_(\d+)$", picture_id)
            asset_rows.append(
                {
                    "id": asset_id,
                    "dataset_id": DATASET_ID,
                    "document_id": document.id,
                    "index_version": INDEX_VERSION,
                    "asset_type": "image",
                    "page_number": int(page_match.group(1)) + 1 if page_match else 1,
                    "storage_path": str(target.relative_to(database.data_dir)),
                    "caption": str(caption.get("short_caption", "")),
                    "ocr_text": str(caption.get("content", "")),
                }
            )
        total_assets += repository.create_assets_bulk(asset_rows)

        parent_rows = [
            {
                "local_id": f"section-{int(item['section_id'])}",
                "title": str(item.get("heading", product)),
                "heading_path": [str(value) for value in item.get("heading_path", [])],
                "text": str(item.get("text", "")),
                "page_start": int(item["section_id"]) + 1,
                "page_end": int(item["section_id"]) + 1,
                "token_count": int(item.get("char_len", len(str(item.get("text", ""))))),
                "product": product,
            }
            for item in product_sections
        ]
        child_rows = [
            {
                "local_id": f"child-{int(item['subchunk_id'])}",
                "parent_local_id": f"section-{int(item['parent_section_id'])}",
                "title": str(item.get("heading", product)),
                "text": str(item.get("text", "")),
                "normalized_text": " ".join(str(item.get("text", "")).lower().split()),
                "page_start": int(item["parent_section_id"]) + 1,
                "page_end": int(item["parent_section_id"]) + 1,
                "token_count": int(item.get("char_len", len(str(item.get("text", ""))))),
                "keywords": [],
                "questions": [],
                "tags": [str(value) for value in item.get("tags", [])],
                "asset_ids": [asset_map[value] for value in item.get("pics", []) if value in asset_map],
                "product": product,
            }
            for item in product_children
        ]
        repository.replace_chunks(
            document_id=document.id,
            dataset_id=DATASET_ID,
            index_version=INDEX_VERSION,
            parents=parent_rows,
            children=child_rows,
        )

    repository.publish_dataset(DATASET_ID, INDEX_VERSION)
    metrics = repository.dataset_metrics(DATASET_ID)
    print(
        "Imported V6 knowledge: "
        f"{metrics['document_count']} documents, {metrics['parent_count']} parents, "
        f"{metrics['child_count']} children, {metrics['asset_count']} assets"
    )


if __name__ == "__main__":
    main()

