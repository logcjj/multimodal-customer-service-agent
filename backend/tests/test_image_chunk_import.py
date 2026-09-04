from __future__ import annotations

import json

from app.knowledge.image_retrieval import (
    AssetImportContext,
    AssetMatchCandidate,
    build_image_chunk_rows,
    match_insight_to_asset,
    normalize_image_insight,
)


def test_sha256_match_wins_over_filename() -> None:
    candidates = [
        AssetMatchCandidate(
            asset_id="asset-by-name",
            document_id="doc-1",
            sha256="a" * 64,
            source_name="Blower_01.png",
        ),
        AssetMatchCandidate(
            asset_id="asset-by-hash",
            document_id="doc-1",
            sha256="b" * 64,
            source_name="different.png",
        ),
    ]

    match = match_insight_to_asset(
        {"sha256": "b" * 64, "file_name": "Blower_01.png", "image_id": "Blower_01"},
        candidates,
    )

    assert match.asset_id == "asset-by-hash"
    assert match.strategy == "sha256"


def test_unique_stem_matches_but_ambiguous_stem_is_rejected() -> None:
    unique = match_insight_to_asset(
        {"file_name": "Blower_02.png", "image_id": "Blower_02"},
        [
            AssetMatchCandidate(
                asset_id="asset-1",
                document_id="doc-1",
                sha256="a" * 64,
                source_name="blower_02.jpg",
            )
        ],
    )
    ambiguous = match_insight_to_asset(
        {"file_name": "Blower_02.png", "image_id": "Blower_02"},
        [
            AssetMatchCandidate("asset-1", "doc-1", "a" * 64, "blower_02.jpg"),
            AssetMatchCandidate("asset-2", "doc-2", "b" * 64, "Blower_02.png"),
        ],
    )

    assert unique.asset_id == "asset-1"
    assert unique.strategy == "stem"
    assert ambiguous.asset_id is None
    assert ambiguous.strategy == "ambiguous"


def test_normalized_insight_preserves_retrieval_fields_and_stable_id() -> None:
    raw = {
        "image_id": "Blower_02",
        "manual_name": "吹风机手册",
        "chapter_hint": "安全说明 / 防护罩警告",
        "retrieval_text": "防护罩损坏时禁止使用",
        "visible_text": ["WARNING", "Do not use with a damaged guard"],
        "visual_summary": "防护罩安全警告",
        "visual_meaning": "损坏时必须停用",
        "search_terms": ["防护罩", "damaged guard", "防护罩"],
        "applicable_questions": ["防护罩坏了还能用吗？"],
        "issue_signals": ["unsafe operation"],
        "confidence": 0.99,
        "sha256": "c" * 64,
    }

    first = normalize_image_insight(
        raw,
        dataset_id="manuals",
        document_id="doc-1",
        index_version="idx-1",
        asset_id="asset-1",
        page_number=8,
    )
    second = normalize_image_insight(
        raw,
        dataset_id="manuals",
        document_id="doc-1",
        index_version="idx-1",
        asset_id="asset-1",
        page_number=8,
    )

    assert first["id"] == second["id"]
    assert first["visible_text"] == ["WARNING", "Do not use with a damaged guard"]
    assert first["search_terms"] == ["防护罩", "damaged guard"]
    assert "防护罩坏了还能用吗" in first["embedding_text"]
    assert first["confidence"] == 0.99


def test_jsonl_import_reports_unmatched_and_creates_asset_fallback(tmp_path) -> None:
    insights = tmp_path / "insights.jsonl"
    insights.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image_id": "matched",
                        "sha256": "a" * 64,
                        "manual_name": "手册",
                        "chapter_hint": "安全",
                        "visual_summary": "必须断电",
                        "retrieval_text": "操作前必须断电",
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "image_id": "missing",
                        "sha256": "f" * 64,
                        "manual_name": "其他手册",
                        "retrieval_text": "未关联图片",
                        "confidence": 0.8,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    contexts = [
        AssetImportContext(
            asset_id="asset-matched",
            dataset_id="manuals",
            document_id="doc-1",
            index_version="idx-1",
            page_number=2,
            sha256="a" * 64,
            source_name="matched.png",
            manual_name="手册",
            caption="已有说明",
            ocr_text="",
            related_parent_ids=["parent-1"],
            related_child_ids=["child-1"],
        ),
        AssetImportContext(
            asset_id="asset-fallback",
            dataset_id="manuals",
            document_id="doc-1",
            index_version="idx-1",
            page_number=3,
            sha256="b" * 64,
            source_name="fallback.png",
            manual_name="手册",
            caption="没有洞察时使用现有 Caption",
            ocr_text="VISIBLE",
            related_parent_ids=["parent-2"],
            related_child_ids=["child-2"],
        ),
    ]

    rows, report = build_image_chunk_rows(insights, contexts)

    assert len(rows["doc-1"]) == 2
    assert report.matched_sha256 == 1
    assert report.unmatched_insights == 1
    assert report.fallback_assets == 1
    matched = next(item for item in rows["doc-1"] if item["asset_id"] == "asset-matched")
    fallback = next(item for item in rows["doc-1"] if item["asset_id"] == "asset-fallback")
    assert matched["related_child_ids"] == ["child-1"]
    assert fallback["caption"] == "没有洞察时使用现有 Caption"
