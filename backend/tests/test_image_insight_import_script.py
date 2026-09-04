from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient

from app.main import create_app
from scripts.import_manual_image_insights import import_insights


def test_import_script_creates_embedded_image_chunk_and_report(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    client = TestClient(app)
    dataset_id = client.post("/api/datasets", json={"name": "图片洞察"}).json()["id"]
    file_id = client.post(
        "/api/files",
        files={"file": ("manual.md", "# 铭牌\n查看铭牌。".encode(), "text/markdown")},
    ).json()["id"]
    document_id = client.post(
        f"/api/datasets/{dataset_id}/documents",
        json={"file_id": file_id},
    ).json()["id"]
    job = client.post(f"/api/documents/{document_id}/parse").json()
    assert client.post(
        f"/api/datasets/{dataset_id}/publish",
        json={"index_version": job["index_version"]},
    ).status_code == 200

    image_bytes = b"real-image-bytes"
    image_path = tmp_path / "knowledge-assets" / "fixture.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)
    app.state.knowledge_service.repository.create_asset(
        asset_id="asset-fixture",
        dataset_id=dataset_id,
        document_id=document_id,
        index_version=job["index_version"],
        asset_type="image",
        page_number=1,
        storage_path=str(image_path.relative_to(tmp_path)),
        caption="铭牌位置",
    )
    insights_path = tmp_path / "manual_image_insights.jsonl"
    insights_path.write_text(
        json.dumps(
            {
                "image_id": "fixture",
                "file_name": "fixture.jpg",
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "manual_name": "图片洞察",
                "chapter_hint": "铭牌",
                "visual_summary": "铭牌位置图",
                "retrieval_text": "图中展示产品铭牌位置",
                "confidence": 0.95,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report, report_path = import_insights(
        dataset_id=dataset_id,
        data_dir=tmp_path,
        insights_path=insights_path,
        embed=lambda texts: [[1.0, 0.0] for _ in texts],
    )

    records = app.state.knowledge_service.repository.list_image_chunks(
        dataset_id=dataset_id,
        published_only=True,
    )
    assert report.matched_sha256 == 1
    assert report.embedded_rows == 1
    assert report_path.is_file()
    assert len(records) == 1
    assert json.loads(records[0].embedding_json or "[]") == [1.0, 0.0]
