from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import create_app


def test_image_chunks_are_independent_published_resources(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    client = TestClient(app)
    dataset_id = client.post("/api/datasets", json={"name": "图文手册"}).json()["id"]
    file_id = client.post(
        "/api/files",
        files={"file": ("manual.md", "# 防护罩\n损坏时禁止使用。".encode(), "text/markdown")},
    ).json()["id"]
    document_id = client.post(
        f"/api/datasets/{dataset_id}/documents",
        json={"file_id": file_id},
    ).json()["id"]
    job = client.post(f"/api/documents/{document_id}/parse").json()
    child = app.state.knowledge_service.repository.list_children(
        document_id=document_id,
        index_version=job["index_version"],
    )[0]
    asset = app.state.knowledge_service.repository.create_asset(
        asset_id="asset-guard",
        dataset_id=dataset_id,
        document_id=document_id,
        index_version=job["index_version"],
        asset_type="image",
        page_number=1,
        storage_path="knowledge-assets/test/guard.png",
        caption="防护罩警告",
        ocr_text="WARNING",
    )
    app.state.knowledge_service.repository.replace_image_chunks(
        document_id=document_id,
        dataset_id=dataset_id,
        index_version=job["index_version"],
        items=[
            {
                "id": "image-guard",
                "asset_id": asset.id,
                "image_id": "guard-01",
                "manual_name": "图文手册",
                "chapter_title": "防护罩",
                "page_number": 1,
                "caption": "防护罩损坏或缺失时严禁使用设备",
                "ocr_text": "WARNING",
                "visible_text": ["WARNING"],
                "visual_summary": "防护罩警告",
                "visual_meaning": "损坏时停用",
                "retrieval_text": "防护罩坏了不能使用",
                "search_terms": ["防护罩"],
                "applicable_questions": ["防护罩坏了还能用吗"],
                "issue_signals": ["unsafe operation"],
                "related_parent_ids": [child.parent_id],
                "related_child_ids": [child.id],
                "confidence": 0.99,
                "content_hash": "a" * 64,
                "embedding": [1.0, 0.0],
            }
        ],
    )
    assert client.post(
        f"/api/datasets/{dataset_id}/publish",
        json={"index_version": job["index_version"]},
    ).status_code == 200

    listed = client.get(f"/api/datasets/{dataset_id}/image-chunks")
    detail = client.get("/api/image-chunks/image-guard")

    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["asset_id"] == "asset-guard"
    assert listed.json()[0]["related_child_ids"] == [child.id]
    assert detail.status_code == 200
    assert detail.json()["asset_url"] == "/api/assets/asset-guard"
    assert detail.json()["embedding_dimension"] == 2

    build = client.post(f"/api/datasets/{dataset_id}/index-builds").json()
    deadline = time.monotonic() + 5
    while build["state"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.02)
        build = client.get(f"/api/index-builds/{build['id']}").json()
    manifest = client.get(f"/api/datasets/{dataset_id}/index-manifest").json()

    assert build["state"] == "succeeded"
    assert manifest["counts"]["image_chunks"] == 1
    assert manifest["counts"]["image_caption_vectors"] == 1


def test_unpublished_image_chunks_are_not_listed(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    client = TestClient(app)
    dataset_id = client.post("/api/datasets", json={"name": "未发布图片"}).json()["id"]

    response = client.get(f"/api/datasets/{dataset_id}/image-chunks")

    assert response.status_code == 200
    assert response.json() == []


def test_image_chunk_list_honors_limit_for_large_knowledge_bases(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    dataset_id = client.post("/api/datasets", json={"name": "图片分页"}).json()["id"]

    response = client.get(
        f"/api/datasets/{dataset_id}/image-chunks",
        params={"limit": 1, "offset": 0},
    )

    assert response.status_code == 200
    assert len(response.json()) <= 1
