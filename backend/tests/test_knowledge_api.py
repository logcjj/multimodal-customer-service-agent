from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_fresh_app_does_not_seed_demo_knowledge(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get("/api/datasets")

    assert response.status_code == 200
    assert response.json() == []


def test_upload_parse_publish_and_report_real_metrics(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    dataset = client.post(
        "/api/datasets",
        json={"name": "售后知识库", "description": "真实测试", "parser_profile": "manual"},
    )
    assert dataset.status_code == 201
    dataset_id = dataset.json()["id"]

    uploaded = client.post(
        "/api/files",
        files={"file": ("../manual.txt", "# E03 排水故障\n先断电，再清理排水过滤器。".encode(), "text/plain")},
    )
    assert uploaded.status_code == 201
    assert ".." not in uploaded.json()["storage_path"]

    duplicate = client.post(
        "/api/files",
        files={"file": ("copy.txt", "# E03 排水故障\n先断电，再清理排水过滤器。".encode(), "text/plain")},
    )
    assert duplicate.json()["id"] == uploaded.json()["id"]

    document = client.post(
        f"/api/datasets/{dataset_id}/documents",
        json={"file_id": uploaded.json()["id"], "parser_profile": "manual"},
    )
    assert document.status_code == 201

    parsed = client.post(f"/api/documents/{document.json()['id']}/parse")
    assert parsed.status_code == 200
    assert parsed.json()["state"] == "succeeded"
    assert parsed.json()["index_version"]

    published = client.post(
        f"/api/datasets/{dataset_id}/publish",
        json={"index_version": parsed.json()["index_version"]},
    )
    assert published.status_code == 200

    listed = client.get("/api/datasets").json()
    current = next(item for item in listed if item["id"] == dataset_id)
    assert current["document_count"] == 1
    assert current["parent_count"] >= 1
    assert current["child_count"] >= 1


def test_rejects_mismatched_file_signature(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.post(
        "/api/files",
        files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 415


def test_original_file_content_endpoint_returns_stored_file_and_404s_unknown_id(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    content = "# 昨日工作日报\n完成动态知识库入库。".encode()
    uploaded = client.post(
        "/api/files",
        files={"file": ("工作日报.md", content, "text/markdown")},
    ).json()

    response = client.get(f"/api/files/{uploaded['id']}/content")
    missing = client.get("/api/files/not-found/content")

    assert response.status_code == 200
    assert response.content == content
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["content-disposition"].startswith("inline;")
    assert missing.status_code == 404


def test_parse_fails_at_embedding_stage_when_configured_model_returns_no_vectors(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    service = app.state.knowledge_service
    service.embedding_configured_provider = lambda: True
    service.embedding_model_provider = lambda: "configured-embedding"
    service.embed_override = lambda texts: []
    client = TestClient(app)
    dataset_id = client.post(
        "/api/datasets",
        json={"name": "日报知识库", "parser_profile": "manual"},
    ).json()["id"]
    file_id = client.post(
        "/api/files",
        files={"file": ("daily.md", "# 工作日报\n完成动态入库验证。".encode(), "text/markdown")},
    ).json()["id"]
    document_id = client.post(
        f"/api/datasets/{dataset_id}/documents",
        json={"file_id": file_id, "parser_profile": "manual"},
    ).json()["id"]

    response = client.post(f"/api/documents/{document_id}/parse")
    job = service.repository.latest_job(document_id)

    assert response.status_code == 502
    assert "Embedding" in response.json()["detail"]
    assert job is not None
    assert job.state == "failed"
    assert job.stage == "embedding"
    assert job.progress == 100
    assert job.error_code == "EmbeddingIncomplete"


def test_publish_rejects_configured_candidate_with_missing_embeddings(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    service = app.state.knowledge_service
    service.embedding_configured_provider = lambda: True
    service.embedding_model_provider = lambda: "configured-embedding"
    client = TestClient(app)
    dataset_id = client.post(
        "/api/datasets",
        json={"name": "待发布知识库", "parser_profile": "manual"},
    ).json()["id"]
    file_id = client.post(
        "/api/files",
        files={"file": ("candidate.md", "# 候选版本\n尚未向量化。".encode(), "text/markdown")},
    ).json()["id"]
    document_id = client.post(
        f"/api/datasets/{dataset_id}/documents",
        json={"file_id": file_id, "parser_profile": "manual"},
    ).json()["id"]
    job = service.ingestion.parse_document(document_id)

    response = client.post(
        f"/api/datasets/{dataset_id}/publish",
        json={"index_version": job.index_version},
    )

    assert response.status_code == 409
    assert "Embedding" in response.json()["detail"]
    assert service.repository.get_dataset(dataset_id).published_version is None
