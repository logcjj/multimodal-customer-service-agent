from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from app.knowledge.index_bundle import (
    IndexBundle,
    IndexBundleWriter,
    SourceManifest,
    plan_incremental_build,
)
from app.knowledge.service import KnowledgeService
from app.main import create_app
from app.storage.database import Database
from scripts.build_index_bundle import build_bundle


def _source(name: str, digest: str) -> SourceManifest:
    return SourceManifest(
        document_id=f"doc-{name}",
        file_id=f"file-{name}",
        source_name=name,
        source_sha256=digest,
        mime_type="application/pdf",
        size_bytes=100,
        parser_fingerprint="parser-v1",
        document_version="idx-source-v1",
    )


def test_bundle_round_trip_and_checksum_validation(tmp_path) -> None:
    writer = IndexBundleWriter(tmp_path)
    bundle = writer.write(
        dataset_id="manuals",
        index_version="idx-v1",
        parser_version="parser-v1",
        embedding_model="embedding-test",
        vector_dimension=2,
        sources=[_source("a.pdf", "a" * 64)],
        text_chunks=[{"id": "child-1", "document_id": "doc-a.pdf", "text": "E03 排水故障"}],
        image_chunks=[],
        assets=[],
        text_vectors={"child-1": [0.1, 0.2]},
        image_caption_vectors={},
    )

    loaded = IndexBundle.load(bundle.root)

    assert loaded.manifest.dataset_id == "manuals"
    assert loaded.manifest.counts["text_chunks"] == 1
    assert loaded.validate().valid is True
    with np.load(bundle.root / "text_vectors.npz", allow_pickle=False) as vectors:
        assert vectors["ids"].tolist() == ["child-1"]
        assert vectors["vectors"].shape == (1, 2)

    (bundle.root / "text_chunks.jsonl").write_text("tampered\n", encoding="utf-8")
    report = IndexBundle.load(bundle.root).validate()

    assert report.valid is False
    assert any(item.code == "index_checksum_mismatch" for item in report.errors)


def test_bundle_rejects_non_finite_or_wrong_dimension_vectors(tmp_path) -> None:
    writer = IndexBundleWriter(tmp_path)

    try:
        writer.write(
            dataset_id="manuals",
            index_version="idx-invalid",
            parser_version="parser-v1",
            embedding_model="embedding-test",
            vector_dimension=2,
            sources=[_source("a.pdf", "a" * 64)],
            text_chunks=[{"id": "child-1", "document_id": "doc-a.pdf", "text": "text"}],
            image_chunks=[],
            assets=[],
            text_vectors={"child-1": [float("nan")]},
            image_caption_vectors={},
        )
    except ValueError as exc:
        assert "vector" in str(exc).lower()
    else:
        raise AssertionError("invalid vector must be rejected")


def test_incremental_plan_reuses_changes_adds_and_deletes() -> None:
    previous = [_source("a.pdf", "a" * 64), _source("b.pdf", "b" * 64)]
    current = [_source("a.pdf", "a" * 64), _source("b.pdf", "c" * 64), _source("c.pdf", "d" * 64)]

    plan = plan_incremental_build(previous, current)

    assert plan.reuse == ["doc-a.pdf"]
    assert plan.update == ["doc-b.pdf"]
    assert plan.add == ["doc-c.pdf"]
    assert plan.delete == []

    delete_plan = plan_incremental_build(current, [current[0]])
    assert delete_plan.delete == ["doc-b.pdf", "doc-c.pdf"]


def test_manifest_and_jsonl_are_deterministic(tmp_path) -> None:
    writer = IndexBundleWriter(tmp_path)
    bundle = writer.write(
        dataset_id="manuals",
        index_version="idx-deterministic",
        parser_version="parser-v1",
        embedding_model=None,
        vector_dimension=0,
        sources=[_source("a.pdf", "a" * 64)],
        text_chunks=[{"text": "内容", "id": "child-1"}],
        image_chunks=[],
        assets=[],
        text_vectors={},
        image_caption_vectors={},
    )

    line = (bundle.root / "text_chunks.jsonl").read_text(encoding="utf-8").strip()
    payload = json.loads(line)

    assert payload == {"id": "child-1", "text": "内容"}
    assert line.index('"id"') < line.index('"text"')


def test_offline_cli_builder_creates_valid_runtime_bundle(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        dataset_id = client.post("/api/datasets", json={"name": "CLI 知识库"}).json()["id"]
        file_id = client.post(
            "/api/files",
            files={"file": ("manual.md", "# 安全\n操作前断电。".encode(), "text/markdown")},
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

    manifest = build_bundle(dataset_id=dataset_id, data_dir=tmp_path, timeout_seconds=5)

    assert manifest.dataset_id == dataset_id
    assert manifest.validation_status == "valid"
    assert manifest.counts["text_chunks"] >= 1


def test_offline_cli_builder_records_the_default_embedding_model(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        created_model = client.post(
            "/api/models",
            json={
                "provider": "OpenAI-API-Compatible",
                "name": "text-embedding-v4",
                "kind": "embedding",
                "base_url": "https://example.com/v1",
                "api_key": "test-only-secret",
            },
        )
        assert created_model.status_code == 201
        client.app.state.knowledge_service.embed_override = lambda texts: [
            [float(index + 1), 1.0] for index, _text in enumerate(texts)
        ]

        dataset_id = client.post("/api/datasets", json={"name": "Embedding CLI 知识库"}).json()[
            "id"
        ]
        file_id = client.post(
            "/api/files",
            files={"file": ("manual.md", "# 安全\n操作前断电。".encode(), "text/markdown")},
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

    manifest = build_bundle(dataset_id=dataset_id, data_dir=tmp_path, timeout_seconds=5)

    assert manifest.embedding_model == "text-embedding-v4"


def test_embedding_model_change_creates_a_new_bundle_identity(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        first_model_id = client.post(
            "/api/models",
            json={
                "provider": "OpenAI-API-Compatible",
                "name": "embedding-model-v1",
                "kind": "embedding",
                "base_url": "https://example.com/v1",
                "api_key": "test-only-secret",
            },
        ).json()["id"]
        client.app.state.knowledge_service.embed_override = lambda texts: [
            [float(index + 1), 1.0] for index, _text in enumerate(texts)
        ]
        dataset_id = client.post("/api/datasets", json={"name": "Bundle 身份知识库"}).json()["id"]
        file_id = client.post(
            "/api/files",
            files={"file": ("manual.md", "# 安全\n操作前断电。".encode(), "text/markdown")},
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

    first = build_bundle(dataset_id=dataset_id, data_dir=tmp_path, timeout_seconds=5)

    with TestClient(create_app(data_dir=tmp_path)) as client:
        second_model_id = client.post(
            "/api/models",
            json={
                "provider": "OpenAI-API-Compatible",
                "name": "embedding-model-v2",
                "kind": "embedding",
                "base_url": "https://example.com/v1",
                "api_key": "test-only-secret",
            },
        ).json()["id"]
        assert first_model_id != second_model_id
        assert client.post(f"/api/models/{second_model_id}/default").status_code == 200

    second = build_bundle(dataset_id=dataset_id, data_dir=tmp_path, timeout_seconds=5)

    assert first.embedding_model == "embedding-model-v1"
    assert second.embedding_model == "embedding-model-v2"
    assert first.index_version != second.index_version


def test_offline_index_mode_serves_active_bundle_without_runtime_chunk_query(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        dataset_id = client.post("/api/datasets", json={"name": "离线运行知识库"}).json()["id"]
        file_id = client.post(
            "/api/files",
            files={
                "file": (
                    "manual.md",
                    "# 安全操作\n启动设备前必须先检查电源线。".encode(),
                    "text/markdown",
                )
            },
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

    build_bundle(dataset_id=dataset_id, data_dir=tmp_path, timeout_seconds=5)
    service = KnowledgeService(Database(tmp_path))

    def fail_runtime_query(dataset_ids=None):
        raise AssertionError("offline mode must not query runtime child rows")

    service._indexed_children = fail_runtime_query
    try:
        evidence = service.retriever([dataset_id]).search("启动前检查什么？", top_k=1)
    finally:
        service.shutdown()

    assert evidence
    assert evidence[0].document_name == "manual.md"
    assert "检查电源线" in evidence[0].text


def test_invalid_active_bundle_falls_back_to_database_and_reports_failure(tmp_path) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        dataset_id = client.post("/api/datasets", json={"name": "离线降级知识库"}).json()["id"]
        file_id = client.post(
            "/api/files",
            files={
                "file": (
                    "fallback.md",
                    "# 安全操作\n维护设备前必须断开电源。".encode(),
                    "text/markdown",
                )
            },
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

    manifest = build_bundle(dataset_id=dataset_id, data_dir=tmp_path, timeout_seconds=5)
    bundle_root = tmp_path / "index-bundles" / dataset_id / manifest.index_version
    (bundle_root / "text_chunks.jsonl").write_text("tampered\n", encoding="utf-8")
    service = KnowledgeService(Database(tmp_path))
    try:
        evidence = service.retriever([dataset_id]).search("维护前应该做什么？", top_k=1)
        status = service.offline_index_status()[dataset_id]
    finally:
        service.shutdown()

    assert evidence
    assert "断开电源" in evidence[0].text
    assert status["status"] == "failed"
    assert status["error_code"] == "index_load_failed"
