from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.knowledge.models import DatasetRecord
from app.main import create_app


def _published_dataset(client: TestClient) -> str:
    dataset_id = client.post(
        "/api/datasets",
        json={"name": "离线索引测试", "parser_profile": "manual"},
    ).json()["id"]
    file_id = client.post(
        "/api/files",
        files={"file": ("manual.md", "# E03\n先断电，再检查排水过滤器。".encode(), "text/markdown")},
    ).json()["id"]
    document_id = client.post(
        f"/api/datasets/{dataset_id}/documents",
        json={"file_id": file_id, "parser_profile": "manual"},
    ).json()["id"]
    parsed = client.post(f"/api/documents/{document_id}/parse").json()
    published = client.post(
        f"/api/datasets/{dataset_id}/publish",
        json={"index_version": parsed["index_version"]},
    )
    assert published.status_code == 200
    return dataset_id


def test_index_build_runs_outside_request_and_exposes_active_manifest(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    client = TestClient(app)
    dataset_id = _published_dataset(client)

    created = client.post(f"/api/datasets/{dataset_id}/index-builds")

    assert created.status_code == 202
    assert created.json()["state"] in {"queued", "running", "succeeded"}
    job_id = created.json()["id"]

    deadline = time.monotonic() + 5
    job = created.json()
    while job["state"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.02)
        job = client.get(f"/api/index-builds/{job_id}").json()

    assert job["state"] == "succeeded"
    assert job["stage"] == "validated"
    assert job["progress"] == 100

    manifest = client.get(f"/api/datasets/{dataset_id}/index-manifest")
    assert manifest.status_code == 200
    assert manifest.json()["dataset_id"] == dataset_id
    assert manifest.json()["counts"]["text_chunks"] >= 1
    assert manifest.json()["validation_status"] == "valid"


def test_index_manifest_requires_published_dataset(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    dataset_id = client.post("/api/datasets", json={"name": "空知识库"}).json()["id"]

    created = client.post(f"/api/datasets/{dataset_id}/index-builds")
    manifest = client.get(f"/api/datasets/{dataset_id}/index-manifest")

    assert created.status_code == 409
    assert manifest.status_code == 404


def test_bundle_version_changes_when_published_content_changes(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    client = TestClient(app)
    dataset_id = _published_dataset(client)

    first = client.post(f"/api/datasets/{dataset_id}/index-builds").json()
    deadline = time.monotonic() + 5
    while first["state"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.02)
        first = client.get(f"/api/index-builds/{first['id']}").json()
    first_manifest = client.get(f"/api/datasets/{dataset_id}/index-manifest").json()

    document = app.state.knowledge_service.repository.list_document_refs(dataset_id=dataset_id)[0]
    child = app.state.knowledge_service.repository.list_children(
        document_id=document.id,
        index_version=document.published_version,
    )[0]
    app.state.knowledge_service.repository.replace_image_chunks(
        document_id=document.id,
        dataset_id=dataset_id,
        index_version=document.published_version,
        items=[
            {
                "id": "image-added",
                "asset_id": "asset-added",
                "image_id": "added",
                "manual_name": "离线索引测试",
                "chapter_title": "新增图片",
                "page_number": 1,
                "caption": "新增图片说明",
                "retrieval_text": "新增图片说明",
                "related_parent_ids": [child.parent_id],
                "related_child_ids": [child.id],
                "confidence": 0.9,
                "content_hash": "d" * 64,
            }
        ],
    )

    second = client.post(f"/api/datasets/{dataset_id}/index-builds").json()
    deadline = time.monotonic() + 5
    while second["state"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.02)
        second = client.get(f"/api/index-builds/{second['id']}").json()
    second_manifest = client.get(f"/api/datasets/{dataset_id}/index-manifest").json()

    assert first["state"] == "succeeded"
    assert second["state"] == "succeeded"
    assert first_manifest["index_version"] != second_manifest["index_version"]
    assert second_manifest["counts"]["image_chunks"] == 1


def test_index_runtime_endpoint_reports_loaded_active_bundle(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        dataset_id = _published_dataset(client)
        job = client.post(f"/api/datasets/{dataset_id}/index-builds").json()
        deadline = time.monotonic() + 5
        while job["state"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            job = client.get(f"/api/index-builds/{job['id']}").json()

        response = client.get("/api/index-runtime")

    assert response.status_code == 200
    assert response.json()["mode"] == "on"
    status = response.json()["datasets"][dataset_id]
    assert status["status"] == "ready"
    assert status["index_version"] == job["index_version"]
    assert status["child_chunks"] >= 1


def test_previously_active_bundle_can_be_rolled_back_atomically(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        dataset_id = _published_dataset(client)
        first = client.post(f"/api/datasets/{dataset_id}/index-builds").json()
        deadline = time.monotonic() + 5
        while first["state"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            first = client.get(f"/api/index-builds/{first['id']}").json()

        document = app.state.knowledge_service.repository.list_document_refs(
            dataset_id=dataset_id
        )[0]
        child = app.state.knowledge_service.repository.list_children(
            document_id=document.id,
            index_version=document.published_version,
        )[0]
        app.state.knowledge_service.repository.replace_image_chunks(
            document_id=document.id,
            dataset_id=dataset_id,
            index_version=document.published_version,
            items=[
                {
                    "id": "rollback-image",
                    "asset_id": "rollback-asset",
                    "image_id": "rollback",
                    "manual_name": "回滚测试",
                    "chapter_title": "候选图片",
                    "caption": "候选图片",
                    "retrieval_text": "候选图片",
                    "related_parent_ids": [child.parent_id],
                    "related_child_ids": [child.id],
                    "confidence": 0.9,
                    "content_hash": "e" * 64,
                }
            ],
        )
        second = client.post(f"/api/datasets/{dataset_id}/index-builds").json()
        deadline = time.monotonic() + 5
        while second["state"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            second = client.get(f"/api/index-builds/{second['id']}").json()

        assert first["index_version"] != second["index_version"]
        rolled_back = client.post(
            f"/api/datasets/{dataset_id}/index-manifests/{first['index_version']}/activate"
        )
        active = client.get(f"/api/datasets/{dataset_id}/index-manifest")
        runtime = client.get("/api/index-runtime")

    assert rolled_back.status_code == 200
    assert rolled_back.json()["index_version"] == first["index_version"]
    assert active.json()["index_version"] == first["index_version"]
    assert runtime.json()["datasets"][dataset_id]["index_version"] == first["index_version"]


def test_system_dataset_changed_bundle_stays_candidate_without_release_evidence(tmp_path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        dataset_id = _published_dataset(client)
        with app.state.database.session() as session:
            dataset = session.get(DatasetRecord, dataset_id)
            assert dataset is not None
            dataset.is_system = True
            session.add(dataset)
            session.commit()

        first = client.post(f"/api/datasets/{dataset_id}/index-builds").json()
        deadline = time.monotonic() + 5
        while first["state"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            first = client.get(f"/api/index-builds/{first['id']}").json()

        document = app.state.knowledge_service.repository.list_document_refs(
            dataset_id=dataset_id
        )[0]
        child = app.state.knowledge_service.repository.list_children(
            document_id=document.id,
            index_version=document.published_version,
        )[0]
        app.state.knowledge_service.repository.replace_image_chunks(
            document_id=document.id,
            dataset_id=dataset_id,
            index_version=document.published_version,
            items=[
                {
                    "id": "gated-image",
                    "asset_id": "gated-asset",
                    "image_id": "gated",
                    "manual_name": "门禁测试",
                    "chapter_title": "候选图片",
                    "caption": "候选图片",
                    "retrieval_text": "候选图片",
                    "related_parent_ids": [child.parent_id],
                    "related_child_ids": [child.id],
                    "confidence": 0.9,
                    "content_hash": "f" * 64,
                }
            ],
        )
        second = client.post(f"/api/datasets/{dataset_id}/index-builds").json()
        deadline = time.monotonic() + 5
        while second["state"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
            second = client.get(f"/api/index-builds/{second['id']}").json()

        active = client.get(f"/api/datasets/{dataset_id}/index-manifest")
        candidate = client.get(
            f"/api/datasets/{dataset_id}/index-manifests/{second['index_version']}"
        )
        activation = client.post(
            f"/api/datasets/{dataset_id}/index-manifests/{second['index_version']}/activate"
        )
        app.state.evaluation_service.assess_release_gate = lambda **kwargs: {
            "status": "approved",
            "reason_code": "all_gates_passed",
        }
        approved_activation = client.post(
            f"/api/datasets/{dataset_id}/index-manifests/{second['index_version']}/activate",
            json={"evaluation_run_id": "approved-run", "frozen_score": 0.88375},
        )
        active_after_approval = client.get(f"/api/datasets/{dataset_id}/index-manifest")

    assert first["index_version"] != second["index_version"]
    assert active.json()["index_version"] == first["index_version"]
    assert candidate.status_code == 200
    assert candidate.json()["index_version"] == second["index_version"]
    assert candidate.json()["approval_status"] == "awaiting_approval"
    assert activation.status_code == 409
    assert "frozen_result_missing" in activation.text
    assert approved_activation.status_code == 200
    assert active_after_approval.json()["index_version"] == second["index_version"]
