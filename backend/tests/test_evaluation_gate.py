from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.evaluation.service import EvaluationService, PublishGateError
from app.knowledge.service import KnowledgeService
from app.storage.database import Database


def test_candidate_cannot_publish_when_locked_cases_regress(tmp_path) -> None:
    database = Database(tmp_path)
    knowledge = KnowledgeService(database)
    service = EvaluationService(database, knowledge)
    case = service.create_case(
        question="E03 如何处理",
        dataset_ids=["missing-dataset"],
        target_parent_ids=["target-parent"],
        reference_answer="先断电",
        required_facts=["断电"],
        locked=True,
    )

    run = service.run(candidate_version="candidate-v2", case_ids=[case.id])

    assert run.passed is False
    assert run.metrics["recall_at_5"] == 0
    with pytest.raises(PublishGateError):
        service.approve(run.id)


def test_metrics_include_recall_mrr_and_fact_coverage(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    dataset_id = client.post("/api/datasets", json={"name": "洗衣机评测知识库"}).json()["id"]
    file_id = client.post(
        "/api/files",
        files={
            "file": (
                "washing-machine.md",
                "# E03 排水故障\n洗衣机显示 E03 时，先关闭电源并检查排水管。".encode(),
                "text/markdown",
            )
        },
    ).json()["id"]
    document_id = client.post(
        f"/api/datasets/{dataset_id}/documents",
        json={"file_id": file_id},
    ).json()["id"]
    version = client.post(f"/api/documents/{document_id}/parse").json()["index_version"]
    client.post(f"/api/datasets/{dataset_id}/publish", json={"index_version": version})
    database = client.app.state.database
    knowledge = client.app.state.knowledge_service
    service = EvaluationService(database, knowledge)
    target = knowledge.retriever([dataset_id]).explain("洗衣机 E03 怎么处理").results[0]
    case = service.create_case(
        question="洗衣机 E03 怎么处理",
        dataset_ids=[target.dataset_id],
        target_parent_ids=[target.parent_id],
        reference_answer="先关闭电源并检查排水管",
        required_facts=["关闭电源", "排水管"],
        locked=True,
    )

    run = service.run(candidate_version="candidate-ok", case_ids=[case.id])

    assert run.passed is True
    assert run.metrics["recall_at_5"] == 1
    assert run.metrics["mrr"] == 1
    assert run.metrics["fact_coverage"] == 1
    assert service.approve(run.id).status == "approved"


def test_locked_cases_block_api_publish_until_matching_run_is_approved(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))
    dataset_id = client.post("/api/datasets", json={"name": "门禁知识库"}).json()["id"]
    file_id = client.post(
        "/api/files",
        files={"file": ("manual.md", b"# E03\nE03 drain filter maintenance.", "text/markdown")},
    ).json()["id"]
    document_id = client.post(
        f"/api/datasets/{dataset_id}/documents",
        json={"file_id": file_id},
    ).json()["id"]
    version = client.post(f"/api/documents/{document_id}/parse").json()["index_version"]
    client.post(f"/api/datasets/{dataset_id}/publish", json={"index_version": version})
    child = client.get(f"/api/documents/{document_id}/chunks").json()["children"][0]
    draft = client.patch(f"/api/chunks/{child['id']}", json={"text": "E03 drain filter maintenance and power off."}).json()
    case = client.post(
        "/api/evaluations/cases",
        json={
            "question": "E03 drain filter",
            "dataset_ids": [dataset_id],
            "target_parent_ids": [],
            "required_facts": ["E03"],
            "locked": True,
        },
    ).json()

    blocked = client.post(
        f"/api/datasets/{dataset_id}/publish",
        json={"index_version": draft["index_version"]},
    )
    run = client.post(
        "/api/evaluations/runs",
        json={"candidate_version": draft["index_version"], "case_ids": [case["id"]]},
    ).json()
    approved = client.post(f"/api/evaluations/runs/{run['id']}/approve").json()
    published = client.post(
        f"/api/datasets/{dataset_id}/publish",
        json={"index_version": draft["index_version"], "evaluation_run_id": approved["id"]},
    )

    assert blocked.status_code == 409
    assert run["passed"] is True
    assert approved["status"] == "approved"
    assert published.status_code == 200
