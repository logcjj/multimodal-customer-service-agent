from __future__ import annotations

import json
from time import monotonic

from app.observability.metrics import MetricsRegistry
from tests.knowledge_fixtures import create_client_with_manuals


def test_metrics_capture_stage_latency_modes_and_downgrades_without_content() -> None:
    registry = MetricsRegistry(clock=monotonic)
    tracker = registry.start_request()
    tracker.observe_event({"type": "run.started", "status": "running", "payload": {}})
    tracker.observe_event(
        {
            "type": "retrieval.completed",
            "status": "completed",
            "payload": {"mode": "hybrid-rerank"},
        }
    )
    tracker.observe_event(
        {
            "type": "generation.completed",
            "status": "completed",
            "payload": {"provider_latency_ms": 120},
        }
    )
    tracker.observe_event(
        {"type": "answer.delta", "status": "running", "payload": {"delta": "敏感答案"}}
    )
    tracker.observe_event(
        {
            "type": "ocr.completed",
            "status": "failed",
            "payload": {"provider_status": "provider_error"},
        }
    )
    tracker.complete(success=True)

    payload = registry.snapshot()
    serialized = json.dumps(payload, ensure_ascii=False)
    assert payload["request_count"] == 1
    assert payload["retrieval_modes"] == {"hybrid-rerank": 1}
    assert payload["downgrade_count"] == 1
    assert payload["provider_latency_ms"]["count"] == 1
    assert payload["time_to_first_event_ms"]["count"] == 1
    assert payload["time_to_first_answer_delta_ms"]["count"] == 1
    assert "敏感答案" not in serialized
    assert "question" not in serialized.lower()


def test_metrics_api_reports_aggregates_after_chat_without_user_content(tmp_path) -> None:
    client = create_client_with_manuals(tmp_path, rollout_mode="agent_first")
    question = "洗衣机 E03 怎么处理？"
    assert client.post("/api/chat", json={"question": question}).status_code == 200

    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.json()["request_count"] >= 1
    assert response.json()["retrieval_modes"] == {"lexical-only": 1}
    assert response.json()["time_to_first_event_ms"]["count"] == 1
    assert question not in response.text
