from __future__ import annotations

import json

import pytest
from sqlalchemy.exc import OperationalError

from tests.knowledge_fixtures import create_client_with_manuals
from tests.test_dynamic_orchestrator import routed_generate


def _events(response) -> list[dict[str, object]]:
    return [json.loads(line) for line in response.iter_lines() if line]


def test_begin_turn_database_failure_emits_event_but_answer_still_completes(
    tmp_path,
    monkeypatch,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )

    def fail_begin(*args, **kwargs):
        raise OperationalError("insert turn", {}, RuntimeError("database locked"))

    monkeypatch.setattr(client.app.state.conversations, "begin_turn", fail_begin)

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "question": "把明天下午开会改写正式一点",
            "session_id": "persist-begin",
            "user_id": "owner-a",
        },
    ) as response:
        events = _events(response)

    assert events[-1]["type"] == "run.completed"
    assert events[-1]["payload"]["response"]["route"] == "general_llm"
    assert any(event["type"] == "persistence.failed" for event in events)


def test_complete_turn_database_failure_emits_event_but_answer_still_completes(
    tmp_path,
    monkeypatch,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )

    def fail_complete(*args, **kwargs):
        raise OperationalError("update turn", {}, RuntimeError("database locked"))

    monkeypatch.setattr(client.app.state.conversations, "complete_turn", fail_complete)

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "question": "把明天下午开会改写正式一点",
            "session_id": "persist-complete",
            "user_id": "owner-a",
        },
    ) as response:
        events = _events(response)

    assert events[-1]["type"] == "run.completed"
    assert events[-1]["payload"]["response"]["route"] == "general_llm"
    assert any(event["type"] == "persistence.failed" for event in events)
    detail = client.get(
        "/api/conversations/persist-complete",
        params={"user_id": "owner-a"},
    ).json()
    assert detail["turns"][0]["status"] == "failed"


@pytest.mark.parametrize("failing_method", ["record_user_turn", "load_context"])
def test_layered_memory_failure_after_begin_marks_turn_failed_without_losing_answer(
    tmp_path,
    monkeypatch,
    failing_method: str,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )

    def fail_memory(*args, **kwargs):
        raise OperationalError(
            "update conversation state",
            {},
            RuntimeError("database locked"),
        )

    monkeypatch.setattr(
        client.app.state.conversation_memory,
        failing_method,
        fail_memory,
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "question": "把明天下午开会改写正式一点",
            "session_id": f"persist-{failing_method}",
            "user_id": "owner-a",
        },
    ) as response:
        events = _events(response)

    body = events[-1]["payload"]["response"]
    detail = client.get(
        f"/api/conversations/persist-{failing_method}",
        params={"user_id": "owner-a"},
    ).json()
    event_types = [event["type"] for event in events]

    assert body["route"] == "general_llm"
    assert "persistence.failed" in event_types
    assert "memory.updated" not in event_types
    assert detail["turns"][0]["status"] == "failed"
    assert all(
        span["name"] != "layered_memory"
        for span in body["trace"]["spans"]
    )


def test_fail_turn_persistence_error_does_not_mask_original_execution_error(
    tmp_path,
    monkeypatch,
) -> None:
    client = create_client_with_manuals(tmp_path, rollout_mode="agent_first")

    class OriginalExecutionError(RuntimeError):
        pass

    def fail_execute(*args, **kwargs):
        raise OriginalExecutionError("original execution failure")

    def fail_turn(*args, **kwargs):
        raise OperationalError("update failed turn", {}, RuntimeError("database locked"))

    monkeypatch.setattr(client.app.state.orchestrator, "_execute", fail_execute)
    monkeypatch.setattr(client.app.state.conversations, "fail_turn", fail_turn)

    with pytest.raises(OriginalExecutionError, match="original execution failure"):
        client.post(
            "/api/chat",
            json={
                "question": "触发执行失败",
                "session_id": "persist-original-error",
                "user_id": "owner-a",
            },
        )
