from __future__ import annotations

from tests.knowledge_fixtures import create_client_with_manuals


def test_session_api_returns_redacted_memory_and_delete_is_idempotent(tmp_path) -> None:
    client = create_client_with_manuals(tmp_path, rollout_mode="agent_first")
    response = client.post(
        "/api/chat",
        json={
            "question": "洗衣机 E03 怎么处理？",
            "session_id": "api-session",
            "user_id": "user-1",
        },
    )
    assert response.status_code == 200

    loaded = client.get("/api/sessions/api-session", params={"user_id": "user-1"})
    assert loaded.status_code == 200
    body = loaded.json()
    assert body["turn_count"] == 1
    assert "E03" in body["model_codes"]
    assert "base64" not in loaded.text.lower()
    assert client.get("/api/sessions/api-session", params={"user_id": "other"}).status_code == 404

    assert client.delete("/api/sessions/api-session", params={"user_id": "user-1"}).status_code == 204
    assert client.delete("/api/sessions/api-session", params={"user_id": "user-1"}).status_code == 204


def test_session_api_owner_cannot_read_or_delete_legacy_anonymous_record(
    tmp_path,
) -> None:
    client = create_client_with_manuals(tmp_path, rollout_mode="agent_first")
    client.app.state.session_memory.save_turn(
        session_id="legacy-api-anonymous",
        user_id=None,
        question="旧版匿名问题",
        products=[],
        model_codes=[],
        intent="technical",
        answer="旧版匿名回答",
        evidence=[],
        visual_context=None,
        missing_information=[],
        risk_state="unknown",
    )

    assert client.get(
        "/api/sessions/legacy-api-anonymous",
        params={"user_id": "user-a"},
    ).status_code == 404
    assert client.delete(
        "/api/sessions/legacy-api-anonymous",
        params={"user_id": "user-a"},
    ).status_code == 204

    legacy = client.get("/api/sessions/legacy-api-anonymous")
    assert legacy.status_code == 200
    assert legacy.json()["user_id"] is None
