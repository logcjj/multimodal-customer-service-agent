from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from tests.knowledge_fixtures import create_client_with_manuals


def test_chat_creates_persistent_conversation_and_refresh_restores_turns(
    tmp_path,
) -> None:
    client = create_client_with_manuals(tmp_path, rollout_mode="agent_first")

    answer = client.post(
        "/api/chat",
        json={
            "question": "洗衣机 E03 怎么处理",
            "session_id": "c1",
            "user_id": "owner-a",
        },
    )
    detail = client.get(
        "/api/conversations/c1",
        params={"user_id": "owner-a"},
    )

    assert answer.status_code == 200
    assert detail.status_code == 200
    assert detail.json()["turns"][0]["assistant_text"] == answer.json()["answer"]
    assert detail.json()["turns"][0]["response"]["routing"]["final_route"] == (
        "technical_knowledge"
    )


def test_conversation_crud_is_owner_scoped_and_delete_is_idempotent(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path, rollout_mode="agent_first"))

    created = client.post(
        "/api/conversations",
        params={"user_id": "owner-a"},
        json={"id": "manual-c1", "title": "设备排障"},
    )
    renamed = client.patch(
        "/api/conversations/manual-c1",
        params={"user_id": "owner-a"},
        json={"title": "  洗衣机排障  "},
    )

    assert created.status_code == 201
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "洗衣机排障"
    assert client.get(
        "/api/conversations/manual-c1",
        params={"user_id": "owner-b"},
    ).status_code == 404
    assert client.patch(
        "/api/conversations/manual-c1",
        params={"user_id": "owner-b"},
        json={"title": "越权修改"},
    ).status_code == 404

    assert client.delete(
        "/api/conversations/manual-c1",
        params={"user_id": "owner-b"},
    ).status_code == 204
    assert client.get(
        "/api/conversations/manual-c1",
        params={"user_id": "owner-a"},
    ).status_code == 200
    assert client.delete(
        "/api/conversations/manual-c1",
        params={"user_id": "owner-a"},
    ).status_code == 204
    assert client.delete(
        "/api/conversations/manual-c1",
        params={"user_id": "owner-a"},
    ).status_code == 204


def test_conversation_list_is_paginated_and_owner_is_required(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path, rollout_mode="agent_first"))
    for conversation_id in ("c1", "c2", "c3"):
        response = client.post(
            "/api/conversations",
            params={"user_id": "owner-a"},
            json={"id": conversation_id, "title": conversation_id},
        )
        assert response.status_code == 201

    page = client.get(
        "/api/conversations",
        params={"user_id": "owner-a", "offset": 1, "limit": 1},
    )

    assert page.status_code == 200
    assert len(page.json()) == 1
    assert client.get("/api/conversations").status_code == 422


def test_deleting_conversation_also_removes_legacy_session_memory(tmp_path) -> None:
    client = create_client_with_manuals(tmp_path, rollout_mode="agent_first")
    client.post(
        "/api/chat",
        json={
            "question": "洗衣机 E03 怎么处理",
            "session_id": "delete-c1",
            "user_id": "owner-a",
        },
    )
    assert client.get(
        "/api/sessions/delete-c1",
        params={"user_id": "owner-a"},
    ).status_code == 200

    response = client.delete(
        "/api/conversations/delete-c1",
        params={"user_id": "owner-a"},
    )

    assert response.status_code == 204
    assert client.get(
        "/api/sessions/delete-c1",
        params={"user_id": "owner-a"},
    ).status_code == 404


def test_delete_uses_conversation_store_transaction_not_second_session_delete(
    tmp_path,
    monkeypatch,
) -> None:
    client = create_client_with_manuals(tmp_path, rollout_mode="agent_first")
    client.post(
        "/api/chat",
        json={
            "question": "洗衣机 E03 怎么处理",
            "session_id": "atomic-delete",
            "user_id": "owner-a",
        },
    )

    def unexpected_second_delete(*args, **kwargs):
        raise AssertionError("legacy session must be deleted in the store transaction")

    monkeypatch.setattr(client.app.state.session_memory, "delete", unexpected_second_delete)

    response = client.delete(
        "/api/conversations/atomic-delete",
        params={"user_id": "owner-a"},
    )

    assert response.status_code == 204
