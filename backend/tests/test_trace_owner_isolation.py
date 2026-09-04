from __future__ import annotations

import sqlite3

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.contracts.models import AgentTrace
from app.main import create_app
from app.observability.traces import TraceStore
from app.storage.database import Database


def _create_legacy_trace_database(tmp_path) -> None:
    database_path = tmp_path / "aka_multi_agent.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE tracerecord (
                request_id VARCHAR NOT NULL PRIMARY KEY,
                session_id VARCHAR NOT NULL,
                route VARCHAR NOT NULL,
                selected_agents_json VARCHAR NOT NULL,
                steps_json VARCHAR NOT NULL,
                spans_json VARCHAR NOT NULL DEFAULT '[]',
                fallback_reason VARCHAR,
                total_latency_ms INTEGER NOT NULL,
                created_at DATETIME NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO tracerecord VALUES (
                'legacy-request', 'legacy-session', 'technical_knowledge',
                '[]', '[]', '[]', NULL, 1, '2026-07-25 00:00:00'
            )
            """
        )
        connection.commit()


def test_trace_list_and_detail_are_isolated_by_anonymous_client_owner(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path, rollout_mode="agent_first"))
    first = client.post(
        "/api/chat",
        json={"question": "今天天气怎么样", "user_id": "owner-a"},
    ).json()
    second = client.post(
        "/api/chat",
        json={"question": "把会议通知改写正式一点", "user_id": "owner-b"},
    ).json()

    missing_owner = client.get("/api/traces")
    owner_a = client.get("/api/traces", params={"user_id": "owner-a"})
    owner_b_from_header = client.get(
        "/api/traces",
        headers={"X-Client-ID": "owner-b"},
    )
    cross_owner = client.get(
        f"/api/traces/{second['request_id']}",
        params={"user_id": "owner-a"},
    )
    own_detail = client.get(
        f"/api/traces/{first['request_id']}",
        params={"user_id": "owner-a"},
    )

    assert missing_owner.status_code == 422
    assert [item["request_id"] for item in owner_a.json()] == [first["request_id"]]
    assert [item["request_id"] for item in owner_b_from_header.json()] == [
        second["request_id"]
    ]
    assert cross_owner.status_code == 404
    assert own_detail.status_code == 200


def test_existing_sqlite_trace_table_gets_non_claimable_owner_column(tmp_path) -> None:
    _create_legacy_trace_database(tmp_path)
    database_path = tmp_path / "aka_multi_agent.db"

    database = Database(tmp_path)

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(tracerecord)")
        }
        owner_id = connection.execute(
            "SELECT owner_id FROM tracerecord WHERE request_id = 'legacy-request'"
        ).fetchone()[0]
    assert "owner_id" in columns
    assert owner_id == "__legacy_anonymous__"


def test_trace_api_rejects_reserved_legacy_owner_from_external_request(tmp_path) -> None:
    _create_legacy_trace_database(tmp_path)
    client = TestClient(create_app(data_dir=tmp_path, rollout_mode="agent_first"))

    listed = client.get(
        "/api/traces",
        params={"user_id": "__legacy_anonymous__"},
    )
    detail = client.get(
        "/api/traces/legacy-request",
        params={"user_id": "__legacy_anonymous__"},
    )

    assert listed.status_code == 422
    assert detail.status_code == 422


def test_trace_store_rejects_reserved_owner_without_breaking_ordinary_owner(
    tmp_path,
) -> None:
    _create_legacy_trace_database(tmp_path)
    store = TraceStore(Database(tmp_path))

    with pytest.raises(HTTPException) as list_error:
        store.list(TraceStore.LEGACY_ANONYMOUS_OWNER)
    with pytest.raises(HTTPException) as detail_error:
        store.get("legacy-request", TraceStore.LEGACY_ANONYMOUS_OWNER)

    assert list_error.value.status_code == 422
    assert detail_error.value.status_code == 422

    ordinary_trace = AgentTrace(
        request_id="ordinary-request",
        session_id="ordinary-session",
        route="general",
        selected_agents=["general"],
        steps=[],
        total_latency_ms=1,
    )
    store.save(ordinary_trace, owner_id="owner-a")

    assert [item.request_id for item in store.list("owner-a")] == [
        "ordinary-request"
    ]
    assert store.get("ordinary-request", "owner-a").request_id == "ordinary-request"
