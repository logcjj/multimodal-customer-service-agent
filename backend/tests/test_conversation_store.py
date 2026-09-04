from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy.exc import IntegrityError

from app.contracts.models import (
    AgentResponse,
    AgentTrace,
    Evidence,
    RoutingDecision,
    VerificationReport,
)
from app.conversations.models import (
    ConversationRecord,
    ConversationStateRecord,
    ConversationTurnRecord,
)
from app.conversations.store import ConversationStore
from app.runtime.session_memory import SessionMemoryRecord
from app.storage.database import Database
from sqlmodel import select
from sqlmodel import Session


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 25, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.value += timedelta(seconds=1)
        return self.value


def make_response(
    *,
    request_id: str,
    session_id: str,
    answer: str,
    route: str = "technical",
) -> AgentResponse:
    return AgentResponse(
        request_id=request_id,
        session_id=session_id,
        answer=answer,
        route=route,
        verification=VerificationReport(
            passed=True,
            action="accept",
            confidence=0.9,
        ),
        trace=AgentTrace(
            request_id=request_id,
            session_id=session_id,
            route=route,
            selected_agents=["orchestrator", "verifier"],
            steps=[],
            total_latency_ms=10,
        ),
        routing=RoutingDecision(
            initial_route="technical_candidate",
            final_route="technical_knowledge",
            route_label="技术知识库",
            route_reason="检索到说明书证据",
            coverage_status="covered",
            knowledge_covered=True,
            risk_level="medium",
        ),
    )


def test_conversations_are_isolated_and_ordered_by_latest_activity(tmp_path) -> None:
    store = ConversationStore(Database(tmp_path), now=Clock())
    store.begin_turn("c1", "owner-a", "req-1", "洗衣机 E03 怎么处理", [])
    store.complete_turn(
        "req-1",
        make_response(
            request_id="req-1",
            session_id="c1",
            answer="请先断电并检查排水管。",
        ),
    )
    store.begin_turn("c2", "owner-a", "req-2", "改写会议通知", [])
    store.complete_turn(
        "req-2",
        make_response(
            request_id="req-2",
            session_id="c2",
            answer="兹定于明天下午召开会议。",
            route="general_llm",
        ),
    )

    assert [item.id for item in store.list("owner-a")] == ["c2", "c1"]
    assert store.get("c1", "owner-b") is None
    assert store.list("owner-b") == []

    detail = store.get("c1", "owner-a")
    assert detail is not None
    assert detail.turns[0].user_text == "洗衣机 E03 怎么处理"
    assert detail.turns[0].assistant_text == "请先断电并检查排水管。"
    assert detail.turns[0].response is not None
    assert detail.turns[0].response.routing is not None
    assert detail.turns[0].response.routing.final_route == "technical_knowledge"
    assert detail.updated_at.tzinfo is not None
    assert detail.updated_at.utcoffset() == timedelta(0)
    assert detail.turns[0].created_at.tzinfo is not None


def test_begin_turn_is_idempotent_for_request_id(tmp_path) -> None:
    store = ConversationStore(Database(tmp_path), now=Clock())

    first = store.begin_turn("c1", "owner-a", "req-1", "第一次问题", [])
    second = store.begin_turn("c1", "owner-a", "req-1", "重复请求", [])

    assert second.id == first.id
    detail = store.get("c1", "owner-a")
    assert detail is not None
    assert len(detail.turns) == 1
    assert detail.turns[0].user_text == "第一次问题"


def test_concurrent_create_same_explicit_id_is_idempotent_for_same_owner(
    tmp_path,
) -> None:
    store = ConversationStore(Database(tmp_path))
    workers = 12
    barrier = Barrier(workers)

    def create(_: int):
        barrier.wait()
        return store.create(
            "owner-a",
            conversation_id="shared-conversation",
            title="并发创建",
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        created = list(executor.map(create, range(workers)))

    assert {item.id for item in created} == {"shared-conversation"}
    assert {item.owner_id for item in created} == {"owner-a"}
    assert len(store.list("owner-a")) == 1


def test_concurrent_create_same_explicit_id_cannot_be_claimed_by_other_owner(
    tmp_path,
) -> None:
    store = ConversationStore(Database(tmp_path))
    barrier = Barrier(2)

    def create(owner_id: str):
        barrier.wait()
        try:
            return store.create(owner_id, conversation_id="owner-race")
        except PermissionError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create, ("owner-a", "owner-b")))

    winners = [item for item in results if item is not None]
    assert len(winners) == 1
    winner = winners[0]
    assert store.get("owner-race", winner.owner_id) is not None
    loser = "owner-b" if winner.owner_id == "owner-a" else "owner-a"
    assert store.get("owner-race", loser) is None


def test_explicit_conversation_create_uses_conversation_stripe_lock(
    tmp_path,
    monkeypatch,
) -> None:
    store = ConversationStore(Database(tmp_path))
    entered: list[str] = []
    original_lock = store.conversation_lock

    def recording_lock(conversation_id: str):
        entered.append(conversation_id)
        return original_lock(conversation_id)

    monkeypatch.setattr(store, "conversation_lock", recording_lock)

    store.create("owner-a", conversation_id="locked-create")

    assert entered == ["locked-create"]


@pytest.mark.parametrize(
    ("winner_owner", "caller_allowed"),
    [("owner-a", True), ("owner-b", False)],
)
def test_create_rereads_owner_in_new_transaction_after_unique_conflict(
    tmp_path,
    monkeypatch,
    winner_owner: str,
    caller_allowed: bool,
) -> None:
    database = Database(tmp_path)
    store = ConversationStore(database)
    original_session = database.session
    injected = False
    opened_sessions = 0

    def conflicting_session():
        nonlocal injected, opened_sessions
        opened_sessions += 1
        session = original_session()
        original_commit = session.commit

        def commit() -> None:
            nonlocal injected
            if not injected:
                injected = True
                with original_session() as winner_session:
                    winner_session.add(
                            ConversationRecord(
                                id="cross-instance-create",
                                owner_id=winner_owner,
                            title="先到实例",
                        )
                    )
                    winner_session.commit()
                raise IntegrityError(
                    "insert conversation",
                    {},
                    RuntimeError("duplicate key"),
                )
            original_commit()

        session.commit = commit
        return session

    monkeypatch.setattr(database, "session", conflicting_session)

    if caller_allowed:
        created = store.create(
            "owner-a",
            conversation_id="cross-instance-create",
            title="后到实例",
        )
        assert created.owner_id == "owner-a"
        assert created.title == "先到实例"
    else:
        with pytest.raises(PermissionError):
            store.create(
                "owner-a",
                conversation_id="cross-instance-create",
                title="不得抢占",
            )
    assert opened_sessions >= 2


def test_duplicate_request_id_cannot_cross_conversation_owner_boundary(tmp_path) -> None:
    store = ConversationStore(Database(tmp_path), now=Clock())
    store.begin_turn("c1", "owner-a", "req-1", "用户 A 的问题", [])

    with pytest.raises(PermissionError):
        store.begin_turn("c2", "owner-b", "req-1", "用户 B 的问题", [])


def test_concurrent_turns_in_one_conversation_receive_unique_ordinals(tmp_path) -> None:
    store = ConversationStore(Database(tmp_path))
    store.create("owner-a", conversation_id="c1")
    workers = 12
    barrier = Barrier(workers)

    def begin(index: int) -> int:
        barrier.wait()
        return store.begin_turn(
            "c1",
            "owner-a",
            f"req-{index}",
            f"并发问题 {index}",
            [],
        ).ordinal

    with ThreadPoolExecutor(max_workers=workers) as executor:
        ordinals = list(executor.map(begin, range(workers)))

    assert sorted(ordinals) == list(range(1, workers + 1))


def test_concurrent_complete_turns_do_not_lose_message_count(tmp_path) -> None:
    store = ConversationStore(Database(tmp_path))
    store.begin_turn("c1", "owner-a", "req-1", "问题 1", [])
    store.begin_turn("c1", "owner-a", "req-2", "问题 2", [])
    barrier = Barrier(2)

    def complete(request_id: str) -> None:
        barrier.wait()
        store.complete_turn(
            request_id,
            make_response(
                request_id=request_id,
                session_id="c1",
                answer=f"{request_id} 的回答",
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(complete, ("req-1", "req-2")))

    detail = store.get("c1", "owner-a")
    assert detail is not None
    assert detail.message_count == 4
    assert [turn.status for turn in detail.turns] == ["completed", "completed"]


def test_conversation_lock_registry_has_fixed_number_of_stripes(tmp_path) -> None:
    store = ConversationStore(Database(tmp_path))

    assert getattr(store, "lock_stripe_count", 0) == 64
    for index in range(500):
        with store.conversation_lock(f"conversation-{index}"):
            pass
    assert store.lock_stripe_count == 64


def test_attachment_metadata_whitelist_never_persists_base64(tmp_path) -> None:
    database = Database(tmp_path)
    store = ConversationStore(database, now=Clock())

    store.begin_turn(
        "c1",
        "owner-a",
        "req-1",
        "看一下图片",
        [
            {
                "name": "fault.png",
                "mime_type": "image/png",
                "sha256": "abc",
                "size_bytes": 3,
                "data": "data:image/png;base64,TOP-SECRET",
            }
        ],
    )

    with database.session() as session:
        raw = session.exec(
            select(ConversationTurnRecord).where(
                ConversationTurnRecord.request_id == "req-1"
            )
        ).one().attachment_metadata_json
    assert "fault.png" in raw
    assert "base64" not in raw
    assert "TOP-SECRET" not in raw
    assert '"data"' not in raw


def test_rename_and_delete_apply_only_to_owner(tmp_path) -> None:
    database = Database(tmp_path)
    store = ConversationStore(database, now=Clock())
    store.begin_turn("c1", "owner-a", "req-1", "原始问题", [])

    assert store.rename("c1", "owner-b", "越权标题") is None
    renamed = store.rename("c1", "owner-a", "洗衣机排障")
    assert renamed is not None
    assert renamed.title == "洗衣机排障"
    assert store.delete("c1", "owner-b") is False
    assert store.get("c1", "owner-a") is not None
    assert store.delete("c1", "owner-a") is True
    assert store.get("c1", "owner-a") is None
    with database.session() as session:
        assert session.exec(
            select(ConversationTurnRecord).where(
                ConversationTurnRecord.request_id == "req-1"
            )
        ).first() is None


def test_failed_turn_is_preserved_without_fake_assistant_answer(tmp_path) -> None:
    store = ConversationStore(Database(tmp_path), now=Clock())
    store.begin_turn("c1", "owner-a", "req-1", "网络失败的问题", [])

    store.fail_turn("req-1", "provider_timeout")

    detail = store.get("c1", "owner-a")
    assert detail is not None
    assert detail.turns[0].status == "failed"
    assert detail.turns[0].assistant_text == ""
    assert detail.turns[0].error_code == "provider_timeout"


def test_complete_turn_persists_stable_citation_locators_without_evidence_text(
    tmp_path,
) -> None:
    database = Database(tmp_path)
    store = ConversationStore(database, now=Clock())
    store.begin_turn("c1", "owner-a", "req-1", "洗衣机 E03 怎么处理", [])
    response = make_response(
        request_id="req-1",
        session_id="c1",
        answer="请检查排水管。",
    ).model_copy(
        update={
            "citations": [
                Evidence(
                    evidence_id="manual:e03",
                    source_type="manual",
                    title="E03 排水故障",
                    text="这段证据正文不应写入引用审计索引",
                    dataset_id="manuals",
                    document_id="washer-manual",
                    document_version="v2",
                    section_id="section-e03",
                    parent_id="section-e03",
                    image_chunk_ids=["image-e03"],
                    asset_ids=["asset-e03"],
                    chapter_title="排水故障",
                    page_start=12,
                    locator_label="第 12 页 / E03 排水故障",
                )
            ]
        }
    )

    store.complete_turn("req-1", response)

    with database.session() as session:
        state = session.get(ConversationStateRecord, "c1")
        assert state is not None
        refs = json.loads(state.evidence_refs_json)
        turn = session.exec(
            select(ConversationTurnRecord).where(
                ConversationTurnRecord.request_id == "req-1"
            )
        ).one()
        raw_response = turn.response_json or ""
    assert refs == [
        {
            "evidence_id": "manual:e03",
            "source_type": "manual",
            "title": "E03 排水故障",
            "dataset_id": "manuals",
            "document_id": "washer-manual",
            "file_id": None,
            "document_version": "v2",
            "parent_id": "section-e03",
            "page_start": 12,
            "page_end": None,
            "locator_label": "第 12 页 / E03 排水故障",
            "image_chunk_ids": ["image-e03"],
        }
    ]
    assert "证据正文" not in state.evidence_refs_json
    assert "这段证据正文不应写入引用审计索引" not in raw_response

    detail = store.get("c1", "owner-a")
    assert detail is not None
    restored = detail.turns[0].response
    assert restored is not None
    assert restored.citations[0].text == ""
    assert restored.citations[0].document_id == "washer-manual"
    assert restored.citations[0].parent_id == "section-e03"
    assert restored.citations[0].section_id == "section-e03"
    assert restored.citations[0].image_chunk_ids == ["image-e03"]
    assert restored.citations[0].asset_ids == ["asset-e03"]
    assert restored.citations[0].page_start == 12
    assert restored.citations[0].locator_label == "第 12 页 / E03 排水故障"

    database_path = str(database.engine.url.database)
    with sqlite3.connect(database_path) as connection:
        database_dump = "\n".join(connection.iterdump())
    assert "这段证据正文不应写入引用审计索引" not in database_dump


def test_delete_atomically_removes_conversation_and_legacy_session_record(
    tmp_path,
) -> None:
    database = Database(tmp_path)
    store = ConversationStore(database, now=Clock())
    store.begin_turn("c1", "owner-a", "req-1", "问题", [])
    with database.session() as session:
        session.add(
            SessionMemoryRecord(
                session_id="c1",
                user_id="owner-a",
                expires_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
        )
        session.commit()

    assert store.delete("c1", "owner-a") is True

    with database.session() as session:
        assert session.get(SessionMemoryRecord, "c1") is None
        assert session.get(ConversationStateRecord, "c1") is None


def test_postgres_style_row_locks_follow_conversation_then_turn_and_state_order(
    tmp_path,
    monkeypatch,
) -> None:
    store = ConversationStore(Database(tmp_path), now=Clock())
    store.begin_turn("c1", "owner-a", "req-complete", "问题", [])
    store.begin_turn("c1", "owner-a", "req-fail", "另一个问题", [])
    original_exec = Session.exec
    locked_entities: list[str] = []

    def recording_exec(session, statement, *args, **kwargs):
        if getattr(statement, "_for_update_arg", None) is not None:
            entities = [
                description.get("entity")
                for description in getattr(statement, "column_descriptions", [])
            ]
            locked_entities.extend(
                entity.__name__ for entity in entities if entity is not None
            )
        return original_exec(session, statement, *args, **kwargs)

    monkeypatch.setattr(Session, "exec", recording_exec)

    store.complete_turn(
        "req-complete",
        make_response(request_id="req-complete", session_id="c1", answer="回答"),
    )
    assert locked_entities[:3] == [
        "ConversationRecord",
        "ConversationTurnRecord",
        "ConversationStateRecord",
    ]

    locked_entities.clear()
    store.fail_turn("req-fail", "provider_error")
    assert locked_entities[:2] == [
        "ConversationRecord",
        "ConversationTurnRecord",
    ]

    locked_entities.clear()
    assert store.delete("c1", "owner-a") is True
    assert locked_entities[:3] == [
        "ConversationRecord",
        "ConversationTurnRecord",
        "ConversationStateRecord",
    ]
