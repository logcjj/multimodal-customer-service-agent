from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from sqlmodel import select

from app.contracts.models import Evidence, VisualContext
from app.runtime.session_memory import SessionMemoryRecord, SessionMemoryStore
from app.storage.database import Database


def test_session_memory_persists_across_store_instances_without_image_bytes(tmp_path) -> None:
    database = Database(tmp_path)
    store = SessionMemoryStore(database, ttl_seconds=3600)
    store.save_turn(
        session_id="session-1",
        user_id="user-1",
        question="洗衣机 E03 怎么处理？",
        products=["washing-machine"],
        model_codes=["E03"],
        intent="technical",
        answer="请先关闭电源并检查排水管。",
        evidence=[
            Evidence(
                evidence_id="manual:e03",
                source_type="manual",
                title="E03 排水故障",
                text="请先关闭电源并检查排水管。",
                document_id="doc-1",
                parent_id="parent-1",
            )
        ],
        visual_context=VisualContext(
            image_hashes=["hash-only"],
            ocr_text="E03",
            detected_codes=["E03"],
            provider_status={"ocr": "ok", "vlm": "ok"},
            confidence=0.9,
        ),
        missing_information=[],
        risk_state="verified",
    )

    loaded = SessionMemoryStore(database, ttl_seconds=3600).load("session-1", "user-1")

    assert loaded is not None
    assert loaded.turn_count == 1
    assert loaded.products == ["washing-machine"]
    assert loaded.model_codes == ["E03"]
    assert loaded.evidence_refs[0].evidence_id == "manual:e03"
    with database.session() as session:
        record = session.exec(select(SessionMemoryRecord)).one()
    serialized = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
    assert "base64" not in serialized.lower()
    assert "data:image" not in serialized.lower()


def test_expired_memory_is_ignored_and_deleted(tmp_path) -> None:
    current = datetime(2026, 7, 24, tzinfo=UTC)
    database = Database(tmp_path)
    store = SessionMemoryStore(database, ttl_seconds=300, now=lambda: current)
    store.save_turn(
        session_id="expired",
        user_id=None,
        question="问题",
        products=[],
        model_codes=[],
        intent="technical",
        answer="回答",
        evidence=[],
        visual_context=None,
        missing_information=[],
        risk_state="unknown",
    )
    later = SessionMemoryStore(
        database,
        ttl_seconds=300,
        now=lambda: current + timedelta(seconds=301),
    )

    assert later.load("expired", None) is None
    assert later.delete("expired", None) is False


def test_load_relevant_keeps_followup_context_but_rejects_product_switch(tmp_path) -> None:
    database = Database(tmp_path)
    store = SessionMemoryStore(database, ttl_seconds=3600)
    store.save_turn(
        session_id="followup",
        user_id=None,
        question="洗衣机 E03 怎么处理？",
        products=["washing-machine"],
        model_codes=["E03"],
        intent="technical",
        answer="请先断电。",
        evidence=[],
        visual_context=None,
        missing_information=[],
        risk_state="verified",
    )

    assert store.load_relevant("followup", None, "那还能继续使用吗？") is not None
    assert store.load_relevant("followup", None, "空气净化器滤网怎么清洗？") is None


def test_owned_request_cannot_read_delete_or_claim_legacy_anonymous_memory(
    tmp_path,
) -> None:
    database = Database(tmp_path)
    store = SessionMemoryStore(database, ttl_seconds=3600)
    store.save_turn(
        session_id="legacy-anonymous",
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

    assert store.load("legacy-anonymous", "user-a") is None
    assert store.delete("legacy-anonymous", "user-a") is False
    assert store.load("legacy-anonymous", None) is not None

    try:
        store.save_turn(
            session_id="legacy-anonymous",
            user_id="user-a",
            question="尝试认领",
            products=[],
            model_codes=[],
            intent="technical",
            answer="不应写入",
            evidence=[],
            visual_context=None,
            missing_information=[],
            risk_state="unknown",
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("带 owner 的请求不得认领匿名旧会话")

    legacy = store.load("legacy-anonymous", None)
    assert legacy is not None
    assert legacy.user_id is None
    assert legacy.turn_count == 1
