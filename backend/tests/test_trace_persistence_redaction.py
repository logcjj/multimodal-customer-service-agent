from __future__ import annotations

from sqlmodel import select

from app.contracts.models import AgentTrace, TraceSpan
from app.conversations.models import ConversationTurnRecord
from app.conversations.store import ConversationStore
from app.observability.traces import TraceRecord, TraceStore
from app.storage.database import Database
from tests.test_conversation_store import Clock, make_response


ORDER_ID = "ORDER-20260725-998877"
SERIAL_ID = "ZX-9988-SECRET"
OCR_QUERY_WITH_DUPLICATED_SERIAL = (
    f"图片可见信息：{SERIAL_ID} S/N: {SERIAL_ID}"
)


def sensitive_trace(request_id: str, session_id: str) -> AgentTrace:
    return AgentTrace(
        request_id=request_id,
        session_id=session_id,
        route="technical_knowledge",
        selected_agents=["knowledge"],
        steps=[],
        spans=[
            TraceSpan(
                name="query_rewrite",
                input_summary=f"查询订单号：{ORDER_ID}",
                output_summary=OCR_QUERY_WITH_DUPLICATED_SERIAL,
                attributes={
                    "search_query": (
                        f"订单号 {ORDER_ID} 的设备怎么维修\n"
                        f"{OCR_QUERY_WITH_DUPLICATED_SERIAL}"
                    ),
                    "parent_query": f"S/N: {SERIAL_ID}",
                    "ocr_text": f"铭牌序列号：{SERIAL_ID}",
                    "detected_numbers": ["13800138000", "220"],
                },
            )
        ],
        total_latency_ms=1,
    )


def test_trace_store_redacts_identifiers_only_in_persisted_copy(tmp_path) -> None:
    database = Database(tmp_path)
    store = TraceStore(database)
    trace = sensitive_trace("req-trace", "c1")

    store.save(trace, owner_id="owner-a")

    assert ORDER_ID in trace.spans[0].input_summary
    assert trace.spans[0].output_summary.count(SERIAL_ID) == 2
    assert SERIAL_ID in str(trace.spans[0].attributes)
    with database.session() as session:
        raw = session.get(TraceRecord, "req-trace").spans_json
    assert ORDER_ID not in raw
    assert SERIAL_ID not in raw
    assert "13800138000" not in raw
    assert "220" in raw
    assert "已脱敏" in raw


def test_conversation_response_json_uses_same_trace_redaction(tmp_path) -> None:
    database = Database(tmp_path)
    conversations = ConversationStore(database, now=Clock())
    conversations.begin_turn("c1", "owner-a", "req-ledger", "订单问题", [])
    response = make_response(
        request_id="req-ledger",
        session_id="c1",
        answer="请联系售后。",
    ).model_copy(update={"trace": sensitive_trace("req-ledger", "c1")})

    conversations.complete_turn("req-ledger", response)

    with database.session() as session:
        raw = session.exec(
            select(ConversationTurnRecord).where(
                ConversationTurnRecord.request_id == "req-ledger"
            )
        ).one().response_json
    assert raw is not None
    assert ORDER_ID not in raw
    assert SERIAL_ID not in raw
    assert "13800138000" not in raw
    assert "已脱敏" in raw
