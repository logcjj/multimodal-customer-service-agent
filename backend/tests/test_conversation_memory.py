from __future__ import annotations

from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from app.agents.memory_curator import MemoryCuratorAgent
from app.contracts.models import (
    ClarificationRequest,
    ConversationTurnView,
    ModelKind,
    RoutingIntent,
    VisualContext,
)
from app.conversations.store import ConversationStore
from app.runtime.conversation_memory import (
    ConversationMemoryService,
    PendingClarificationState,
)
from app.storage.database import Database


def make_service(tmp_path):
    database = Database(tmp_path)
    store = ConversationStore(database)
    return store, ConversationMemoryService(database, store)


def begin(
    store: ConversationStore,
    conversation_id: str,
    owner_id: str,
    request_id: str,
    text: str,
):
    return store.begin_turn(
        conversation_id,
        owner_id,
        request_id,
        text,
        [],
    )


def test_latest_explicit_user_correction_supersedes_old_error_code(tmp_path) -> None:
    store, memory = make_service(tmp_path)
    first = begin(store, "c1", "owner-a", "req-1", "洗衣机显示 E03")
    memory.record_user_turn("c1", "owner-a", first.id, first.user_text)
    second = begin(store, "c1", "owner-a", "req-2", "更正一下，不是 E03，是 E30")
    memory.record_user_turn("c1", "owner-a", second.id, second.user_text)

    context = memory.load_context("c1", "owner-a", "现在怎么处理")

    assert context is not None
    assert context.active_value("error_code") == "E30"
    assert context.superseded_values("error_code") == ["E03"]
    active = context.active_slot("error_code")
    assert active is not None
    assert active.source_turn_id == second.id
    assert active.source_kind == "explicit_user"


def test_visual_code_does_not_override_explicit_user_code(tmp_path) -> None:
    store, memory = make_service(tmp_path)
    first = begin(store, "c1", "owner-a", "req-1", "用户确认错误码是 E03")
    memory.record_user_turn("c1", "owner-a", first.id, first.user_text)
    second = begin(store, "c1", "owner-a", "req-2", "我再上传一张图片")
    memory.record_user_turn(
        "c1",
        "owner-a",
        second.id,
        second.user_text,
        VisualContext(
            detected_codes=["E30"],
            provider_status={"ocr": "ok"},
            confidence=0.71,
        ),
    )

    context = memory.load_context("c1", "owner-a", "继续")

    assert context is not None
    assert context.active_value("error_code") == "E03"
    assert "E30" not in context.superseded_values("error_code")


def test_switching_product_stops_injecting_old_model_and_code(tmp_path) -> None:
    store, memory = make_service(tmp_path)
    first = begin(
        store,
        "c1",
        "owner-a",
        "req-1",
        "洗衣机型号 XQG100，错误码 E03",
    )
    memory.record_user_turn("c1", "owner-a", first.id, first.user_text)
    second = begin(
        store,
        "c1",
        "owner-a",
        "req-2",
        "现在改问空气净化器的滤网怎么清洁",
    )
    memory.record_user_turn("c1", "owner-a", second.id, second.user_text)

    context = memory.load_context("c1", "owner-a", "滤网能水洗吗")

    assert context is not None
    assert context.active_value("product") == "空气净化器手册"
    assert context.active_value("model") is None
    assert context.active_value("error_code") is None


def test_context_budget_keeps_complete_turns_and_rolling_summary() -> None:
    now = datetime(2026, 7, 25, tzinfo=UTC)
    turns = [
        ConversationTurnView(
            id=f"turn-{index}",
            conversation_id="c1",
            ordinal=index,
            request_id=f"req-{index}",
            user_text=f"第{index}轮用户问题" + "很长的内容" * 20,
            assistant_text=f"第{index}轮助手回答" + "完整回答" * 20,
            status="completed",
            created_at=now,
            completed_at=now,
        )
        for index in range(1, 5)
    ]

    prompt = ConversationMemoryService.build_prompt_context(
        turns,
        rolling_summary="更早对话摘要：用户正在处理家电问题。",
        current_question="请继续说明",
        budget_tokens=260,
    )

    assert prompt.estimated_tokens <= 260
    assert prompt.rolling_summary.startswith("更早对话摘要")
    assert prompt.included_ordinals
    assert prompt.included_ordinals[-1] == 4
    for ordinal in prompt.included_ordinals:
        assert f"第{ordinal}轮用户问题" in prompt.text
        assert f"第{ordinal}轮助手回答" in prompt.text


def test_prompt_context_does_not_repeat_current_running_turn(tmp_path) -> None:
    store, memory = make_service(tmp_path)
    running = begin(store, "c1", "owner-a", "req-1", "这是当前正在处理的问题")
    memory.record_user_turn("c1", "owner-a", running.id, running.user_text)

    context = memory.load_context(
        "c1",
        "owner-a",
        "这是当前正在处理的问题",
    )

    assert context is not None
    assert "这是当前正在处理的问题" not in context.prompt.text
    assert context.prompt.included_ordinals == []


def test_structured_memory_records_symptom_action_order_and_service_request(
    tmp_path,
) -> None:
    store, memory = make_service(tmp_path)
    texts = [
        "现象是排水时有明显异响",
        "已经尝试过断电重启",
        "订单号是 ORD-20260725-001",
        "我希望申请换货",
    ]
    for index, text in enumerate(texts, start=1):
        turn = begin(store, "c1", "owner-a", f"req-{index}", text)
        memory.record_user_turn("c1", "owner-a", turn.id, turn.user_text)

    context = memory.load_context("c1", "owner-a", "继续处理")

    assert context is not None
    assert context.active_value("symptom") == "排水时有明显异响"
    assert context.active_value("attempted_action") == "断电重启"
    assert context.active_value("order_identifier") == "ORD-20260725-001"
    assert context.active_value("service_request") == "换货"


def test_concurrent_slot_updates_share_conversation_lock_without_lost_state(
    tmp_path,
) -> None:
    store, memory = make_service(tmp_path)
    first = begin(store, "c1", "owner-a", "req-1", "型号是 XQG100")
    second = begin(store, "c1", "owner-a", "req-2", "显示 E03")
    barrier = Barrier(2)

    def record(turn) -> None:
        barrier.wait()
        memory.record_user_turn(
            "c1",
            "owner-a",
            turn.id,
            turn.user_text,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(record, (first, second)))

    context = memory.load_context("c1", "owner-a", "继续")
    assert context is not None
    assert context.active_value("model") == "XQG100"
    assert context.active_value("error_code") == "E03"


def test_state_and_pending_writes_use_conversation_store_lock(
    tmp_path,
    monkeypatch,
) -> None:
    store, memory = make_service(tmp_path)
    turn = begin(store, "c1", "owner-a", "req-1", "型号是 XQG100")
    entered: list[str] = []
    original_lock = store.conversation_lock

    def recording_lock(conversation_id: str):
        entered.append(conversation_id)
        return original_lock(conversation_id)

    monkeypatch.setattr(store, "conversation_lock", recording_lock)
    memory.record_user_turn("c1", "owner-a", turn.id, turn.user_text)
    memory.set_pending_clarification(
        "c1",
        "owner-a",
        ClarificationRequest(
            case_id="case-1",
            field="error_code",
            question="请提供错误码。",
            round=1,
        ),
        original_question="设备报警",
    )
    memory.clear_pending_clarification("c1", "owner-a")
    memory.save_summary("c1", "owner-a", "摘要", through_ordinal=1)

    assert entered == ["c1", "c1", "c1", "c1"]


def test_pending_clarification_is_scoped_to_one_conversation(tmp_path) -> None:
    store, memory = make_service(tmp_path)
    begin(store, "c1", "owner-a", "req-1", "设备一直报警")
    begin(store, "c2", "owner-a", "req-2", "帮我改写通知")
    clarification = ClarificationRequest(
        case_id="case-1",
        field="model",
        question="请提供产品型号。",
        round=1,
        max_rounds=3,
    )

    memory.set_pending_clarification(
        "c1",
        "owner-a",
        clarification,
        original_question="设备一直报警",
    )

    first = memory.load_context("c1", "owner-a", "XQG100")
    second = memory.load_context("c2", "owner-a", "继续")
    assert first is not None and first.pending_clarification is not None
    assert first.pending_clarification.case_id == "case-1"
    assert second is not None and second.pending_clarification is None

    memory.clear_pending_clarification("c1", "owner-a")
    cleared = memory.load_context("c1", "owner-a", "继续")
    assert cleared is not None
    assert cleared.pending_clarification is None


def test_bare_pending_field_values_are_not_mistaken_for_general_topic_switch() -> None:
    general_intent = RoutingIntent(
        initial_route="general_candidate",
        risk_level="low",
        requires_knowledge_check=False,
        reason_code="deterministic-general",
    )
    technical_intent = RoutingIntent(
        initial_route="technical_candidate",
        risk_level="medium",
        requires_knowledge_check=True,
        reason_code="model-technical",
    )

    for field, text, candidate_intent in (
        ("product", "火星牌除味柜", general_intent),
        ("symptom", "一直漏水", technical_intent),
        ("attempted_action", "已经断电重启了", technical_intent),
    ):
        pending = ClarificationRequest(
            case_id=f"case-{field}",
            field=field,
            question="请补充信息。",
            round=1,
        )
        memory_pending = PendingClarificationState(
            **pending.model_dump(),
            original_question="原问题",
        )
        assert ConversationMemoryService.should_end_pending_for_topic_switch(
            memory_pending,
            text,
            candidate_intent=candidate_intent,
        ) is False


def test_unknown_bare_product_reply_is_stored_in_pending_product_slot(tmp_path) -> None:
    store, memory = make_service(tmp_path)
    first = begin(store, "c1", "owner-a", "req-1", "设备一直报警")
    memory.set_pending_clarification(
        "c1",
        "owner-a",
        ClarificationRequest(
            case_id="case-product",
            field="product",
            question="请提供产品名称。",
            round=1,
        ),
        original_question="设备一直报警",
    )
    second = begin(store, "c1", "owner-a", "req-2", "火星牌除味柜")

    memory.record_user_turn("c1", "owner-a", second.id, second.user_text)

    context = memory.load_context("c1", "owner-a", "继续")
    assert context is not None
    assert context.active_value("product") == "火星牌除味柜"


class SummaryGateway:
    def available(self, kind: ModelKind = ModelKind.LLM) -> bool:
        return True

    def generate(self, **kwargs):
        return SimpleNamespace(
            text="用户正在处理洗衣机报警，已确认型号 XQG100。",
            model="summary-test-model",
        )


def test_memory_curator_saves_traceable_rolling_summary(tmp_path) -> None:
    store, memory = make_service(tmp_path)
    for index, text in enumerate(
        [
            "洗衣机一直报警",
            "型号是 XQG100",
            "我还没有看到错误码",
        ],
        start=1,
    ):
        turn = begin(store, "c1", "owner-a", f"req-{index}", text)
        memory.record_user_turn("c1", "owner-a", turn.id, turn.user_text)

    result = MemoryCuratorAgent(memory, SummaryGateway()).run("c1", "owner-a")
    context = memory.load_context("c1", "owner-a", "继续")

    assert result.status == "completed"
    assert result.llm_generated is True
    assert result.model_used == "summary-test-model"
    assert context is not None
    assert context.prompt.rolling_summary == "用户正在处理洗衣机报警，已确认型号 XQG100。"


def test_older_summary_job_cannot_overwrite_newer_summary(tmp_path) -> None:
    store, memory = make_service(tmp_path)
    store.create("owner-a", conversation_id="c1")

    memory.save_summary("c1", "owner-a", "覆盖到第 5 轮的新摘要", through_ordinal=5)
    memory.save_summary("c1", "owner-a", "迟到的第 3 轮旧摘要", through_ordinal=3)

    context = memory.load_context("c1", "owner-a", "继续")
    detail = store.get("c1", "owner-a")
    assert context is not None
    assert detail is not None and detail.state is not None
    assert context.prompt.rolling_summary == "覆盖到第 5 轮的新摘要"
    assert detail.state.summary_through_ordinal == 5


def test_memory_curator_degrades_without_erasing_existing_state(tmp_path) -> None:
    store, memory = make_service(tmp_path)
    turn = begin(store, "c1", "owner-a", "req-1", "型号是 XQG100")
    memory.record_user_turn("c1", "owner-a", turn.id, turn.user_text)

    result = MemoryCuratorAgent(memory, None).run("c1", "owner-a")
    context = memory.load_context("c1", "owner-a", "继续")

    assert result.status == "completed"
    assert result.llm_generated is False
    assert context is not None
    assert context.active_value("model") == "XQG100"


def test_memory_curator_can_run_as_background_job_and_shutdown(tmp_path) -> None:
    store, memory = make_service(tmp_path)
    turn = begin(store, "c1", "owner-a", "req-1", "洗衣机一直报警")
    memory.record_user_turn("c1", "owner-a", turn.id, turn.user_text)
    curator = MemoryCuratorAgent(memory, SummaryGateway())

    result = curator.submit("c1", "owner-a").result(timeout=2)
    curator.shutdown()

    assert result.status == "completed"


def test_memory_curator_records_background_failure_and_shutdown_converges(
    tmp_path,
) -> None:
    _, memory = make_service(tmp_path)
    curator = MemoryCuratorAgent(memory, SummaryGateway())

    future = curator.submit("missing-conversation", "owner-a")
    result = future.result(timeout=2)
    curator.shutdown()

    assert result.status == "failed"
    assert curator.failure_count == 1
    assert curator.last_error == "conversation-not-found"


def test_memory_curator_records_unhandled_background_exception(tmp_path) -> None:
    store, memory = make_service(tmp_path)
    turn = begin(store, "c1", "owner-a", "req-1", "洗衣机一直报警")
    memory.record_user_turn("c1", "owner-a", turn.id, turn.user_text)

    class ExplodingGateway(SummaryGateway):
        def generate(self, **kwargs):
            raise RuntimeError("summary-provider-failed")

    curator = MemoryCuratorAgent(memory, ExplodingGateway())
    future = curator.submit("c1", "owner-a")

    with pytest.raises(RuntimeError, match="summary-provider-failed"):
        future.result(timeout=2)
    curator.shutdown()

    assert curator.failure_count == 1
    assert curator.last_error == "RuntimeError"
