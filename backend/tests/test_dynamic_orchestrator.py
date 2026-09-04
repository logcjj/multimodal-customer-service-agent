from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.contracts.models import ClarificationRequest, Evidence
from app.knowledge.hybrid import IndexedChild, PublishedHybridRetriever
from app.main import create_app
from tests.knowledge_fixtures import create_client_with_manuals


def routed_generate(kind: str, system: str, user: str, images: list[str]) -> str:
    if "路由分类器" in system:
        if any(term in user for term in ("改写", "翻译", "写一段")):
            return (
                '{"route":"general_candidate","risk":"low",'
                '"reason_code":"writing"}'
            )
        return (
            '{"route":"technical_candidate","risk":"medium",'
            '"reason_code":"product-support"}'
        )
    if "检索查询改写器" in system:
        return "中文：洗衣机 E03 排水故障 | English: washing machine E03 drain"
    if "通用对话智能体" in system:
        return "兹定于明天下午召开会议，请相关人员准时参加。"
    if "Memory Curator" in system:
        return "用户正在处理洗衣机报警问题。"
    return "洗衣机出现 E03 时，请先断电，检查排水管并清理排水过滤器。"


def camera_error_retriever() -> PublishedHybridRetriever:
    text = (
        "Err 02 means there is a problem with the CF card; remove and re-insert it, "
        "format it, or use another CF card. Err CF means the CF card cannot be read "
        "or written. no CF means no CF card is installed. FuLL CF means the CF card "
        "is full."
    )
    return PublishedHybridRetriever(
        [
            IndexedChild(
                child_id="camera-error-codes",
                parent_id="camera-errors",
                dataset_id="v6-manuals",
                document_id="camera-manual",
                document_version="v6-import-v1",
                file_id="camera-file",
                document_name="Camera.json",
                document_mime_type="application/json",
                title="Camera Error Code Countermeasures",
                text=text,
                parent_text=text,
                product="Camera",
                page_start=236,
                page_end=236,
            )
        ]
    )


def camera_error_generate(
    kind: str,
    system: str,
    user: str,
    images: list[str],
) -> str:
    if "路由分类器" in system:
        if any(code in user.lower() for code in ("err 02", "err cf", "no cf", "full cf")):
            return (
                '{"route":"general_candidate","risk":"low",'
                '"reason_code":"short-utterance"}'
            )
        return (
            '{"route":"technical_candidate","risk":"medium",'
            '"reason_code":"product-support"}'
        )
    if "检索查询改写器" in system:
        return "中文：相机 CF 卡错误码 | English: Camera CF card error code"
    if "通用对话智能体" in system:
        return "这是一个通用回答，不应在该场景出现。"
    if "Memory Curator" in system:
        return "用户正在排查相机错误码。"
    return (
        "相机显示该错误码时，应按手册检查 CF 卡；Err 02 可重新插卡、"
        "格式化或更换 CF 卡。"
    )


def test_open_question_runs_router_general_and_general_verifier(tmp_path) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )

    body = client.post(
        "/api/chat",
        json={
            "question": "把明天下午开会改写正式一点",
            "session_id": "general-c1",
            "user_id": "owner-a",
        },
    ).json()

    assert body["route"] == "general_llm"
    assert body["routing"]["final_route"] == "general_llm"
    assert body["routing"]["route_label"] == "通用大模型"
    assert body["citations"] == []
    assert body["verification"]["passed"] is True
    assert body["answer"].startswith("兹定于")
    assert body["trace"]["selected_agents"] == [
        "orchestrator",
        "router",
        "general",
        "verifier",
        "memory-curator",
    ]
    assert any(
        span["name"] == "intent_routing" for span in body["trace"]["spans"]
    )
    general_span = next(
        span for span in body["trace"]["spans"] if span["name"] == "general_answer"
    )
    assert general_span["attributes"]["model_used"] == "injected-test-model"


def test_unknown_high_risk_repair_never_reaches_general_agent(tmp_path) -> None:
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
            llm_generate_func=routed_generate,
        )
    )

    body = client.post(
        "/api/chat",
        json={
            "question": "火星牌 ZX999 冒烟了，怎么拆开自己维修",
            "session_id": "unsafe-c1",
            "user_id": "owner-a",
        },
    ).json()

    assert body["routing"]["final_route"] == "safe_handoff"
    assert body["routing"]["coverage_status"] == "unsafe_uncovered"
    assert body["citations"] == []
    assert "general" not in body["trace"]["selected_agents"]
    assert "拆机步骤" not in body["answer"]
    assert "停止使用" in body["answer"]


@pytest.mark.parametrize(
    "question",
    [
        "收到后发现洗衣机冒烟，我要退款",
        "收到后发现设备漏电，我要退货",
        "收到后发现电池鼓包，要求退货退款",
    ],
)
def test_high_risk_fault_plus_refund_stays_mixed_and_safety_answer_wins(
    tmp_path,
    question: str,
) -> None:
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
            llm_generate_func=routed_generate,
        )
    )

    body = client.post("/api/chat", json={"question": question}).json()

    assert body["routing"]["initial_route"] == "mixed_candidate"
    assert body["routing"]["final_route"] == "safe_handoff"
    assert body["routing"]["risk_level"] == "high"
    assert "停止使用" in body["answer"]
    assert "general" not in body["trace"]["selected_agents"]


def test_clarification_rounds_retrieve_after_model_and_code_are_supplied(
    tmp_path,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )
    common = {"session_id": "clarify-c1", "user_id": "owner-a"}

    first = client.post(
        "/api/chat",
        json={"question": "洗衣机一直报警怎么办", **common},
    ).json()
    second = client.post(
        "/api/chat",
        json={"question": "型号是 XQG100", **common},
    ).json()
    third = client.post(
        "/api/chat",
        json={"question": "显示 E03", **common},
    ).json()

    assert first["routing"]["final_route"] == "evidence_clarification"
    assert first["routing"]["clarification"]["field"] == "model"
    assert first["routing"]["clarification"]["round"] == 1
    assert second["routing"]["final_route"] == "evidence_clarification"
    assert second["routing"]["clarification"]["field"] == "error_code"
    assert second["routing"]["clarification"]["round"] == 2
    assert third["routing"]["final_route"] == "technical_knowledge"
    assert third["routing"]["knowledge_covered"] is True
    assert third["citations"]
    assert "E03" in third["answer"]


def test_stream_exposes_route_coverage_resolution_and_final_response(tmp_path) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "question": "把明天下午开会改写正式一点",
            "session_id": "stream-general",
            "user_id": "owner-a",
        },
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    types = [event["type"] for event in events]
    assert "route.detected" in types
    assert "knowledge.coverage" in types
    assert "route.resolved" in types
    assert "answer.delta" in types
    assert types[-1] == "run.completed"
    assert events[-1]["payload"]["response"]["routing"]["final_route"] == "general_llm"


def test_mixed_parallel_retrieval_keeps_its_own_explanation_in_event_and_trace(
    tmp_path,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "洗衣机 E03 故障处理不好能退货吗"},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    body = events[-1]["payload"]["response"]
    retrieval = next(
        event for event in events if event["type"] == "retrieval.completed"
    )
    spans = {span["name"]: span for span in body["trace"]["spans"]}

    assert retrieval["payload"]["result_count"] > 0
    assert retrieval["payload"]["query"].startswith("洗衣机 E03")
    assert spans["lexical_retrieval"]["attributes"]["candidate_count"] > 0
    assert spans["parent_aggregation"]["attributes"]["query"].startswith(
        "洗衣机 E03"
    )


def test_stream_emits_clarification_before_resolving_evidence_gap(tmp_path) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "question": "洗衣机一直报警怎么办",
            "session_id": "stream-clarification",
            "user_id": "owner-a",
        },
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    types = [event["type"] for event in events]
    clarification = next(
        event for event in events if event["type"] == "clarification.required"
    )
    assert clarification["payload"]["field"] == "model"
    assert types.index("knowledge.coverage") < types.index("clarification.required")
    assert types.index("clarification.required") < types.index("route.resolved")
    assert types.index("route.resolved") < types.index("run.completed")


def test_legacy_only_keeps_frozen_answer_even_when_router_prefers_general(
    tmp_path,
) -> None:
    frozen = "冻结冠军答案"
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="legacy_only",
            legacy_answer_func=lambda question, images: frozen,
            llm_generate_func=routed_generate,
        )
    )

    body = client.post(
        "/api/chat",
        json={"question": "把明天下午开会改写正式一点"},
    ).json()

    assert body["answer"] == frozen
    assert body["used_legacy"] is True
    assert body["trace"]["fallback_reason"] == "仅守护链路模式"


@pytest.mark.parametrize("rollout_mode", ["champion_guarded", "legacy_only"])
def test_legacy_takeover_has_one_consistent_final_route_everywhere(
    tmp_path,
    rollout_mode: str,
) -> None:
    frozen = "冠军链路：洗衣机 E03 时请先断电并检查排水管。"
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode=rollout_mode,
        legacy_answer_func=lambda question, images: frozen,
        llm_generate_func=routed_generate,
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "洗衣机 E03 怎么处理"},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    body = events[-1]["payload"]["response"]
    final_route_event = [
        event for event in events if event["type"] == "route.resolved"
    ][-1]

    assert body["answer"] == frozen
    assert body["used_legacy"] is True
    assert body["route"] == "technical_knowledge"
    assert body["routing"]["final_route"] == "technical_knowledge"
    assert body["trace"]["route"] == "technical_knowledge"
    assert final_route_event["payload"]["final_route"] == "technical_knowledge"
    assert "守护链路" in body["routing"]["route_reason"]
    assert body["routing"]["clarification"] is None


def test_uncovered_legacy_takeover_keeps_honest_coverage_and_marks_compatibility(
    tmp_path,
) -> None:
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
            legacy_answer_func=lambda question, images: (
                "冠军链路：ZX999 请按设备面板提示完成设置。"
            ),
            llm_generate_func=routed_generate,
        )
    )

    body = client.post(
        "/api/chat",
        json={"question": "火星牌 ZX999 设备应该如何设置"},
    ).json()

    assert body["used_legacy"] is True
    assert body["route"] in {
        "technical_knowledge",
        "customer_service",
        "mixed",
        "evidence_clarification",
        "general_llm",
        "safe_handoff",
        "general_unavailable",
    }
    assert body["routing"]["knowledge_covered"] is False
    assert body["routing"]["coverage_status"] == "unsafe_uncovered"
    assert "守护链路" in body["routing"]["route_label"]
    assert "守护链路" in body["routing"]["route_reason"]


def test_legacy_takeover_does_not_emit_stale_clarification_or_handoff_route(
    tmp_path,
) -> None:
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
            legacy_answer_func=lambda question, images: (
                "冠军链路：请先停止使用并联系官方售后。"
            ),
            llm_generate_func=routed_generate,
        )
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "设备一直报警怎么办"},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    body = events[-1]["payload"]["response"]
    resolved_routes = [
        event["payload"]["final_route"]
        for event in events
        if event["type"] == "route.resolved"
    ]

    assert body["used_legacy"] is True
    assert resolved_routes == ["technical_knowledge"]
    assert "clarification.required" not in [event["type"] for event in events]


def test_router_misclassification_cannot_send_electrical_repair_to_general(
    tmp_path,
) -> None:
    prompts: list[str] = []

    def unsafe_generate(kind: str, system: str, user: str, images: list[str]) -> str:
        prompts.append(system)
        if "路由分类器" in system:
            return (
                '{"route":"general_candidate","risk":"low",'
                '"reason_code":"chat"}'
            )
        return "先关闭总闸，取下旧保险丝并换上新的，然后重新送电测试。"

    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
            llm_generate_func=unsafe_generate,
        )
    )

    body = client.post(
        "/api/chat",
        json={"question": "怎样更换保险丝", "session_id": "unsafe-general"},
    ).json()

    assert body["routing"]["final_route"] == "safe_handoff"
    assert body["routing"]["coverage_status"] == "unsafe_uncovered"
    assert body["citations"] == []
    assert "general" not in body["trace"]["selected_agents"]
    assert not any("通用对话智能体" in prompt for prompt in prompts)


def test_pending_clarification_is_ended_when_user_switches_to_writing(
    tmp_path,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )
    common = {"session_id": "topic-switch", "user_id": "owner-a"}

    first = client.post(
        "/api/chat",
        json={"question": "洗衣机一直报警怎么办", **common},
    ).json()
    second = client.post(
        "/api/chat",
        json={"question": "帮我把明天下午开会改写得正式一些", **common},
    ).json()

    assert first["routing"]["final_route"] == "evidence_clarification"
    assert second["routing"]["initial_route"] == "general_candidate"
    assert second["routing"]["final_route"] == "general_llm"
    assert second["answer"].startswith("兹定于")


def test_pending_clarification_ends_for_unrelated_general_knowledge_topics(
    tmp_path,
) -> None:
    def general_topic_generate(
        kind: str,
        system: str,
        user: str,
        images: list[str],
    ) -> str:
        if "路由分类器" in system and any(
            term in user for term in ("量子计算", "光合作用")
        ):
            return (
                '{"route":"general_candidate","risk":"low",'
                '"reason_code":"general-knowledge"}'
            )
        if "通用对话智能体" in system:
            return "这是一个通用科普问题。"
        return routed_generate(kind, system, user, images)

    for index, topic in enumerate(("量子计算是什么", "讲讲光合作用"), start=1):
        client = create_client_with_manuals(
            tmp_path / str(index),
            rollout_mode="agent_first",
            llm_generate_func=general_topic_generate,
        )
        common = {"session_id": f"topic-switch-{index}", "user_id": "owner-a"}
        first = client.post(
            "/api/chat",
            json={"question": "设备一直报警怎么办", **common},
        ).json()
        second = client.post(
            "/api/chat",
            json={"question": topic, **common},
        ).json()

        assert first["routing"]["final_route"] == "evidence_clarification"
        assert second["routing"]["initial_route"] == "general_candidate"
        assert second["routing"]["final_route"] == "general_llm"


@pytest.mark.parametrize(
    "topic",
    [
        "我想了解量子计算",
        "换个话题说量子计算",
        "我们聊聊光合作用吧",
    ],
)
def test_pending_product_uses_raw_router_for_natural_general_topic_switch(
    tmp_path,
    topic: str,
) -> None:
    router_prompts: list[str] = []

    def topic_router_generate(
        kind: str,
        system: str,
        user: str,
        images: list[str],
    ) -> str:
        if "路由分类器" in system:
            router_prompts.append(user)
            if topic in user:
                return (
                    '{"route":"general_candidate","risk":"low",'
                    '"reason_code":"new-general-topic"}'
                )
            return (
                '{"route":"technical_candidate","risk":"medium",'
                '"reason_code":"product-support"}'
            )
        if "通用对话智能体" in system:
            return "已切换到新的通用话题。"
        return routed_generate(kind, system, user, images)

    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=topic_router_generate,
    )
    common = {"session_id": "natural-topic-switch", "user_id": "owner-a"}
    first = client.post(
        "/api/chat",
        json={"question": "设备一直报警怎么办", **common},
    ).json()
    second = client.post(
        "/api/chat",
        json={"question": topic, **common},
    ).json()
    detail = client.get(
        "/api/conversations/natural-topic-switch",
        params={"user_id": "owner-a"},
    ).json()

    assert first["routing"]["clarification"]["field"] == "product"
    assert second["routing"]["initial_route"] == "general_candidate"
    assert second["routing"]["final_route"] == "general_llm"
    assert detail["state"]["pending_clarification"] is None
    assert "product" not in detail["state"]["slots"]
    raw_prompt = next(prompt for prompt in router_prompts if topic in prompt)
    assert "会话相关上下文：无" in raw_prompt
    assert "设备一直报警怎么办" not in raw_prompt


@pytest.mark.parametrize(
    "product_reply",
    ["火星牌净化器", "ZX999 设备", "洗衣机"],
)
def test_pending_product_accepts_shape_limited_real_product_values(
    tmp_path,
    product_reply: str,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )
    common = {
        "session_id": f"product-value-{product_reply}",
        "user_id": "owner-a",
    }
    first = client.post(
        "/api/chat",
        json={"question": "设备一直报警怎么办", **common},
    ).json()
    second = client.post(
        "/api/chat",
        json={"question": product_reply, **common},
    ).json()
    detail = client.get(
        f"/api/conversations/{common['session_id']}",
        params={"user_id": "owner-a"},
    ).json()

    assert first["routing"]["clarification"]["field"] == "product"
    assert second["routing"]["clarification"]["field"] == "model"
    assert detail["state"]["slots"]["product"][-1]["value"]


def test_pending_product_model_and_error_code_accept_bare_values_one_at_a_time(
    tmp_path,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )
    common = {"session_id": "bare-slot-values", "user_id": "owner-a"}

    first = client.post(
        "/api/chat",
        json={"question": "设备一直报警怎么办", **common},
    ).json()
    second = client.post(
        "/api/chat",
        json={"question": "洗衣机", **common},
    ).json()
    third = client.post(
        "/api/chat",
        json={"question": "XQG100", **common},
    ).json()
    fourth = client.post(
        "/api/chat",
        json={"question": "E03", **common},
    ).json()

    assert first["routing"]["clarification"]["field"] == "product"
    assert first["routing"]["clarification"]["round"] == 1
    assert second["routing"]["clarification"]["field"] == "model"
    assert second["routing"]["clarification"]["round"] == 2
    assert third["routing"]["clarification"]["field"] == "error_code"
    assert third["routing"]["clarification"]["round"] == 3
    assert fourth["routing"]["final_route"] == "technical_knowledge"
    detail = client.get(
        "/api/conversations/bare-slot-values",
        params={"user_id": "owner-a"},
    ).json()
    assert detail["state"]["slots"]["model"][-1]["value"] == "XQG100"
    assert detail["state"]["slots"]["error_code"][-1]["value"] == "E03"


@pytest.mark.parametrize(
    ("display_text", "normalized_code"),
    [
        ("屏幕显示 ERR02", "ERR 02"),
        ("屏幕显示 ERR_02", "ERR 02"),
        ("屏幕显示 Err 02", "ERR 02"),
        ("屏幕显示 Err CF", "ERR CF"),
        ("屏幕显示 no CF", "NO CF"),
        ("屏幕显示 FuLL CF", "FULL CF"),
    ],
)
def test_spaced_camera_error_code_followup_never_switches_to_general(
    tmp_path,
    display_text: str,
    normalized_code: str,
) -> None:
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
            llm_generate_func=camera_error_generate,
        )
    )
    retriever = camera_error_retriever()
    client.app.state.orchestrator.knowledge.retriever = retriever
    common = {
        "session_id": f"camera-error-{normalized_code.replace(' ', '-').lower()}",
        "user_id": "owner-a",
    }

    first = client.post(
        "/api/chat",
        json={"question": "设备一直报错怎么办？", **common},
    ).json()
    second = client.post(
        "/api/chat",
        json={"question": "是相机", **common},
    ).json()
    third = client.post(
        "/api/chat",
        json={"question": display_text, **common},
    ).json()
    detail = client.get(
        f"/api/conversations/{common['session_id']}",
        params={"user_id": "owner-a"},
    ).json()

    assert first["routing"]["clarification"]["field"] == "product"
    assert second["routing"]["clarification"]["field"] == "model"
    assert third["routing"]["initial_route"] == "technical_candidate"
    assert third["routing"]["final_route"] == "technical_knowledge"
    assert third["citations"]
    assert "general" not in third["trace"]["selected_agents"]
    assert third["citations"][0]["parent_id"] == "camera-errors"
    assert detail["state"]["slots"]["error_code"][-1]["value"] == normalized_code


def test_topic_switch_is_decided_before_pending_symptom_can_pollute_slots(
    tmp_path,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )
    client.app.state.conversations.create(
        "owner-a",
        conversation_id="pre-slot-topic-switch",
    )
    client.app.state.conversation_memory.set_pending_clarification(
        "pre-slot-topic-switch",
        "owner-a",
        ClarificationRequest(
            case_id="case-symptom",
            field="symptom",
            question="请说明具体故障现象。",
            round=1,
        ),
        original_question="设备有什么故障",
    )

    body = client.post(
        "/api/chat",
        json={
            "question": "帮我改写会议通知",
            "session_id": "pre-slot-topic-switch",
            "user_id": "owner-a",
        },
    ).json()
    detail = client.get(
        "/api/conversations/pre-slot-topic-switch",
        params={"user_id": "owner-a"},
    ).json()

    assert body["routing"]["final_route"] == "general_llm"
    assert detail["state"]["pending_clarification"] is None
    assert "symptom" not in detail["state"]["slots"]


def test_champion_guarded_general_route_still_runs_legacy_guard(tmp_path) -> None:
    frozen = "冠军链路：保持冻结答案。"
    prompts: list[str] = []

    def generate(kind: str, system: str, user: str, images: list[str]) -> str:
        prompts.append(system)
        return routed_generate(kind, system, user, images)

    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="champion_guarded",
            legacy_answer_func=lambda question, images: frozen,
            llm_generate_func=generate,
        )
    )

    body = client.post(
        "/api/chat",
        json={"question": "把明天下午开会改写正式一点"},
    ).json()

    assert body["answer"] == frozen
    assert body["used_legacy"] is True
    assert "general" not in body["trace"]["selected_agents"]
    assert not any("通用对话智能体" in prompt for prompt in prompts)


def test_legacy_trace_records_dedicated_qwen_runtime(tmp_path) -> None:
    app = create_app(
        data_dir=tmp_path,
        rollout_mode="legacy_only",
        llm_generate_func=routed_generate,
    )
    client = TestClient(app)
    client.post(
        "/api/models",
        json={
            "provider": "DeepSeek",
            "name": "deepseek-v4-flash",
            "kind": "llm",
            "base_url": "https://api.deepseek.com",
            "api_key": "deepseek-secret",
        },
    )
    client.post(
        "/api/models",
        json={
            "provider": "Tongyi-Qianwen",
            "name": "qwen3-max",
            "kind": "llm",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "qwen-secret",
        },
    )

    class FakeEngine:
        use_llm = False
        use_llm_manual_polish = False
        use_llm_query_frame = False
        use_llm_query_rewrite = False
        use_ann = False
        llm = None
        customer_llm = None

        def answer(self, question, images):
            if self.use_llm and self.llm is not None:
                return self.llm.chat([{"role": "user", "content": question}])
            return "确定性冠军答案"

    class FakeClient:
        def chat(self, messages, temperature=None):
            return "Qwen冠军答案"

    adapter = app.state.orchestrator.legacy
    adapter._engine = FakeEngine()
    adapter._llm_client_factory = lambda runtime: FakeClient()

    body = client.post(
        "/api/chat",
        json={"question": "相机显示 ERR02 怎么办"},
    ).json()

    legacy_span = next(
        span for span in body["trace"]["spans"] if span["name"] == "legacy_champion"
    )
    assert body["used_legacy"] is True
    assert body["answer"] == "Qwen冠军答案"
    assert legacy_span["attributes"]["llm_used"] is True
    assert legacy_span["attributes"]["model_used"] == "qwen3-max"
    assert legacy_span["attributes"]["fallback_reason"] is None


def test_champion_guarded_clarification_and_safety_routes_do_not_bypass_legacy(
    tmp_path,
) -> None:
    calls: list[str] = []

    def legacy(question: str, images: list[str]) -> str:
        calls.append(question)
        return "冠军链路：请停止操作并联系专业人员。"

    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="champion_guarded",
            legacy_answer_func=legacy,
            llm_generate_func=routed_generate,
        )
    )

    clarification = client.post(
        "/api/chat",
        json={
            "question": "洗衣机一直报警怎么办",
            "session_id": "guard-clarify",
            "user_id": "owner-a",
        },
    ).json()
    safety = client.post(
        "/api/chat",
        json={"question": "火星牌 ZX999 冒烟了，怎么拆开自己维修"},
    ).json()

    assert clarification["used_legacy"] is True
    assert len(calls) == 2
    assert any(
        step["agent_id"] == "legacy-champion"
        for step in safety["trace"]["steps"]
    )
    assert safety["used_legacy"] is False
    assert "停止使用" in safety["answer"]


def test_dynamic_routing_shadow_records_router_without_changing_frozen_answer(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AKA_DYNAMIC_ROUTING", "shadow")
    frozen = "冻结冠军答案"
    prompts: list[str] = []

    def generate(kind: str, system: str, user: str, images: list[str]) -> str:
        prompts.append(system)
        return routed_generate(kind, system, user, images)

    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="legacy_only",
            legacy_answer_func=lambda question, images: frozen,
            llm_generate_func=generate,
        )
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "把明天下午开会改写正式一点"},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    body = events[-1]["payload"]["response"]
    assert body["answer"] == frozen
    assert any(event["type"] == "route.detected" for event in events)
    assert any("路由分类器" in prompt for prompt in prompts)
    assert "general" not in body["trace"]["selected_agents"]


def test_layered_memory_shadow_does_not_apply_pending_case_to_current_answer(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AKA_LAYERED_MEMORY", "shadow")
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )
    common = {"session_id": "layered-shadow", "user_id": "owner-a"}

    first = client.post(
        "/api/chat",
        json={"question": "洗衣机一直报警怎么办", **common},
    ).json()
    second = client.post(
        "/api/chat",
        json={"question": "把明天下午开会改写正式一点", **common},
    ).json()
    detail = client.get(
        "/api/conversations/layered-shadow",
        params={"user_id": "owner-a"},
    ).json()

    assert first["routing"]["final_route"] == "evidence_clarification"
    assert second["routing"]["final_route"] == "general_llm"
    assert second["answer"].startswith("兹定于")
    assert detail["state"] is not None
    assert detail["state"]["pending_clarification"] is None


def test_layered_memory_off_never_runs_or_reports_memory_curator(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AKA_LAYERED_MEMORY", "off")
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )
    submitted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        client.app.state.memory_curator,
        "submit",
        lambda conversation_id, owner_id: submitted.append(
            (conversation_id, owner_id)
        ),
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={
            "question": "把明天下午开会改写正式一点",
            "session_id": "layered-off",
            "user_id": "owner-a",
        },
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    body = events[-1]["payload"]["response"]
    detail = client.get(
        "/api/conversations/layered-off",
        params={"user_id": "owner-a"},
    ).json()

    assert submitted == []
    assert "memory-curator" not in body["trace"]["selected_agents"]
    assert all(
        span["name"] != "layered_memory"
        for span in body["trace"]["spans"]
    )
    assert all(event["agent_id"] != "memory-curator" for event in events)
    assert detail["state"] is not None
    assert detail["state"]["slots"] == {}
    assert detail["state"]["pending_clarification"] is None
    assert detail["state"]["rolling_summary"] == ""


def test_pending_retrieval_query_uses_original_slots_once_without_assistant_text(
    tmp_path,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )
    common = {"session_id": "clean-pending-query", "user_id": "owner-a"}
    first = client.post(
        "/api/chat",
        json={"question": "洗衣机一直报警怎么办", **common},
    ).json()
    second = client.post(
        "/api/chat",
        json={"question": "型号是 XQG100", **common},
    ).json()
    rewrite = next(
        span for span in second["trace"]["spans"] if span["name"] == "query_rewrite"
    )["output_summary"]

    assert first["routing"]["final_route"] == "evidence_clarification"
    assert "洗衣机一直报警怎么办" in rewrite
    assert rewrite.count("型号是 XQG100") == 1
    assert "为了匹配正确的产品手册" not in rewrite


def test_followup_retrieval_uses_structured_slots_not_previous_assistant_body(
    tmp_path,
) -> None:
    def generate(kind: str, system: str, user: str, images: list[str]) -> str:
        if "路由分类器" in system:
            return (
                '{"route":"technical_candidate","risk":"medium",'
                '"reason_code":"followup"}'
            )
        if "检索查询改写器" in system:
            return "中文：继续处理 | English: continue troubleshooting"
        if "Memory Curator" in system:
            return "用户正在处理洗衣机故障。"
        return "洗衣机出现 E03 时，请先断电，检查排水管并清理排水过滤器。"

    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=generate,
    )
    common = {"session_id": "clean-followup-query", "user_id": "owner-a"}
    first = client.post(
        "/api/chat",
        json={"question": "洗衣机显示 E03 怎么处理", **common},
    ).json()
    second = client.post(
        "/api/chat",
        json={"question": "还需要做什么", **common},
    ).json()
    rewrite = next(
        span for span in second["trace"]["spans"] if span["name"] == "query_rewrite"
    )["output_summary"]

    assert first["answer"]
    assert "还需要做什么" in rewrite
    assert "Washing Machine" in rewrite
    assert "E03" in rewrite
    assert "检查排水管并清理排水过滤器" not in rewrite


def test_realtime_order_question_collects_order_and_service_request_one_at_a_time(
    tmp_path,
) -> None:
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
            llm_generate_func=routed_generate,
        )
    )
    common = {"session_id": "order-clarify", "user_id": "owner-a"}

    first = client.post(
        "/api/chat",
        json={"question": "我的订单有问题，应该怎么办", **common},
    ).json()
    second = client.post(
        "/api/chat",
        json={"question": "订单号是 ORD-20260725-001", **common},
    ).json()
    third = client.post(
        "/api/chat",
        json={"question": "我希望申请换货", **common},
    ).json()

    assert first["routing"]["final_route"] == "evidence_clarification"
    assert first["routing"]["clarification"]["field"] == "order_identifier"
    assert second["routing"]["final_route"] == "evidence_clarification"
    assert second["routing"]["clarification"]["field"] == "service_request"
    assert third["routing"]["final_route"] == "customer_service"
    assert third["citations"][0]["source_type"] == "policy"


def test_general_after_sales_policy_question_does_not_require_order_identifier(
    tmp_path,
) -> None:
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
            llm_generate_func=routed_generate,
        )
    )

    body = client.post(
        "/api/chat",
        json={
            "question": "发票丢了还能申请售后吗",
            "session_id": "policy-question",
            "user_id": "owner-a",
        },
    ).json()

    assert body["routing"]["final_route"] == "customer_service"
    assert body["routing"]["coverage_status"] == "covered"
    assert body["citations"][0]["source_type"] == "policy"


def test_router_model_fallback_emits_auditable_event_and_trace_span(tmp_path) -> None:
    def invalid_router(kind: str, system: str, user: str, images: list[str]) -> str:
        if "路由分类器" in system:
            return "不是合法 JSON"
        return ""

    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
            llm_generate_func=invalid_router,
        )
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "今天心情怎么样", "session_id": "router-fallback"},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    types = [event["type"] for event in events]
    body = events[-1]["payload"]["response"]
    fallback = next(event for event in events if event["type"] == "router.fallback")
    span = next(
        item for item in body["trace"]["spans"] if item["name"] == "router_fallback"
    )

    assert types.index("route.detected") < types.index("router.fallback")
    assert fallback["payload"]["reason_code"] == "deterministic-general"
    assert fallback["payload"]["model_used"] is None
    assert span["status"] == "completed"
    assert span["attributes"]["reason_code"] == "deterministic-general"


def test_covered_technical_answer_uses_primary_evidence_instead_of_repeating_models(
    tmp_path,
) -> None:
    legacy_calls: list[str] = []

    def unsupported_generation(
        kind: str,
        system: str,
        user: str,
        images: list[str],
    ) -> str:
        return "洗衣机 E03 出现后，请接入 999 V 电源继续运行。"

    def legacy_answer(question: str, images: list[str]) -> str:
        legacy_calls.append(question)
        return "不应调用的 Legacy 答案"

    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        legacy_answer_func=legacy_answer,
        llm_generate_func=unsupported_generation,
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "洗衣机 E03 怎么处理"},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    body = events[-1]["payload"]["response"]
    event_types = [event["type"] for event in events]

    assert body["verification"]["passed"] is True
    assert body["used_legacy"] is False
    assert legacy_calls == []
    assert "999 V" not in body["answer"]
    assert "检查排水管并清理排水过滤器" in body["answer"]
    assert "answer.evidence_fallback" in event_types
    assert "answer.revision.started" not in event_types


def test_historical_citation_refs_are_not_reused_as_current_general_evidence(
    tmp_path,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=routed_generate,
    )
    common = {"session_id": "historical-refs", "user_id": "owner-a"}

    technical = client.post(
        "/api/chat",
        json={"question": "洗衣机 E03 怎么处理", **common},
    ).json()
    general = client.post(
        "/api/chat",
        json={"question": "把明天下午开会改写正式一点", **common},
    ).json()

    assert technical["citations"]
    assert general["routing"]["final_route"] == "general_llm"
    assert general["citations"] == []
    assert "knowledge" not in general["trace"]["selected_agents"]
