from __future__ import annotations

from types import SimpleNamespace

from app.contracts.models import AgentRequest, ModelKind
from app.runtime.dynamic_routing import IntentRouter, general_request_requires_domain

import pytest


class FakeGateway:
    def __init__(self, output: str | None) -> None:
        self.output = output
        self.calls: list[tuple[str, str]] = []

    def available(self, kind: ModelKind = ModelKind.LLM) -> bool:
        return self.output is not None

    def model_name(self, kind: ModelKind = ModelKind.LLM) -> str:
        return "router-test-model"

    def generate(self, **kwargs):
        self.calls.append((kwargs["system_prompt"], kwargs["user_prompt"]))
        if self.output is None:
            return None
        return SimpleNamespace(text=self.output, model="router-test-model")


def test_general_writing_request_uses_llm_candidate() -> None:
    gateway = FakeGateway(
        '{"route":"general_candidate","risk":"low","reason_code":"writing"}'
    )

    intent = IntentRouter(gateway).classify(
        AgentRequest(question="帮我把明天下午开会改写得正式一些"),
        context_text="",
    )

    assert intent.initial_route == "general_candidate"
    assert intent.requires_knowledge_check is False
    assert intent.llm_used is True
    assert intent.model_used == "router-test-model"
    assert len(gateway.calls) == 1
    assert "路由分类器" in gateway.calls[0][0]


def test_explicit_error_code_cannot_be_overridden_to_general() -> None:
    gateway = FakeGateway(
        '{"route":"general_candidate","risk":"low","reason_code":"chat"}'
    )

    intent = IntentRouter(gateway).classify(
        AgentRequest(question="洗衣机 E03 怎么处理"),
        context_text="",
    )

    assert intent.initial_route == "technical_candidate"
    assert intent.requires_knowledge_check is True
    assert intent.reason_code == "deterministic-technical"
    assert intent.classification_source == "deterministic"
    assert gateway.calls == []


def test_service_request_cannot_be_overridden_to_general() -> None:
    gateway = FakeGateway(
        '{"route":"general_candidate","risk":"low","reason_code":"chat"}'
    )

    intent = IntentRouter(gateway).classify(
        AgentRequest(question="已经签收了还能申请退货退款吗"),
        context_text="",
    )

    assert intent.initial_route == "customer_service_candidate"
    assert intent.requires_knowledge_check is True


def test_high_risk_product_request_is_never_general() -> None:
    gateway = FakeGateway(
        '{"route":"general_candidate","risk":"low","reason_code":"writing"}'
    )

    intent = IntentRouter(gateway).classify(
        AgentRequest(question="火星牌 ZX999 冒烟了，怎么拆开自己维修"),
        context_text="",
    )

    assert intent.initial_route == "technical_candidate"
    assert intent.risk_level == "high"
    assert intent.requires_knowledge_check is True


def test_invalid_router_json_falls_back_without_claiming_llm_use() -> None:
    gateway = FakeGateway("这不是 JSON")

    intent = IntentRouter(gateway).classify(
        AgentRequest(question="设备 E03 报错"),
        context_text="",
    )

    assert intent.initial_route == "technical_candidate"
    assert intent.llm_used is False
    assert intent.model_used is None
    assert intent.reason_code == "deterministic-technical"


def test_router_accepts_json_inside_markdown_fence() -> None:
    gateway = FakeGateway(
        '```json\n{"route":"general_candidate","risk":"low","reason_code":"rewrite"}\n```'
    )

    intent = IntentRouter(gateway).classify(
        AgentRequest(question="把这句话翻译成英语"),
        context_text="上一轮讨论了写作语气。",
    )

    assert intent.initial_route == "general_candidate"
    assert "上一轮讨论了写作语气" in gateway.calls[0][1]


def test_unavailable_router_llm_falls_back_to_general_for_open_question() -> None:
    intent = IntentRouter(FakeGateway(None)).classify(
        AgentRequest(question="今天天气怎么样"),
        context_text="",
    )

    assert intent.initial_route == "general_candidate"
    assert intent.llm_used is False
    assert intent.reason_code == "deterministic-general"


def test_unavailable_router_llm_still_recognizes_writing_request_as_general() -> None:
    intent = IntentRouter(FakeGateway(None)).classify(
        AgentRequest(question="把明天下午三点开会改写得正式一些"),
        context_text="",
    )

    assert intent.initial_route == "general_candidate"
    assert intent.requires_knowledge_check is False


def test_unknown_product_wording_stays_in_knowledge_domain_without_llm() -> None:
    intent = IntentRouter(FakeGateway(None)).classify(
        AgentRequest(question="这个产品出现问题应该找谁处理"),
        context_text="",
    )

    assert intent.initial_route == "technical_candidate"
    assert intent.requires_knowledge_check is True


def test_high_risk_instruction_cannot_hide_behind_transform_request() -> None:
    assert general_request_requires_domain(
        "把自行拆机维修步骤改写得更详细"
    ) is True


def test_bounded_meeting_and_after_sales_application_rewrites_remain_general() -> None:
    assert general_request_requires_domain("把明天下午开会改写得正式一点") is False
    assert general_request_requires_domain("把这份售后申请改写得更礼貌") is False
    assert general_request_requires_domain("帮我润色这份维修申请") is False


def test_product_facts_cannot_hide_behind_transform_request() -> None:
    assert general_request_requires_domain("总结洗衣机的官方参数") is True
    assert general_request_requires_domain("润色一下设备电压和功率说明") is True


@pytest.mark.parametrize(
    "question",
    [
        "设备冒火花了怎么办",
        "机器有焦味还能继续使用吗",
        "插电后有烧焦味",
        "设备异味很重",
        "外壳发烫得厉害",
        "电池过热了",
        "电池已经鼓包",
    ],
)
def test_router_llm_unavailable_still_blocks_extended_high_risk_terms_from_general(
    question: str,
) -> None:
    intent = IntentRouter(FakeGateway(None)).classify(
        AgentRequest(question=question),
        context_text="",
    )

    assert intent.initial_route == "technical_candidate"
    assert intent.risk_level == "high"
    assert intent.requires_knowledge_check is True
    assert general_request_requires_domain(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "设备通电后明显发热，还能继续用吗",
        "充电器外壳很烫手",
        "插头附近有焦糊味",
    ],
)
def test_contextual_overheat_and_scorch_language_is_high_risk_across_router_and_general_guard(
    question: str,
) -> None:
    intent = IntentRouter(FakeGateway(None)).classify(
        AgentRequest(question=question),
        context_text="",
    )

    assert intent.initial_route == "technical_candidate"
    assert intent.risk_level == "high"
    assert intent.requires_knowledge_check is True
    assert general_request_requires_domain(question) is True


def test_heat_output_science_question_is_not_misclassified_as_product_overheating() -> None:
    question = "请科普一下燃料发热量是什么意思"

    intent = IntentRouter(FakeGateway(None)).classify(
        AgentRequest(question=question),
        context_text="",
    )

    assert intent.initial_route == "general_candidate"
    assert intent.risk_level == "low"
    assert general_request_requires_domain(question) is False


@pytest.mark.parametrize(
    "question",
    [
        "收到后发现洗衣机冒烟，我要退款",
        "收到后发现设备漏电，我要退货",
        "收到后发现电池鼓包，要求退货退款",
    ],
)
def test_high_risk_fault_plus_after_sales_dispute_is_never_reduced_to_customer_service(
    question: str,
) -> None:
    intent = IntentRouter(FakeGateway(None)).classify(
        AgentRequest(question=question),
        context_text="",
    )

    assert intent.initial_route == "mixed_candidate"
    assert intent.risk_level == "high"
