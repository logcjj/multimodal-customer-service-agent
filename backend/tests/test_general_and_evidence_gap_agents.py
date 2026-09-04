from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from app.agents.evidence_gap import EvidenceGapAgent
from app.agents.general import GeneralAgent
from app.agents.verifier import VerifierAgent
from app.contracts.models import AgentRequest, ModelKind


class FakeGateway:
    def __init__(self, output: str | None) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []

    def available(self, kind: ModelKind = ModelKind.LLM) -> bool:
        return self.output is not None

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.output is None:
            return None
        return SimpleNamespace(
            text=self.output,
            model="injected-test-model",
            latency_ms=12,
        )


def test_general_agent_calls_real_llm_and_has_no_fake_citations() -> None:
    gateway = FakeGateway("兹定于明天下午召开会议，请准时参加。")

    result = GeneralAgent(gateway).run(
        AgentRequest(question="改写会议通知"),
        context_text="用户偏好正式语气。",
    )

    assert result.status == "completed"
    assert result.llm_generated is True
    assert result.model_used == "injected-test-model"
    assert result.evidence == []
    assert result.claims == []
    assert len(gateway.calls) == 1
    assert "用户偏好正式语气" in str(gateway.calls[0]["user_prompt"])


def test_general_agent_injects_current_china_time_into_prompt() -> None:
    gateway = FakeGateway("今天是 2026 年 7 月 27 日，星期一。")
    fixed_time = datetime(
        2026,
        7,
        27,
        20,
        30,
        tzinfo=ZoneInfo("Asia/Shanghai"),
    )

    GeneralAgent(gateway, now=lambda: fixed_time).run(
        AgentRequest(question="今天几号"),
        context_text="",
    )

    assert (
        "当前服务器时间（中国标准时间）：2026年07月27日 星期一 20:30"
        in str(gateway.calls[0]["user_prompt"])
    )


def test_general_agent_reports_failure_when_llm_is_unavailable() -> None:
    result = GeneralAgent(FakeGateway(None)).run(
        AgentRequest(question="改写会议通知"),
        context_text="",
    )

    assert result.status == "failed"
    assert result.llm_generated is False
    assert result.answer_fragment == ""
    assert result.recommended_next_action == "general-model-unavailable"


def test_evidence_gap_asks_exactly_one_highest_priority_field() -> None:
    result, clarification = EvidenceGapAgent(max_rounds=3).run(
        missing_fields=["model", "error_code"],
        round_number=1,
        case_id="case-1",
    )

    assert result.status == "needs_input"
    assert clarification.field == "model"
    assert clarification.round == 1
    assert "型号" in clarification.question
    assert "错误码" not in clarification.question
    assert result.answer_fragment == clarification.question


def test_evidence_gap_accepts_image_for_error_code() -> None:
    _, clarification = EvidenceGapAgent(max_rounds=3).run(
        missing_fields=["error_code"],
        round_number=2,
        case_id="case-1",
    )

    assert clarification.accepted_input_types == ["text", "image"]
    assert clarification.round == 2


def test_general_verifier_accepts_bounded_writing_answer() -> None:
    result = GeneralAgent(FakeGateway("兹定于明天下午召开会议。")).run(
        AgentRequest(question="改写会议通知"),
        context_text="",
    )

    report = VerifierAgent().verify_general(
        AgentRequest(question="改写会议通知"),
        result,
    )

    assert report.passed is True
    assert report.action == "accept"


def test_general_verifier_rejects_product_repair_or_official_commitment() -> None:
    result = GeneralAgent(
        FakeGateway("你可以自行拆机维修，并保证获得全额退款。")
    ).run(
        AgentRequest(question="随便回答"),
        context_text="",
    )

    report = VerifierAgent().verify_general(
        AgentRequest(question="随便回答"),
        result,
    )

    assert report.passed is False
    assert report.action == "handoff"
    assert {issue.code for issue in report.issues} == {
        "general-route-safety-boundary"
    }


def test_general_verifier_rejects_dangerous_original_request_even_when_answer_avoids_denylist() -> None:
    result = GeneralAgent(
        FakeGateway("先关闭总闸，取下旧保险丝并换上新的，然后重新送电测试。")
    ).run(
        AgentRequest(question="怎样更换保险丝"),
        context_text="",
    )

    report = VerifierAgent().verify_general(
        AgentRequest(question="怎样更换保险丝"),
        result,
    )

    assert report.passed is False
    assert report.action == "handoff"
    assert {issue.code for issue in report.issues} == {
        "general-request-out-of-domain"
    }


def test_general_verifier_reuses_transform_boundary_for_product_facts_and_service_copy() -> None:
    product_result = GeneralAgent(FakeGateway("官方参数摘要。")).run(
        AgentRequest(question="总结洗衣机的官方参数"),
        context_text="",
    )
    service_copy_result = GeneralAgent(FakeGateway("请协助安排售后维修。")).run(
        AgentRequest(question="帮我润色这份维修申请"),
        context_text="",
    )

    product_report = VerifierAgent().verify_general(
        AgentRequest(question="总结洗衣机的官方参数"),
        product_result,
    )
    service_copy_report = VerifierAgent().verify_general(
        AgentRequest(question="帮我润色这份维修申请"),
        service_copy_result,
    )

    assert product_report.passed is False
    assert {issue.code for issue in product_report.issues} == {
        "general-request-out-of-domain"
    }
    assert service_copy_report.passed is True


def test_general_verifier_rejects_contextual_product_overheating_request() -> None:
    result = GeneralAgent(FakeGateway("建议继续观察。")).run(
        AgentRequest(question="插头发热而且摸起来烫手，还能继续通电吗"),
        context_text="",
    )

    report = VerifierAgent().verify_general(
        AgentRequest(question="插头发热而且摸起来烫手，还能继续通电吗"),
        result,
    )

    assert report.passed is False
    assert {issue.code for issue in report.issues} == {
        "general-request-out-of-domain"
    }
