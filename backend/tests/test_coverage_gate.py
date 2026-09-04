from __future__ import annotations

import pytest

from app.contracts.models import (
    AgentResult,
    CoverageStatus,
    Evidence,
    RoutingIntent,
)
from app.runtime.dynamic_routing import KnowledgeCoverageGate


def intent(route: str, risk: str = "medium") -> RoutingIntent:
    return RoutingIntent(
        initial_route=route,
        risk_level=risk,
        requires_knowledge_check=route != "general_candidate",
        reason_code="test",
    )


def result(
    agent_id: str,
    *,
    evidence: list[Evidence] | None = None,
    status: str = "completed",
) -> AgentResult:
    return AgentResult(
        task_id=f"{agent_id}-1",
        agent_id=agent_id,
        status=status,
        evidence=evidence or [],
        confidence=0.9 if evidence else 0.2,
    )


def test_manual_evidence_covering_error_code_is_covered() -> None:
    assessment = KnowledgeCoverageGate(max_rounds=3).evaluate(
        intent=intent("technical_candidate"),
        question="洗衣机 E03 怎么处理",
        results=[
            result(
                "knowledge",
                evidence=[
                    Evidence(
                        evidence_id="manual-1",
                        source_type="manual",
                        title="洗衣机 E03 排水故障",
                        text="E03 表示排水异常，请检查排水管。",
                    )
                ],
            )
        ],
        active_slots={"product": "Washing Machine", "error_code": "E03"},
        clarification_round=0,
    )

    assert assessment.status == CoverageStatus.COVERED
    assert assessment.final_route == "technical_knowledge"
    assert assessment.knowledge_covered is True
    assert assessment.evidence_count == 1


def test_generic_alarm_without_product_is_clarifiable_one_field_at_a_time() -> None:
    assessment = KnowledgeCoverageGate(max_rounds=3).evaluate(
        intent=intent("technical_candidate"),
        question="设备一直报警怎么办",
        results=[result("knowledge", status="needs_input")],
        active_slots={},
        clarification_round=0,
    )

    assert assessment.status == CoverageStatus.CLARIFIABLE
    assert assessment.final_route == "evidence_clarification"
    assert assessment.missing_fields[0] == "product"


def test_known_product_alarm_requests_model_before_error_code() -> None:
    assessment = KnowledgeCoverageGate(max_rounds=3).evaluate(
        intent=intent("technical_candidate"),
        question="洗衣机一直报警怎么办",
        results=[result("knowledge", status="needs_input")],
        active_slots={"product": "Washing Machine"},
        clarification_round=0,
    )

    assert assessment.status == CoverageStatus.CLARIFIABLE
    assert assessment.missing_fields[:2] == ["model", "error_code"]


def test_known_product_and_supported_spaced_error_code_do_not_require_model() -> None:
    assessment = KnowledgeCoverageGate(max_rounds=3).evaluate(
        intent=intent("technical_candidate"),
        question="屏幕显示 Err 02",
        results=[
            result(
                "knowledge",
                evidence=[
                    Evidence(
                        evidence_id="camera-err-02",
                        source_type="manual",
                        title="Camera Error Code Countermeasures",
                        text="Err 02 means there is a problem with the CF card.",
                        product="Camera",
                    )
                ],
            )
        ],
        active_slots={"product": "Camera", "error_code": "ERR 02"},
        clarification_round=2,
    )

    assert assessment.status == CoverageStatus.COVERED
    assert assessment.final_route == "technical_knowledge"
    assert assessment.missing_fields == []


@pytest.mark.parametrize("evidence_code", ["ERR02", "ERR_02", "Err 02"])
def test_camera_error_code_separator_variants_are_exactly_equivalent(
    evidence_code: str,
) -> None:
    assessment = KnowledgeCoverageGate(max_rounds=3).evaluate(
        intent=intent("technical_candidate"),
        question="屏幕显示 Err 02",
        results=[
            result(
                "knowledge",
                evidence=[
                    Evidence(
                        evidence_id=f"camera-{evidence_code}",
                        source_type="manual",
                        title="Camera Error Code Countermeasures",
                        text=f"{evidence_code} means the CF card has a problem.",
                        product="Camera",
                    )
                ],
            )
        ],
        active_slots={"product": "Camera", "error_code": "ERR_02"},
        clarification_round=2,
    )

    assert assessment.status == CoverageStatus.COVERED
    assert assessment.final_route == "technical_knowledge"


def test_camera_err_02_is_not_equivalent_to_err_020() -> None:
    assessment = KnowledgeCoverageGate(max_rounds=3).evaluate(
        intent=intent("technical_candidate"),
        question="屏幕显示 Err 02",
        results=[
            result(
                "knowledge",
                evidence=[
                    Evidence(
                        evidence_id="camera-err-020",
                        source_type="manual",
                        title="Camera Error Code Countermeasures",
                        text="Err 020 is a different diagnostic code.",
                        product="Camera",
                    )
                ],
            )
        ],
        active_slots={"product": "Camera", "error_code": "ERR02"},
        clarification_round=2,
    )

    assert assessment.status == CoverageStatus.CLARIFIABLE
    assert assessment.final_route == "evidence_clarification"
    assert assessment.missing_fields[0] == "model"


def test_unsupported_spaced_error_code_continues_precise_model_clarification() -> None:
    assessment = KnowledgeCoverageGate(max_rounds=3).evaluate(
        intent=intent("technical_candidate"),
        question="屏幕显示 Err 77",
        results=[
            result(
                "knowledge",
                evidence=[
                    Evidence(
                        evidence_id="camera-other-errors",
                        source_type="manual",
                        title="Camera Error Code Countermeasures",
                        text="Err 02 means there is a problem with the CF card.",
                        product="Camera",
                    )
                ],
            )
        ],
        active_slots={"product": "Camera", "error_code": "ERR 77"},
        clarification_round=2,
    )

    assert assessment.status == CoverageStatus.CLARIFIABLE
    assert assessment.final_route == "evidence_clarification"
    assert assessment.missing_fields[0] == "model"


def test_high_risk_unknown_repair_is_never_sent_to_general() -> None:
    assessment = KnowledgeCoverageGate(max_rounds=3).evaluate(
        intent=intent("technical_candidate", risk="high"),
        question="火星牌 ZX999 冒烟了，怎么拆开维修",
        results=[result("knowledge", status="needs_input")],
        active_slots={"model": "ZX999"},
        clarification_round=0,
    )

    assert assessment.status == CoverageStatus.UNSAFE_UNCOVERED
    assert assessment.final_route == "safe_handoff"
    assert assessment.missing_fields == []


def test_general_candidate_is_allowed_without_fake_knowledge_evidence() -> None:
    assessment = KnowledgeCoverageGate(max_rounds=3).evaluate(
        intent=intent("general_candidate", risk="low"),
        question="把明天下午开会改写正式一点",
        results=[],
        active_slots={},
        clarification_round=0,
    )

    assert assessment.status == CoverageStatus.GENERAL_ALLOWED
    assert assessment.final_route == "general_llm"
    assert assessment.knowledge_covered is False


def test_general_candidate_with_contextual_overheat_language_is_safely_handed_off() -> None:
    assessment = KnowledgeCoverageGate(max_rounds=3).evaluate(
        intent=intent("general_candidate", risk="low"),
        question="设备外壳发热而且烫手，还能继续用吗",
        results=[],
        active_slots={},
        clarification_round=0,
    )

    assert assessment.status == CoverageStatus.UNSAFE_UNCOVERED
    assert assessment.final_route == "safe_handoff"
    assert assessment.knowledge_covered is False


def test_third_failed_clarification_round_hands_off() -> None:
    assessment = KnowledgeCoverageGate(max_rounds=3).evaluate(
        intent=intent("technical_candidate"),
        question="还是不能用",
        results=[result("knowledge", status="needs_input")],
        active_slots={},
        clarification_round=3,
    )

    assert assessment.status == CoverageStatus.UNSAFE_UNCOVERED
    assert assessment.final_route == "safe_handoff"
