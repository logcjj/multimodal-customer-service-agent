from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from app.contracts.models import (
    AgentRequest,
    AgentResult,
    ClarificationRequest,
    Claim,
    CoverageStatus,
    Evidence,
    ModelConfigurationCreate,
    ModelKind,
    RoutingDecision,
)


def test_model_configuration_never_serializes_plaintext_secret() -> None:
    model = ModelConfigurationCreate(
        provider="OpenAI-API-Compatible",
        name="qwen3-max",
        kind=ModelKind.LLM,
        base_url="https://example.com/v1",
        api_key="super-secret-value",
    )

    payload = model.model_dump(mode="json")

    assert "api_key" not in payload
    assert "super-secret-value" not in str(payload)


def test_agent_result_requires_evidence_for_factual_claim() -> None:
    with pytest.raises(ValidationError):
        AgentResult(
            task_id="technical-1",
            agent_id="knowledge",
            status="completed",
            answer_fragment="滤网需要每两周清洁一次。",
            claims=[Claim(text="滤网需要每两周清洁一次。", evidence_ids=[])],
            confidence=0.9,
        )


def test_agent_result_accepts_structured_evidence() -> None:
    evidence = Evidence(
        evidence_id="ev-1",
        source_type="manual",
        title="清洁滤网",
        text="建议每两周清洁一次滤网。",
        product="air-conditioner",
    )

    result = AgentResult(
        task_id="technical-1",
        agent_id="knowledge",
        status="completed",
        answer_fragment="建议每两周清洁一次滤网。",
        claims=[Claim(text="建议每两周清洁一次滤网。", evidence_ids=["ev-1"])],
        evidence=[evidence],
        confidence=0.92,
    )

    assert result.claims[0].evidence_ids == ["ev-1"]


def test_agent_request_rejects_non_positive_deadline() -> None:
    with pytest.raises(ValidationError):
        AgentRequest(question="怎么清洁滤网？", deadline_ms=0)


def test_routing_decision_exposes_auditable_reason_without_hidden_reasoning() -> None:
    routing = RoutingDecision(
        initial_route="technical_candidate",
        final_route="evidence_clarification",
        route_label="证据补全",
        route_reason="缺少产品型号",
        coverage_status=CoverageStatus.CLARIFIABLE,
        knowledge_covered=False,
        risk_level="medium",
        clarification=ClarificationRequest(
            case_id="case-1",
            field="model",
            question="请提供产品型号。",
            round=1,
            max_rounds=3,
            accepted_input_types=["text", "image"],
        ),
    )

    payload = routing.model_dump(mode="json")

    assert payload["route_reason"] == "缺少产品型号"
    assert payload["clarification"]["field"] == "model"
    assert "reasoning" not in payload
    assert "chain_of_thought" not in payload


def test_clarification_round_cannot_exceed_three() -> None:
    with pytest.raises(ValidationError):
        ClarificationRequest(
            case_id="case-1",
            field="model",
            question="请提供产品型号。",
            round=4,
            max_rounds=3,
        )


def test_final_route_contract_remains_the_seven_frontend_supported_routes() -> None:
    assert set(get_args(RoutingDecision.model_fields["final_route"].annotation)) == {
        "technical_knowledge",
        "customer_service",
        "mixed",
        "evidence_clarification",
        "general_llm",
        "safe_handoff",
        "general_unavailable",
    }
