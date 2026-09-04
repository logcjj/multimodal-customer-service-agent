from __future__ import annotations

from time import monotonic

import pytest

from app.contracts.models import AgentRequest
from app.runtime.planner import Planner
from app.runtime.state import RuntimeState


def test_technical_question_selects_knowledge_and_verifier() -> None:
    plan = Planner().create_plan(AgentRequest(question="空气净化器的滤网怎么清洁？"))

    assert plan.route == "technical"
    assert plan.selected_agents == ["orchestrator", "knowledge", "verifier"]


def test_service_question_selects_customer_service_and_verifier() -> None:
    plan = Planner().create_plan(AgentRequest(question="已经签收了，现在还能申请退款吗？"))

    assert plan.route == "customer_service"
    assert "customer-service" in plan.selected_agents
    assert "knowledge" not in plan.selected_agents


def test_free_repair_or_compensation_promise_routes_to_customer_service() -> None:
    plan = Planner().create_plan(
        AgentRequest(question="你们能保证免费维修并在24小时内赔付吗？")
    )

    assert plan.route == "customer_service"
    assert "customer-service" in plan.selected_agents
    assert "knowledge" not in plan.selected_agents


def test_after_sales_question_with_generic_how_or_use_word_stays_in_customer_service() -> None:
    plan = Planner().create_plan(
        AgentRequest(
            question=(
                "商品实际不支持页面宣传的功能，无法使用，我想退货退款并赔偿，"
                "应该怎么处理？"
            )
        )
    )

    assert plan.route == "customer_service"
    assert "customer-service" in plan.selected_agents
    assert "knowledge" not in plan.selected_agents


def test_description_mismatch_with_feature_name_is_a_service_dispute_not_technical_support() -> None:
    plan = Planner().create_plan(
        AgentRequest(
            question=(
                "收到商品后发现详情页说支持无线充电，实际不支持，"
                "我要求退货退款并赔偿。"
            )
        )
    )

    assert plan.route == "customer_service"
    assert "knowledge" not in plan.selected_agents


def test_order_address_question_routes_to_customer_service() -> None:
    plan = Planner().create_plan(
        AgentRequest(question="下单后能修改收货地址吗？超过时间还能改吗？")
    )

    assert plan.route == "customer_service"
    assert "knowledge" not in plan.selected_agents


def test_explicit_device_fault_plus_warranty_question_still_routes_mixed() -> None:
    plan = Planner().create_plan(
        AgentRequest(question="洗衣机显示 E03 排水故障，还在保修期，应该怎么处理？")
    )

    assert plan.route == "mixed"
    assert "knowledge" in plan.selected_agents
    assert "customer-service" in plan.selected_agents


def test_mixed_question_runs_knowledge_and_service_specialists() -> None:
    plan = Planner().create_plan(AgentRequest(question="设备出现 E03 报错，修不好能退货吗？"))

    assert plan.route == "mixed"
    assert plan.selected_agents == ["orchestrator", "knowledge", "customer-service", "verifier"]
    assert len(plan.subtasks) == 3


@pytest.mark.parametrize(
    "question",
    [
        "收到后发现洗衣机冒烟，我要退款",
        "收到后发现设备漏电，我要退货",
        "收到后发现电池鼓包，要求退货退款",
    ],
)
def test_planner_keeps_high_risk_after_sales_disputes_mixed(question: str) -> None:
    plan = Planner().create_plan(AgentRequest(question=question))

    assert plan.route == "mixed"
    assert "knowledge" in plan.selected_agents
    assert "customer-service" in plan.selected_agents


def test_image_request_adds_multimodal_before_domain_agents() -> None:
    plan = Planner().create_plan(
        AgentRequest(
            question="看一下图片里的报错，应该怎么处理？",
            images=["data:image/png;base64,AAAA"],
            deadline_ms=30_000,
        )
    )

    assert plan.selected_agents == ["orchestrator", "multimodal", "knowledge", "verifier"]
    knowledge_task = next(item for item in plan.subtasks if item.assigned_agent == "knowledge")
    assert knowledge_task.depends_on == ["vision-1"]


def test_runtime_state_reports_expired_deadline() -> None:
    state = RuntimeState(
        request=AgentRequest(question="测试", deadline_ms=1),
        started_at=monotonic() - 1,
    )

    assert state.remaining_ms == 0
    assert state.is_expired is True
