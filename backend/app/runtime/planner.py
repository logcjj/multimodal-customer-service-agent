from __future__ import annotations

import re
from typing import Literal

from app.contracts.models import AgentRequest, SubTask, TaskPlan


SERVICE_TERMS = (
    "退款",
    "退货",
    "换货",
    "运费",
    "物流",
    "发票",
    "订单",
    "投诉",
    "售后",
    "保修",
    "维修费用",
    "免费维修",
    "免费更换",
    "赔付",
    "赔偿",
    "签收",
    "下单",
    "支付",
    "付款方式",
    "收货地址",
    "refund",
    "return",
    "shipping",
    "invoice",
    "warranty",
)

TECHNICAL_TERMS = (
    "怎么",
    "如何",
    "报错",
    "故障",
    "异常",
    "清洁",
    "安装",
    "拆卸",
    "使用",
    "连接",
    "设置",
    "充电",
    "无法",
    "不能",
    "错误码",
    "滤网",
    "电池",
    "按钮",
    "manual",
    "error",
    "install",
    "clean",
    "connect",
    "setup",
)

MIXED_TECHNICAL_TERMS = (
    "报错",
    "故障",
    "异常",
    "错误码",
    "安装",
    "拆卸",
    "清洁",
    "连接",
    "设置",
    "充电",
    "滤网",
    "电池",
    "按钮",
    "error",
    "install",
    "clean",
    "connect",
    "setup",
)

SERVICE_DISPUTE_TERMS = (
    "详情页",
    "描述不符",
    "实际不支持",
    "质量问题",
    "型号发错",
    "收到后发现",
    "要求退货退款",
)

HIGH_RISK_TECHNICAL_TERMS = (
    "冒烟",
    "火花",
    "起火",
    "漏电",
    "触电",
    "焦味",
    "焦糊味",
    "烧焦",
    "异味",
    "爆炸",
    "发烫",
    "烫手",
    "过热",
    "鼓包",
    "短路",
)

_CONTEXTUAL_HEAT_PATTERN = re.compile(r"发热(?!量)")
_PRODUCT_HEAT_CONTEXT_TERMS = (
    "产品",
    "设备",
    "机器",
    "电器",
    "家电",
    "插头",
    "插座",
    "充电器",
    "电源",
    "电源线",
    "外壳",
    "机身",
    "电池",
    "通电",
    "运行",
    "使用",
    "还能用",
    "继续用",
)


def is_high_risk_technical_request(question: str) -> bool:
    """Detect incident-style product hazards without matching heat-value science."""

    normalized = re.sub(r"\s+", " ", question).strip().lower()
    if any(term in normalized for term in HIGH_RISK_TECHNICAL_TERMS):
        return True
    return bool(
        _CONTEXTUAL_HEAT_PATTERN.search(normalized)
        and any(term in normalized for term in _PRODUCT_HEAT_CONTEXT_TERMS)
    )


class Planner:
    def create_plan(
        self,
        request: AgentRequest,
        *,
        route_override: Literal["technical", "customer_service", "mixed"] | None = None,
    ) -> TaskPlan:
        question = re.sub(r"\s+", " ", request.question).strip().lower()
        has_service = any(term in question for term in SERVICE_TERMS)
        has_identifier = bool(
            re.search(r"\b[a-z]{1,4}\d{2,}[a-z0-9-]*\b", question, flags=re.IGNORECASE)
        )
        has_technical = any(term in question for term in TECHNICAL_TERMS) or has_identifier
        has_mixed_technical = any(term in question for term in MIXED_TECHNICAL_TERMS) or has_identifier
        is_service_dispute = any(term in question for term in SERVICE_DISPUTE_TERMS)
        has_high_risk = is_high_risk_technical_request(question)

        if route_override is not None:
            route = route_override
        elif has_service and has_high_risk:
            route = "mixed"
        elif has_service and is_service_dispute and not has_identifier:
            route = "customer_service"
        elif has_service and has_mixed_technical:
            route = "mixed"
        elif has_service:
            route = "customer_service"
        else:
            route = "technical"

        selected_agents = ["orchestrator"]
        subtasks: list[SubTask] = []
        prior_dependency: list[str] = []

        if request.images:
            selected_agents.append("multimodal")
            subtasks.append(
                SubTask(
                    task_id="vision-1",
                    title="提取图片中的产品、文字和异常信号",
                    route="multimodal",
                    assigned_agent="multimodal",
                )
            )
            prior_dependency = ["vision-1"]

        domain_task_ids: list[str] = []
        if route in {"technical", "mixed"}:
            selected_agents.append("knowledge")
            domain_task_ids.append("knowledge-1")
            subtasks.append(
                SubTask(
                    task_id="knowledge-1",
                    title="检索说明书并形成技术证据",
                    route="technical",
                    depends_on=prior_dependency,
                    assigned_agent="knowledge",
                )
            )

        if route in {"customer_service", "mixed"}:
            selected_agents.append("customer-service")
            domain_task_ids.append("service-1")
            subtasks.append(
                SubTask(
                    task_id="service-1",
                    title="核对售后政策和业务条件",
                    route="customer_service",
                    depends_on=prior_dependency,
                    assigned_agent="customer-service",
                )
            )

        selected_agents.append("verifier")
        subtasks.append(
            SubTask(
                task_id="verify-1",
                title="验证事实、图片、政策和子问题覆盖",
                route="verification",
                depends_on=domain_task_ids,
                assigned_agent="verifier",
            )
        )

        return TaskPlan(
            route=route,
            subtasks=subtasks,
            selected_agents=selected_agents,
            max_tool_calls=6 if route == "mixed" else 4,
            max_retries=1,
        )
