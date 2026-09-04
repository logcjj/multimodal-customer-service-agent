from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.contracts.models import (
    AgentRequest,
    AgentResult,
    CoverageStatus,
    ModelKind,
    RoutingIntent,
)
from app.knowledge.product_router import ProductRouter
from app.models.llm_gateway import LLMGateway
from app.runtime.error_codes import (
    extract_manual_display_error_codes,
    extract_normalized_error_codes,
    has_manual_display_error_code,
    normalize_error_code,
)
from app.runtime.planner import (
    HIGH_RISK_TECHNICAL_TERMS,
    MIXED_TECHNICAL_TERMS,
    SERVICE_DISPUTE_TERMS,
    SERVICE_TERMS,
    TECHNICAL_TERMS,
    is_high_risk_technical_request,
)


_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9-])(?=[A-Za-z0-9-]*[A-Za-z])"
    r"(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]+(?![A-Za-z0-9-])",
    flags=re.IGNORECASE,
)
_HIGH_RISK_TERMS = (
    *HIGH_RISK_TECHNICAL_TERMS,
    "拆机",
    "拆开",
    "自行维修",
    "自己维修",
    "高温",
    "烫得厉害",
    "烫伤",
    "进水",
    "smoke",
    "fire",
    "electric shock",
    "disassemble",
)
_AMBIGUOUS_TECHNICAL_TERMS = {
    "怎么",
    "如何",
    "使用",
    "设置",
    "无法",
    "不能",
    "manual",
}
_FOLLOWUP_TECHNICAL_TERMS = (
    "继续使用",
    "还能使用",
    "能继续用",
    "还可以用",
    "继续操作",
)
_DOMAIN_REFERENCE_TERMS = (
    "产品",
    "商品",
    "设备",
    "机器",
    "型号",
    "说明书",
    "手册",
    "官方参数",
)
_GENERAL_REQUEST_TERMS = (
    "改写",
    "润色",
    "翻译",
    "写一段",
    "写一份",
    "起草",
    "天气",
    "闲聊",
    "你好",
)
_GENERAL_TRANSFORM_TERMS = (
    "改写",
    "润色",
    "翻译",
    "总结",
    "摘要",
    "起草",
    "写一段",
    "写一份",
)
_GENERAL_DOMAIN_SAFETY_TERMS = (
    "保险丝",
    "空气开关",
    "断路器",
    "总闸",
    "插座",
    "接线",
    "电路",
    "电源线",
    "燃气",
    "压力容器",
    "制冷剂",
    "维修步骤",
    "维修方法",
    "维修指导",
    "怎么维修",
    "如何维修",
    "拆卸",
    "拆开",
    "更换零件",
)
_GENERAL_SAFE_DOCUMENT_TERMS = (
    "维修申请",
    "售后申请",
    "退换货申请",
    "服务申请",
)
_GENERAL_PRODUCT_FACT_TERMS = (
    "参数",
    "规格",
    "功率",
    "电压",
    "尺寸",
    "容量",
    "支持什么",
    "是否支持",
    "官方",
)
_IMAGE_TECHNICAL_REQUEST_TERMS = (
    "报错",
    "错误",
    "故障",
    "报警",
    "异常",
    "怎么处理",
    "怎么办",
    "怎么修",
    "如何修",
    "怎么了",
    "哪里坏了",
    "是不是坏了",
)


def general_request_requires_domain(question: str) -> bool:
    """Conservative, deterministic boundary around General Agent requests."""

    normalized = re.sub(r"\s+", " ", question).strip().lower()
    if has_manual_display_error_code(normalized):
        return True
    if is_high_risk_technical_request(normalized) or any(
        term in normalized for term in _HIGH_RISK_TERMS
    ):
        return True
    has_transform = any(term in normalized for term in _GENERAL_TRANSFORM_TERMS)
    safe_document_transform = has_transform and any(
        term in normalized for term in _GENERAL_SAFE_DOCUMENT_TERMS
    )
    if (
        any(term in normalized for term in _GENERAL_DOMAIN_SAFETY_TERMS)
        or ("维修" in normalized and not safe_document_transform)
    ):
        return True
    has_product_signal = (
        bool(_IDENTIFIER_PATTERN.search(normalized))
        or bool(ProductRouter().route(normalized).products)
        or any(term in normalized for term in _DOMAIN_REFERENCE_TERMS)
    )
    if has_product_signal and any(
        term in normalized
        for term in (*_GENERAL_PRODUCT_FACT_TERMS, *TECHNICAL_TERMS)
    ):
        return True
    if has_transform:
        return False
    return False


class _RouterPayload(BaseModel):
    route: Literal[
        "technical_candidate",
        "customer_service_candidate",
        "mixed_candidate",
        "general_candidate",
    ]
    risk: Literal["low", "medium", "high"] = "low"
    reason_code: str = "model-classification"


class CoverageAssessment(BaseModel):
    status: CoverageStatus
    final_route: Literal[
        "technical_knowledge",
        "customer_service",
        "mixed",
        "evidence_clarification",
        "general_llm",
        "general_unavailable",
        "safe_handoff",
    ]
    reason: str
    knowledge_covered: bool
    evidence_count: int = 0
    missing_fields: list[str] = Field(default_factory=list)


class IntentRouter:
    id = "router"

    def __init__(self, llm_gateway: LLMGateway | None) -> None:
        self.llm_gateway = llm_gateway
        self.product_router = ProductRouter()

    def classify(
        self,
        request: AgentRequest,
        *,
        context_text: str = "",
    ) -> RoutingIntent:
        normalized = re.sub(r"\s+", " ", request.question).strip().lower()
        deterministic_route, has_domain_guard = self._deterministic_route(normalized)

        # A photo alone is not evidence that the request needs a manual.  Route
        # visual-identification questions to the VLM, while preserving the
        # technical safety boundary for image-backed faults and repair requests.
        if request.images and deterministic_route == "general_candidate":
            if any(term in normalized for term in _IMAGE_TECHNICAL_REQUEST_TERMS):
                deterministic_route = "technical_candidate"
                has_domain_guard = True
            else:
                return RoutingIntent(
                    initial_route="general_candidate",
                    risk_level="low",
                    requires_knowledge_check=False,
                    reason_code="deterministic-visual-general",
                    classification_source="deterministic",
                )
        deterministic_risk = (
            "high"
            if is_high_risk_technical_request(normalized)
            or any(term in normalized for term in _HIGH_RISK_TERMS)
            else "medium"
        )

        # 产品、故障、安全和售后关键词已经形成强约束时，模型不能改变最终路由。
        # 直接采用确定性结果，避免为相同结论额外等待一次模型请求。
        if has_domain_guard:
            return RoutingIntent(
                initial_route=deterministic_route,
                risk_level=deterministic_risk,
                requires_knowledge_check=True,
                reason_code=self._deterministic_reason(deterministic_route),
                classification_source="deterministic",
            )

        parsed = self._classify_with_llm(request.question, context_text)

        if parsed is None:
            return RoutingIntent(
                initial_route=deterministic_route,
                risk_level=deterministic_risk if has_domain_guard else "low",
                requires_knowledge_check=deterministic_route != "general_candidate",
                reason_code=self._deterministic_reason(deterministic_route),
                classification_source="model_fallback",
            )

        route = parsed.route
        reason_code = parsed.reason_code[:80] or "model-classification"

        risk = self._max_risk(parsed.risk, "low")
        return RoutingIntent(
            initial_route=route,
            risk_level=risk,
            requires_knowledge_check=route != "general_candidate",
            reason_code=reason_code,
            llm_used=True,
            model_used=self.llm_gateway.model_name(ModelKind.LLM) if self.llm_gateway else None,
            classification_source="model",
        )

    def _classify_with_llm(
        self,
        question: str,
        context_text: str,
    ) -> _RouterPayload | None:
        gateway = self.llm_gateway
        if gateway is None or not gateway.available(ModelKind.LLM):
            return None
        output = gateway.generate(
            kind=ModelKind.LLM,
            system_prompt=(
                "你是客服 Multi-Agent 的路由分类器，不负责回答问题。"
                "只输出 JSON 对象，字段为 route、risk、reason_code。"
                "route 只能是 technical_candidate、customer_service_candidate、"
                "mixed_candidate、general_candidate。产品事实、故障、维修、安全操作必须归技术；"
                "退款、物流、订单、赔付归客服；两者都有归混合；"
                "写作、翻译、普通知识和闲聊才归通用。不得输出解释或 Markdown。"
            ),
            user_prompt=(
                f"会话相关上下文：{context_text[:1800] or '无'}\n"
                f"当前用户问题：{question}"
            ),
            temperature=0,
            max_tokens=120,
        )
        if output is None or not output.text.strip():
            return None
        try:
            return _RouterPayload.model_validate(json.loads(self._json_object(output.text)))
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError):
            return None

    def _deterministic_route(
        self,
        normalized: str,
    ) -> tuple[
        Literal[
            "technical_candidate",
            "customer_service_candidate",
            "mixed_candidate",
            "general_candidate",
        ],
        bool,
    ]:
        has_service = any(term in normalized for term in SERVICE_TERMS)
        has_identifier = bool(_IDENTIFIER_PATTERN.search(normalized))
        has_manual_error_code = has_manual_display_error_code(normalized)
        has_order_identifier = any(
            term in normalized
            for term in ("订单号", "订单编号", "订单标识", "单号")
        )
        has_technical_identifier = (
            has_identifier or has_manual_error_code
        ) and not has_order_identifier
        has_product = bool(self.product_router.route(normalized).products)
        has_domain_reference = any(
            term in normalized for term in _DOMAIN_REFERENCE_TERMS
        )
        has_specific_technical = any(
            term in normalized
            for term in TECHNICAL_TERMS
            if term not in _AMBIGUOUS_TECHNICAL_TERMS
        )
        has_technical = (
            has_specific_technical
            or has_technical_identifier
            or has_product
            or has_domain_reference
            or is_high_risk_technical_request(normalized)
            or any(term in normalized for term in _HIGH_RISK_TERMS)
            or any(term in normalized for term in _FOLLOWUP_TECHNICAL_TERMS)
        )
        has_mixed_technical = (
            any(term in normalized for term in MIXED_TECHNICAL_TERMS)
            or has_technical_identifier
            or is_high_risk_technical_request(normalized)
            or any(term in normalized for term in _HIGH_RISK_TERMS)
        )
        is_service_dispute = any(term in normalized for term in SERVICE_DISPUTE_TERMS)
        has_high_risk = is_high_risk_technical_request(normalized) or any(
            term in normalized for term in _HIGH_RISK_TERMS
        )

        if has_service and has_high_risk:
            return "mixed_candidate", True
        if has_service and is_service_dispute and not has_technical_identifier:
            return "customer_service_candidate", True
        if has_service and has_mixed_technical:
            return "mixed_candidate", True
        if has_service:
            return "customer_service_candidate", True
        if has_technical:
            return "technical_candidate", True
        if any(term in normalized for term in _GENERAL_REQUEST_TERMS):
            return "general_candidate", False
        return "general_candidate", False

    @staticmethod
    def _json_object(text: str) -> str:
        normalized = text.strip()
        start = normalized.find("{")
        end = normalized.rfind("}")
        if start < 0 or end < start:
            raise ValueError("router output does not contain a JSON object")
        return normalized[start : end + 1]

    @staticmethod
    def _deterministic_reason(route: str) -> str:
        return {
            "technical_candidate": "deterministic-technical",
            "customer_service_candidate": "deterministic-customer-service",
            "mixed_candidate": "deterministic-mixed",
            "general_candidate": "deterministic-general",
        }[route]

    @staticmethod
    def _max_risk(
        first: Literal["low", "medium", "high"],
        second: Literal["low", "medium", "high"],
    ) -> Literal["low", "medium", "high"]:
        order = {"low": 0, "medium": 1, "high": 2}
        return first if order[first] >= order[second] else second


class KnowledgeCoverageGate:
    id = "knowledge-coverage"

    def __init__(self, *, max_rounds: int = 3) -> None:
        self.max_rounds = max_rounds
        self.product_router = ProductRouter()

    def evaluate(
        self,
        *,
        intent: RoutingIntent,
        question: str,
        results: list[AgentResult],
        active_slots: dict[str, str],
        clarification_round: int,
    ) -> CoverageAssessment:
        if intent.initial_route == "general_candidate":
            if general_request_requires_domain(question):
                return CoverageAssessment(
                    status=CoverageStatus.UNSAFE_UNCOVERED,
                    final_route="safe_handoff",
                    reason="问题涉及产品事实、维修或安全操作，不能由通用模型自由生成",
                    knowledge_covered=False,
                )
            return CoverageAssessment(
                status=CoverageStatus.GENERAL_ALLOWED,
                final_route="general_llm",
                reason="问题不依赖产品手册或客服政策",
                knowledge_covered=False,
            )

        evidence = [
            item
            for result in results
            for item in result.evidence
        ]
        manual = [
            item
            for item in evidence
            if item.source_type in {"manual", "image"}
        ]
        policy = [item for item in evidence if item.source_type == "policy"]
        route = intent.initial_route
        missing = self._missing_fields(question, active_slots, route)
        if (
            "model" in missing
            and self._known_product_error_supported(
                question,
                active_slots,
                manual,
            )
        ):
            missing = [field for field in missing if field != "model"]
        must_clarify = bool(missing) and (
            self._requires_disambiguation(question)
            or any(
                field in {"order_identifier", "service_request"}
                for field in missing
            )
        )
        if must_clarify:
            if intent.risk_level == "high" or clarification_round >= self.max_rounds:
                return CoverageAssessment(
                    status=CoverageStatus.UNSAFE_UNCOVERED,
                    final_route="safe_handoff",
                    reason=(
                        "高风险产品问题缺少可靠依据"
                        if intent.risk_level == "high"
                        else f"已达到 {self.max_rounds} 轮证据补全上限"
                    ),
                    knowledge_covered=False,
                    evidence_count=len(evidence),
                    missing_fields=missing,
                )
            return CoverageAssessment(
                status=CoverageStatus.CLARIFIABLE,
                final_route="evidence_clarification",
                reason=f"仍需补充{self._field_label(missing[0])}",
                knowledge_covered=False,
                evidence_count=len(evidence),
                missing_fields=missing,
            )
        covered = (
            bool(manual)
            if route == "technical_candidate"
            else bool(policy)
            if route == "customer_service_candidate"
            else bool(manual) and bool(policy)
        )
        if covered and not self._explicit_error_code_conflicts(
            question,
            evidence,
            active_slots,
        ):
            final_route = {
                "technical_candidate": "technical_knowledge",
                "customer_service_candidate": "customer_service",
                "mixed_candidate": "mixed",
            }[route]
            return CoverageAssessment(
                status=CoverageStatus.COVERED,
                final_route=final_route,
                reason=f"检索到 {len(evidence)} 条可用领域证据",
                knowledge_covered=True,
                evidence_count=len(evidence),
            )

        if intent.risk_level == "high" or clarification_round >= self.max_rounds:
            return CoverageAssessment(
                status=CoverageStatus.UNSAFE_UNCOVERED,
                final_route="safe_handoff",
                reason=(
                    "高风险产品问题缺少可靠依据"
                    if intent.risk_level == "high"
                    else f"已达到 {self.max_rounds} 轮证据补全上限"
                ),
                knowledge_covered=False,
                evidence_count=len(evidence),
            )

        if missing:
            return CoverageAssessment(
                status=CoverageStatus.CLARIFIABLE,
                final_route="evidence_clarification",
                reason=f"仍需补充{self._field_label(missing[0])}",
                knowledge_covered=False,
                evidence_count=len(evidence),
                missing_fields=missing,
            )
        return CoverageAssessment(
            status=CoverageStatus.UNSAFE_UNCOVERED,
            final_route="safe_handoff",
            reason="具体产品问题未检索到可靠手册或政策依据",
            knowledge_covered=False,
            evidence_count=len(evidence),
        )

    @staticmethod
    def _requires_disambiguation(question: str) -> bool:
        normalized = question.lower()
        return any(
            term in normalized
            for term in (
                "报警",
                "报错",
                "错误",
                "故障码",
                "指示灯",
                "error",
                "alarm",
            )
        )

    def _missing_fields(
        self,
        question: str,
        active_slots: dict[str, str],
        route: str,
    ) -> list[str]:
        normalized = question.lower()
        products = self.product_router.route(question).products
        missing: list[str] = []
        if route in {"technical_candidate", "mixed_candidate"}:
            if not active_slots.get("product") and not products:
                missing.append("product")
            alarm = any(
                term in normalized
                for term in ("报警", "报错", "错误", "故障码", "指示灯", "error", "alarm")
            ) or has_manual_display_error_code(question)
            if alarm:
                compact_identifiers = {
                    item.upper() for item in _IDENTIFIER_PATTERN.findall(question)
                }
                manual_error_codes = set(
                    extract_manual_display_error_codes(question)
                )
                product_identifiers = {
                    item.upper()
                    for item in _IDENTIFIER_PATTERN.findall(
                        active_slots.get("product", "")
                    )
                }
                compact_identifiers -= product_identifiers
                active_model = active_slots.get("model", "").upper()
                active_error = active_slots.get("error_code", "").upper()
                model_identifiers = {
                    item
                    for item in compact_identifiers
                    if item not in {active_model, active_error}
                }
                error_identifiers = {
                    *compact_identifiers,
                    *manual_error_codes,
                }
                has_error_identifier = any(
                    item != active_model for item in error_identifiers
                )
                if not active_slots.get("model") and not model_identifiers:
                    missing.append("model")
                if not active_slots.get("error_code") and not has_error_identifier:
                    missing.append("error_code")
            elif not active_slots.get("model") and not products:
                missing.append("model")
        if route in {"customer_service_candidate", "mixed_candidate"}:
            realtime_order_terms = (
                "我的订单",
                "订单状态",
                "查询订单",
                "查订单",
                "物流到哪",
                "物流进度",
                "配送进度",
                "包裹到哪",
                "为什么还没送到",
                "什么时候送到",
                "当前物流",
            )
            requires_order_context = any(
                term in normalized for term in realtime_order_terms
            )
            if requires_order_context and not active_slots.get("order_identifier"):
                missing.append("order_identifier")
            explicit_request = any(
                term in normalized
                for term in ("退款", "退货", "换货", "维修", "查物流", "查询物流")
            )
            ambiguous_request = any(
                term in normalized
                for term in ("有问题", "怎么办", "怎么处理", "需要处理")
            )
            if (
                requires_order_context
                and ambiguous_request
                and not explicit_request
                and not active_slots.get("service_request")
            ):
                missing.append("service_request")
        return list(dict.fromkeys(missing))

    def _known_product_error_supported(
        self,
        question: str,
        active_slots: dict[str, str],
        manual_evidence,
    ) -> bool:
        known_product = bool(
            active_slots.get("product")
            or self.product_router.route(question).products
        )
        if not known_product or not manual_evidence:
            return False
        requested_codes = set(extract_normalized_error_codes(question))
        active_error = normalize_error_code(
            active_slots.get("error_code", "")
        )
        if active_error is not None:
            requested_codes.add(active_error)
        if not requested_codes:
            return False
        support = "\n".join(
            f"{item.product or ''}\n{item.document_name or ''}\n{item.title}\n{item.text}"
            for item in manual_evidence
        )
        supported_codes = set(extract_normalized_error_codes(support))
        return requested_codes <= supported_codes

    @staticmethod
    def _explicit_error_code_conflicts(
        question: str,
        evidence,
        active_slots: dict[str, str],
    ) -> bool:
        if not has_manual_display_error_code(question) and not any(
            term in question.lower()
            for term in ("错误码", "故障码", "报错", "显示", "error")
        ):
            return False
        requested = set(extract_normalized_error_codes(question))
        active_error = normalize_error_code(active_slots.get("error_code", ""))
        if active_error is not None:
            requested.add(active_error)
        if not requested:
            return False
        support = "\n".join(
            f"{item.product or ''}\n{item.document_name or ''}\n{item.title}\n{item.text}"
            for item in evidence
        )
        supported = set(extract_normalized_error_codes(support))
        return not requested <= supported

    @staticmethod
    def _field_label(field: str) -> str:
        return {
            "product": "产品名称",
            "model": "产品型号",
            "error_code": "错误码或报警图片",
            "symptom": "具体故障现象",
            "attempted_action": "已经尝试的操作",
            "order_identifier": "订单标识",
            "service_request": "售后诉求",
        }.get(field, field)
