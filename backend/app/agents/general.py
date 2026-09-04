from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from app.contracts.models import AgentRequest, AgentResult, ModelKind
from app.models.llm_gateway import LLMGateway


class GeneralAgent:
    id = "general"

    def __init__(
        self,
        llm_gateway: LLMGateway | None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.llm_gateway = llm_gateway
        self.now = now or (lambda: datetime.now(ZoneInfo("Asia/Shanghai")))

    def run(self, request: AgentRequest, *, context_text: str) -> AgentResult:
        started = perf_counter()
        gateway = self.llm_gateway
        model_kind = ModelKind.VLM if request.images else ModelKind.LLM
        if gateway is None or not gateway.available(model_kind):
            return AgentResult(
                task_id="general-1",
                agent_id=self.id,
                status="failed",
                confidence=0,
                recommended_next_action="general-model-unavailable",
                latency_ms=round((perf_counter() - started) * 1000),
            )
        output = gateway.generate(
            kind=model_kind,
            system_prompt=(
                "你是通用对话智能体，只处理不依赖产品手册、客服政策或实时订单系统的普通问题。"
                "直接回答当前请求。不得假装引用产品说明书，不得编造产品参数、维修步骤、"
                "安全操作、退款结果或官方承诺。如果问题实际要求这些内容，明确说明应返回专业客服链。"
                "用户提示中提供的当前服务器时间可信；遇到日期或时间问题时必须据此直接回答，"
                "不得声称无法获取当前日期。"
                "当用户附带图片时，只描述图片中实际可见的对象、文字和场景；无法确认的内容必须"
                "使用“疑似”“可能”或“看起来”等限定，不能把图片内容臆断为产品故障、医疗结论或官方信息。"
                "严格只输出一个 JSON 对象：answer 为 Markdown 格式回答，confidence 为 0 到 1 的自评置信度。"
            ),
            user_prompt=(
                f"当前服务器时间（中国标准时间）：{self._current_time_text()}\n"
                f"相关会话上下文：{context_text[:4000] or '无'}\n"
                f"当前用户问题：{request.question}"
            ),
            images=request.images,
            temperature=0.2,
            max_tokens=1200,
        )
        if output is None or not output.text.strip():
            return AgentResult(
                task_id="general-1",
                agent_id=self.id,
                status="failed",
                confidence=0,
                recommended_next_action="general-model-unavailable",
                latency_ms=round((perf_counter() - started) * 1000),
            )
        answer, reported_confidence = self._parse_output(output.text)
        if not answer:
            return AgentResult(
                task_id="general-1",
                agent_id=self.id,
                status="failed",
                confidence=0,
                recommended_next_action="general-model-unavailable",
                latency_ms=round((perf_counter() - started) * 1000),
            )
        return AgentResult(
            task_id="general-1",
            agent_id=self.id,
            status="completed",
            answer_fragment=answer,
            confidence=self._confidence(
                answer,
                reported_confidence=reported_confidence,
                is_visual=model_kind == ModelKind.VLM,
            ),
            latency_ms=round((perf_counter() - started) * 1000),
            llm_generated=True,
            model_used=output.model,
        )

    @staticmethod
    def _parse_output(text: str) -> tuple[str, float | None]:
        """Accept structured output while retaining a safe plain-text fallback."""

        clean = text.strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end >= start:
            try:
                payload = json.loads(clean[start : end + 1])
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                answer = str(payload.get("answer") or "").strip()
                try:
                    confidence = float(payload.get("confidence"))
                except (TypeError, ValueError):
                    confidence = None
                if answer:
                    return answer, confidence
        return clean, None

    @staticmethod
    def _confidence(
        answer: str,
        *,
        reported_confidence: float | None,
        is_visual: bool,
    ) -> float:
        """Calibrate model self-assessment with observable response completeness."""

        detail = min(0.14, max(0, len(answer.strip()) - 20) / 900 * 0.14)
        baseline = 0.52 + detail + (0.08 if is_visual else 0.04)
        if reported_confidence is None:
            return round(min(0.8, baseline), 4)
        reported = min(1.0, max(0.0, reported_confidence))
        return round(min(0.92, max(0.32, reported * 0.68 + baseline * 0.32)), 4)

    def _current_time_text(self) -> str:
        current = self.now()
        timezone = ZoneInfo("Asia/Shanghai")
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone)
        else:
            current = current.astimezone(timezone)
        weekday = (
            "星期一",
            "星期二",
            "星期三",
            "星期四",
            "星期五",
            "星期六",
            "星期日",
        )[current.weekday()]
        return current.strftime(f"%Y年%m月%d日 {weekday} %H:%M")
