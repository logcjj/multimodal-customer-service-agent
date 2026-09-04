from __future__ import annotations

from uuid import uuid4

from app.contracts.models import (
    AgentResult,
    ClarificationRequest,
)


_PRIORITY = (
    "product",
    "model",
    "error_code",
    "symptom",
    "attempted_action",
    "order_identifier",
    "service_request",
)
_QUESTIONS = {
    "product": "为了确定应检索哪一份手册，请提供产品名称。",
    "model": "为了匹配正确的产品手册，请提供产品型号。",
    "error_code": "请提供屏幕上的错误码；也可以上传一张清晰的报警界面照片。",
    "symptom": "请描述当前能观察到的具体故障现象。",
    "attempted_action": "请说明您已经尝试过哪些处理操作。",
    "order_identifier": "请提供订单号或可用于核验购买记录的订单标识。",
    "service_request": "请说明您希望申请维修、换货还是退货退款。",
}


class EvidenceGapAgent:
    id = "evidence-gap"

    def __init__(self, *, max_rounds: int = 3) -> None:
        self.max_rounds = max_rounds

    def run(
        self,
        *,
        missing_fields: list[str],
        round_number: int,
        case_id: str | None = None,
    ) -> tuple[AgentResult, ClarificationRequest]:
        field = next(
            (candidate for candidate in _PRIORITY if candidate in missing_fields),
            missing_fields[0] if missing_fields else "symptom",
        )
        question = _QUESTIONS.get(field, "请补充能够帮助确认问题的具体信息。")
        accepted = ["text", "image"] if field in {"error_code", "symptom"} else ["text"]
        clarification = ClarificationRequest(
            case_id=case_id or f"case-{uuid4()}",
            field=field,
            question=question,
            round=round_number,
            max_rounds=self.max_rounds,
            accepted_input_types=accepted,
        )
        return (
            AgentResult(
                task_id=f"clarify-{round_number}",
                agent_id=self.id,
                status="needs_input",
                answer_fragment=question,
                confidence=0.8,
                missing_information=[field],
                recommended_next_action=f"collect-{field}",
            ),
            clarification,
        )
