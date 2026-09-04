from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field
from sqlmodel import select

from app.contracts.models import (
    ClarificationRequest,
    ConversationTurnView,
    RoutingIntent,
    VisualContext,
)
from app.conversations.models import ConversationRecord, ConversationStateRecord
from app.conversations.store import ConversationStore
from app.knowledge.product_router import ProductRouter
from app.runtime.error_codes import (
    extract_manual_display_error_codes,
    has_manual_display_error_code,
    normalize_error_code,
)
from app.storage.database import Database


_MODEL_PATTERN = re.compile(
    r"(?:型号)(?:是|为|[:：])?\s*([A-Za-z0-9][A-Za-z0-9-]{1,40})",
    flags=re.IGNORECASE,
)
_ERROR_PATTERN = re.compile(
    r"(?:错误码|故障码|报错|显示)(?:是|为|[:：])?\s*"
    r"((?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{2,40})",
    flags=re.IGNORECASE,
)
_CORRECTION_PATTERN = re.compile(
    r"不是\s*([A-Za-z0-9-]{2,40})\s*[，,、]?\s*(?:而)?是\s*"
    r"([A-Za-z0-9-]{2,40})",
    flags=re.IGNORECASE,
)
_SYMPTOM_PATTERN = re.compile(
    r"(?:现象|症状|表现)(?:是|为|[:：])?\s*(.{2,200})",
    flags=re.IGNORECASE,
)
_ATTEMPTED_ACTION_PATTERN = re.compile(
    r"(?:已经|我已|之前)?\s*(?:尝试过|试过|做过)\s*(.{2,200})",
    flags=re.IGNORECASE,
)
_ORDER_IDENTIFIER_PATTERN = re.compile(
    r"(?:订单号|订单编号|订单标识|单号)(?:是|为|[:：])?\s*"
    r"([A-Za-z0-9][A-Za-z0-9-]{3,63})",
    flags=re.IGNORECASE,
)
_SERVICE_REQUEST_TERMS = (
    "退货退款",
    "申请退款",
    "退款",
    "申请退货",
    "退货",
    "申请换货",
    "换货",
    "申请维修",
    "维修",
)
_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")


class MemorySlot(BaseModel):
    value: str
    source_turn_id: str
    source_kind: Literal["explicit_user", "ocr", "vlm", "inferred"]
    confidence: float = Field(ge=0, le=1)
    status: Literal["active", "superseded"] = "active"
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PendingClarificationState(ClarificationRequest):
    original_question: str


class PromptContext(BaseModel):
    text: str
    estimated_tokens: int
    included_ordinals: list[int] = Field(default_factory=list)
    rolling_summary: str = ""


class ConversationContext(BaseModel):
    conversation_id: str
    slots: dict[str, list[MemorySlot]] = Field(default_factory=dict)
    prompt: PromptContext
    pending_clarification: PendingClarificationState | None = None

    def active_slot(self, name: str) -> MemorySlot | None:
        return next(
            (
                item
                for item in reversed(self.slots.get(name, []))
                if item.status == "active"
            ),
            None,
        )

    def active_value(self, name: str) -> str | None:
        slot = self.active_slot(name)
        return slot.value if slot is not None else None

    def superseded_values(self, name: str) -> list[str]:
        return [
            item.value
            for item in self.slots.get(name, [])
            if item.status == "superseded"
        ]

    def context_text(self) -> str:
        labels = {
            "product": "当前产品",
            "model": "当前型号",
            "error_code": "当前错误码",
            "symptom": "当前现象",
            "order_identifier": "当前订单标识",
            "service_request": "当前售后诉求",
        }
        slot_lines = [
            f"{label}：{value}"
            for name, label in labels.items()
            if (value := self.active_value(name))
        ]
        parts = [
            *slot_lines,
            (
                f"待补充字段：{self.pending_clarification.field}\n"
                f"原始问题：{self.pending_clarification.original_question}"
                if self.pending_clarification
                else ""
            ),
            self.prompt.text,
        ]
        return "\n".join(item for item in parts if item).strip()

    def structured_context_text(self) -> str:
        labels = {
            "product": "当前产品",
            "model": "当前型号",
            "error_code": "当前错误码",
            "symptom": "当前现象",
            "attempted_action": "已尝试操作",
            "order_identifier": "当前订单标识",
            "service_request": "当前售后诉求",
        }
        return "\n".join(
            f"{label}：{value}"
            for name, label in labels.items()
            if (value := self.active_value(name))
        )


class ConversationMemoryService:
    def __init__(
        self,
        database: Database,
        conversations: ConversationStore,
        *,
        context_tokens: int = 6000,
    ) -> None:
        self.database = database
        self.conversations = conversations
        self.context_tokens = context_tokens
        self.product_router = ProductRouter()

    def record_user_turn(
        self,
        conversation_id: str,
        owner_id: str,
        turn_id: str,
        text: str,
        visual_context: VisualContext | None = None,
    ) -> None:
        with self.conversations.conversation_lock(conversation_id):
            self._record_user_turn_locked(
                conversation_id,
                owner_id,
                turn_id,
                text,
                visual_context,
            )

    def _record_user_turn_locked(
        self,
        conversation_id: str,
        owner_id: str,
        turn_id: str,
        text: str,
        visual_context: VisualContext | None = None,
    ) -> None:
        with self.database.session() as session:
            self._lock_owned_conversation(
                session,
                conversation_id,
                owner_id,
            )
            state = session.exec(
                select(ConversationStateRecord)
                .where(
                    ConversationStateRecord.conversation_id == conversation_id
                )
                .with_for_update()
            ).first()
            if state is None:
                state = ConversationStateRecord(conversation_id=conversation_id)
            slots = self._load_slots(state.slots_json)
            now = datetime.now(UTC)
            pending = self._pending(state.pending_clarification_json)

            routed = self.product_router.route(text)
            if routed.products:
                product = routed.products[0]
                previous_product = self._active_value(slots, "product")
                if previous_product and previous_product != product:
                    self._supersede(slots, "product")
                    self._supersede(slots, "model")
                    self._supersede(slots, "error_code")
                    state.pending_clarification_json = None
                    state.clarification_round = 0
                self._set_slot(
                    slots,
                    "product",
                    product,
                    turn_id,
                    "explicit_user",
                    1.0,
                    now,
                )

            correction = _CORRECTION_PATTERN.search(text)
            if correction:
                old_value, new_value = (
                    correction.group(1).upper(),
                    correction.group(2).upper(),
                )
                corrected_name = next(
                    (
                        name
                        for name in ("error_code", "model")
                        if self._active_value(slots, name) == old_value
                    ),
                    "error_code",
                )
                if corrected_name == "error_code":
                    new_value = normalize_error_code(new_value) or new_value
                self._set_slot(
                    slots,
                    corrected_name,
                    new_value,
                    turn_id,
                    "explicit_user",
                    1.0,
                    now,
                )
            else:
                model_match = _MODEL_PATTERN.search(text)
                if model_match:
                    self._set_slot(
                        slots,
                        "model",
                        model_match.group(1).upper(),
                        turn_id,
                        "explicit_user",
                        1.0,
                        now,
                    )
                error_match = _ERROR_PATTERN.search(text)
                manual_error_codes = extract_manual_display_error_codes(text)
                error_code = (
                    normalize_error_code(error_match.group(1))
                    or error_match.group(1).upper()
                    if error_match
                    else manual_error_codes[0]
                    if manual_error_codes
                    else None
                )
                if error_code:
                    self._set_slot(
                        slots,
                        "error_code",
                        error_code,
                        turn_id,
                        "explicit_user",
                        1.0,
                        now,
                    )

            symptom_match = _SYMPTOM_PATTERN.search(text)
            if symptom_match:
                self._set_slot(
                    slots,
                    "symptom",
                    symptom_match.group(1),
                    turn_id,
                    "explicit_user",
                    1.0,
                    now,
                )
            attempted_match = _ATTEMPTED_ACTION_PATTERN.search(text)
            if attempted_match:
                self._set_slot(
                    slots,
                    "attempted_action",
                    attempted_match.group(1),
                    turn_id,
                    "explicit_user",
                    1.0,
                    now,
                )
            order_match = _ORDER_IDENTIFIER_PATTERN.search(text)
            if order_match:
                self._set_slot(
                    slots,
                    "order_identifier",
                    order_match.group(1).upper(),
                    turn_id,
                    "explicit_user",
                    1.0,
                    now,
                )
            service_request = next(
                (term for term in _SERVICE_REQUEST_TERMS if term in text),
                None,
            )
            if service_request:
                normalized_request = next(
                    (
                        value
                        for value in ("退货退款", "退款", "退货", "换货", "维修")
                        if value in service_request
                    ),
                    service_request,
                )
                self._set_slot(
                    slots,
                    "service_request",
                    normalized_request,
                    turn_id,
                    "explicit_user",
                    1.0,
                    now,
                )
            if pending is not None:
                pending_value = text.strip()
                bare_identifier = re.fullmatch(
                    r"(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
                    r"[A-Za-z0-9-]{2,40}",
                    pending_value,
                )
                if (
                    pending.field == "product"
                    and not routed.products
                    and self._looks_like_product_value(pending_value)
                ):
                    self._set_slot(
                        slots, "product", pending_value, turn_id,
                        "explicit_user", 1.0, now,
                    )
                elif pending.field == "model" and model_match is None and bare_identifier:
                    self._set_slot(
                        slots, "model", pending_value.upper(), turn_id,
                        "explicit_user", 1.0, now,
                    )
                elif (
                    pending.field == "error_code"
                    and error_match is None
                    and bare_identifier
                ):
                    self._set_slot(
                        slots,
                        "error_code",
                        normalize_error_code(pending_value) or pending_value.upper(),
                        turn_id,
                        "explicit_user", 1.0, now,
                    )
                elif pending.field == "symptom" and symptom_match is None:
                    self._set_slot(
                        slots, "symptom", pending_value, turn_id,
                        "explicit_user", 1.0, now,
                    )
                elif pending.field == "attempted_action" and attempted_match is None:
                    self._set_slot(
                        slots, "attempted_action", pending_value, turn_id,
                        "explicit_user", 1.0, now,
                    )
                elif (
                    pending.field == "order_identifier"
                    and order_match is None
                    and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{3,63}", pending_value)
                ):
                    self._set_slot(
                        slots, "order_identifier", pending_value.upper(), turn_id,
                        "explicit_user", 1.0, now,
                    )
                elif pending.field == "service_request" and service_request is None:
                    self._set_slot(
                        slots, "service_request", pending_value, turn_id,
                        "explicit_user", 1.0, now,
                    )

            if visual_context is not None:
                if visual_context.detected_product:
                    self._set_slot(
                        slots,
                        "product",
                        visual_context.detected_product,
                        turn_id,
                        "vlm",
                        visual_context.confidence,
                        now,
                    )
                for code in visual_context.detected_codes[:1]:
                    self._set_slot(
                        slots,
                        "error_code",
                        code.upper(),
                        turn_id,
                        "ocr",
                        visual_context.confidence,
                        now,
                    )

            state.slots_json = self._dump_slots(slots)
            state.updated_at = now
            session.add(state)
            session.commit()

    def load_context(
        self,
        conversation_id: str,
        owner_id: str,
        current_question: str,
    ) -> ConversationContext | None:
        detail = self.conversations.get(conversation_id, owner_id)
        if detail is None:
            return None
        with self.database.session() as session:
            state = session.get(ConversationStateRecord, conversation_id)
            if state is None:
                slots: dict[str, list[MemorySlot]] = {}
                rolling_summary = ""
                pending = None
            else:
                slots = self._load_slots(state.slots_json)
                rolling_summary = state.rolling_summary
                pending = self._pending(state.pending_clarification_json)
        return ConversationContext(
            conversation_id=conversation_id,
            slots=slots,
            prompt=self.build_prompt_context(
                detail.turns,
                rolling_summary=rolling_summary,
                current_question=current_question,
                budget_tokens=self.context_tokens,
            ),
            pending_clarification=pending,
        )

    def set_pending_clarification(
        self,
        conversation_id: str,
        owner_id: str,
        clarification: ClarificationRequest,
        *,
        original_question: str,
    ) -> None:
        with self.conversations.conversation_lock(conversation_id):
            self._set_pending_clarification_locked(
                conversation_id,
                owner_id,
                clarification,
                original_question=original_question,
            )

    def _set_pending_clarification_locked(
        self,
        conversation_id: str,
        owner_id: str,
        clarification: ClarificationRequest,
        *,
        original_question: str,
    ) -> None:
        pending = PendingClarificationState(
            **clarification.model_dump(),
            original_question=original_question,
        )
        with self.database.session() as session:
            self._lock_owned_conversation(
                session,
                conversation_id,
                owner_id,
            )
            state = session.exec(
                select(ConversationStateRecord)
                .where(
                    ConversationStateRecord.conversation_id == conversation_id
                )
                .with_for_update()
            ).first()
            if state is None:
                state = ConversationStateRecord(conversation_id=conversation_id)
            state.pending_clarification_json = pending.model_dump_json()
            state.clarification_round = clarification.round
            state.updated_at = datetime.now(UTC)
            session.add(state)
            session.commit()

    def clear_pending_clarification(
        self,
        conversation_id: str,
        owner_id: str,
    ) -> None:
        with self.conversations.conversation_lock(conversation_id):
            self._clear_pending_clarification_locked(conversation_id, owner_id)

    def _clear_pending_clarification_locked(
        self,
        conversation_id: str,
        owner_id: str,
    ) -> None:
        with self.database.session() as session:
            self._lock_owned_conversation(
                session,
                conversation_id,
                owner_id,
            )
            state = session.exec(
                select(ConversationStateRecord)
                .where(
                    ConversationStateRecord.conversation_id == conversation_id
                )
                .with_for_update()
            ).first()
            if state is None:
                return
            state.pending_clarification_json = None
            state.clarification_round = 0
            state.updated_at = datetime.now(UTC)
            session.add(state)
            session.commit()

    @staticmethod
    def should_end_pending_for_topic_switch(
        pending: PendingClarificationState,
        text: str,
        *,
        has_images: bool = False,
        candidate_intent: RoutingIntent | None = None,
    ) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        if has_images and "image" in pending.accepted_input_types:
            return False
        if ConversationMemoryService.pending_reply_has_expected_shape(pending, text):
            return False
        if candidate_intent is not None:
            return candidate_intent.initial_route == "general_candidate"
        general_switch_terms = (
            "改写",
            "润色",
            "翻译",
            "总结",
            "起草",
            "写一段",
            "写一份",
            "天气",
            "你好",
            "聊聊",
        )
        return any(term in normalized for term in general_switch_terms)

    @staticmethod
    def pending_reply_has_expected_shape(
        pending: PendingClarificationState,
        text: str,
    ) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        if pending.field == "product":
            return bool(
                ProductRouter().route(normalized).products
                or ConversationMemoryService._looks_like_product_value(
                    normalized
                )
            )
        if pending.field == "error_code" and has_manual_display_error_code(
            normalized
        ):
            return True
        if pending.field in {"model", "error_code"}:
            return bool(
                re.fullmatch(
                    r"(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)"
                    r"[A-Za-z0-9-]{2,40}",
                    normalized,
                )
            )
        if pending.field == "order_identifier":
            return bool(
                _ORDER_IDENTIFIER_PATTERN.search(normalized)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{3,63}", normalized)
            )
        if pending.field == "service_request":
            return any(term in normalized for term in _SERVICE_REQUEST_TERMS)
        if pending.field == "symptom":
            return _SYMPTOM_PATTERN.search(normalized) is not None
        if pending.field == "attempted_action":
            return _ATTEMPTED_ACTION_PATTERN.search(normalized) is not None
        return False

    @staticmethod
    def _looks_like_product_value(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        if not normalized or len(normalized) > 32:
            return False
        if re.search(r"[?？。！!，,；;：:]", normalized):
            return False
        identifier_device = re.fullmatch(
            r"(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)[a-z0-9-]+ "
            r"(?:设备|机器|装置|终端)",
            normalized,
        )
        branded_product = re.fullmatch(
            r"[a-z0-9\-\u3400-\u9fff]{1,12}牌"
            r"[a-z0-9\-\u3400-\u9fff]{1,12}"
            r"(?:设备|机器|装置|终端|机器人|机|器|仪|表|柜|盒|泵|炉|车|锁|灯|扇|箱)",
            normalized,
        )
        return bool(identifier_device or branded_product)

    def save_summary(
        self,
        conversation_id: str,
        owner_id: str,
        summary: str,
        *,
        through_ordinal: int,
    ) -> None:
        with self.conversations.conversation_lock(conversation_id):
            self._save_summary_locked(
                conversation_id,
                owner_id,
                summary,
                through_ordinal=through_ordinal,
            )

    def _save_summary_locked(
        self,
        conversation_id: str,
        owner_id: str,
        summary: str,
        *,
        through_ordinal: int,
    ) -> None:
        with self.database.session() as session:
            self._lock_owned_conversation(
                session,
                conversation_id,
                owner_id,
            )
            state = session.exec(
                select(ConversationStateRecord)
                .where(
                    ConversationStateRecord.conversation_id == conversation_id
                )
                .with_for_update()
            ).first()
            if state is None:
                state = ConversationStateRecord(conversation_id=conversation_id)
            if through_ordinal < state.summary_through_ordinal:
                return
            state.rolling_summary = summary.strip()[:4000]
            state.summary_through_ordinal = through_ordinal
            state.memory_version += 1
            state.updated_at = datetime.now(UTC)
            session.add(state)
            session.commit()

    @staticmethod
    def _lock_owned_conversation(
        session,
        conversation_id: str,
        owner_id: str,
    ) -> ConversationRecord:
        conversation = session.exec(
            select(ConversationRecord)
            .where(ConversationRecord.id == conversation_id)
            .with_for_update()
        ).first()
        if conversation is None or conversation.owner_id != owner_id:
            raise PermissionError("conversation owner mismatch")
        return conversation

    @classmethod
    def build_prompt_context(
        cls,
        turns: list[ConversationTurnView],
        *,
        rolling_summary: str,
        current_question: str,
        budget_tokens: int,
    ) -> PromptContext:
        summary = cls._truncate_tokens(
            rolling_summary.strip(),
            max(0, min(budget_tokens // 4, budget_tokens - cls._estimate_tokens(current_question))),
        )
        base_parts = [f"更早对话摘要：{summary}" if summary else ""]
        used = cls._estimate_tokens(current_question) + cls._estimate_tokens(
            base_parts[0]
        )
        selected: list[tuple[int, str]] = []
        for turn in reversed(turns):
            if turn.status != "completed":
                continue
            assistant = turn.assistant_text.strip()
            block = f"用户：{turn.user_text.strip()}"
            if assistant:
                block += f"\n助手：{assistant}"
            block_tokens = cls._estimate_tokens(block)
            if used + block_tokens > budget_tokens:
                continue
            selected.append((turn.ordinal, block))
            used += block_tokens
        selected.reverse()
        parts = [item for item in base_parts if item]
        parts.extend(block for _, block in selected)
        text = "\n\n".join(parts)
        return PromptContext(
            text=text,
            estimated_tokens=cls._estimate_tokens(text) + cls._estimate_tokens(
                current_question
            ),
            included_ordinals=[ordinal for ordinal, _ in selected],
            rolling_summary=summary,
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        cjk = len(_CJK_PATTERN.findall(text))
        non_cjk = len(_CJK_PATTERN.sub("", text))
        return cjk + math.ceil(non_cjk / 4)

    @classmethod
    def _truncate_tokens(cls, text: str, budget: int) -> str:
        if budget <= 0 or not text:
            return ""
        if cls._estimate_tokens(text) <= budget:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if cls._estimate_tokens(text[:middle]) <= budget:
                low = middle
            else:
                high = middle - 1
        return text[:low].rstrip()

    @staticmethod
    def _set_slot(
        slots: dict[str, list[MemorySlot]],
        name: str,
        value: str,
        source_turn_id: str,
        source_kind: Literal["explicit_user", "ocr", "vlm", "inferred"],
        confidence: float,
        now: datetime,
    ) -> None:
        normalized = value.strip()
        if not normalized:
            return
        active = next(
            (
                item
                for item in reversed(slots.get(name, []))
                if item.status == "active"
            ),
            None,
        )
        if active is not None:
            if active.value == normalized:
                return
            if active.source_kind == "explicit_user" and source_kind != "explicit_user":
                return
            active.status = "superseded"
        slots.setdefault(name, []).append(
            MemorySlot(
                value=normalized,
                source_turn_id=source_turn_id,
                source_kind=source_kind,
                confidence=confidence,
                updated_at=now,
            )
        )

    @staticmethod
    def _supersede(slots: dict[str, list[MemorySlot]], name: str) -> None:
        for item in slots.get(name, []):
            if item.status == "active":
                item.status = "superseded"

    @staticmethod
    def _active_value(
        slots: dict[str, list[MemorySlot]],
        name: str,
    ) -> str | None:
        return next(
            (
                item.value
                for item in reversed(slots.get(name, []))
                if item.status == "active"
            ),
            None,
        )

    @staticmethod
    def _load_slots(raw: str) -> dict[str, list[MemorySlot]]:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return {}
            return {
                str(name): [
                    MemorySlot.model_validate(item)
                    for item in values
                    if isinstance(item, dict)
                ]
                for name, values in payload.items()
                if isinstance(values, list)
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            return {}

    @staticmethod
    def _dump_slots(slots: dict[str, list[MemorySlot]]) -> str:
        return json.dumps(
            {
                name: [item.model_dump(mode="json") for item in values]
                for name, values in slots.items()
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _pending(raw: str | None) -> PendingClarificationState | None:
        if not raw:
            return None
        try:
            return PendingClarificationState.model_validate_json(raw)
        except ValueError:
            return None
