from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlmodel import Field, SQLModel

from app.contracts.models import (
    Evidence,
    SessionEvidenceRef,
    SessionMemoryView,
    VisualContext,
)
from app.knowledge.product_router import ProductRouter
from app.storage.database import Database


class SessionMemoryRecord(SQLModel, table=True):
    session_id: str = Field(primary_key=True)
    user_id: str | None = Field(default=None, index=True)
    turn_count: int = 0
    last_question: str = ""
    products_json: str = "[]"
    model_codes_json: str = "[]"
    intent: str = "technical"
    answer_summary: str = ""
    evidence_refs_json: str = "[]"
    visual_context_json: str | None = None
    missing_information_json: str = "[]"
    risk_state: str = "unknown"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = Field(index=True)


class SessionMemoryStore:
    def __init__(
        self,
        database: Database,
        *,
        ttl_seconds: int,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.database = database
        self.ttl_seconds = ttl_seconds
        self.now = now or (lambda: datetime.now(UTC))
        self.product_router = ProductRouter()

    def save_turn(
        self,
        *,
        session_id: str,
        user_id: str | None,
        question: str,
        products: list[str],
        model_codes: list[str],
        intent: str,
        answer: str,
        evidence: list[Evidence],
        visual_context: VisualContext | None,
        missing_information: list[str],
        risk_state: str,
    ) -> SessionMemoryView:
        now = self._aware(self.now())
        with self.database.session() as session:
            record = session.get(SessionMemoryRecord, session_id)
            if record is None:
                record = SessionMemoryRecord(
                    session_id=session_id,
                    user_id=user_id,
                    expires_at=now + timedelta(seconds=self.ttl_seconds),
                    created_at=now,
                )
            elif record.user_id != user_id:
                raise PermissionError("session owner mismatch")
            previous_products = self._list(record.products_json)
            switched = bool(products and previous_products and set(products).isdisjoint(previous_products))
            resolved_products = products or ([] if switched else previous_products)
            previous_codes = self._list(record.model_codes_json)
            resolved_codes = model_codes or ([] if switched else previous_codes)
            refs = [
                SessionEvidenceRef(
                    evidence_id=item.evidence_id,
                    source_type=item.source_type,
                    title=item.title,
                    dataset_id=item.dataset_id,
                    document_id=item.document_id,
                    parent_id=item.parent_id or item.section_id,
                    image_chunk_ids=item.image_chunk_ids[:5],
                ).model_dump(mode="json")
                for item in evidence[:20]
            ]
            record.turn_count += 1
            record.last_question = question.strip()[:2000]
            record.products_json = self._dump(resolved_products[:10])
            record.model_codes_json = self._dump(resolved_codes[:20])
            record.intent = intent[:80]
            record.answer_summary = answer.strip()[:1200]
            record.evidence_refs_json = self._dump(refs)
            record.visual_context_json = (
                self._dump(visual_context.model_dump(mode="json")) if visual_context else None
            )
            record.missing_information_json = self._dump(missing_information[:10])
            record.risk_state = risk_state[:80]
            record.updated_at = now
            record.expires_at = now + timedelta(seconds=self.ttl_seconds)
            session.add(record)
            session.commit()
            session.refresh(record)
        return self._view(record)

    def load(self, session_id: str, user_id: str | None) -> SessionMemoryView | None:
        with self.database.session() as session:
            record = session.get(SessionMemoryRecord, session_id)
            if record is None or record.user_id != user_id:
                return None
            if self._aware(record.expires_at) <= self._aware(self.now()):
                session.delete(record)
                session.commit()
                return None
            return self._view(record)

    def load_relevant(
        self,
        session_id: str,
        user_id: str | None,
        question: str,
    ) -> SessionMemoryView | None:
        memory = self.load(session_id, user_id)
        if memory is None:
            return None
        current_products = set(self.product_router.route(question).products)
        if current_products and memory.products and current_products.isdisjoint(memory.products):
            return None
        return memory

    def delete(self, session_id: str, user_id: str | None) -> bool:
        with self.database.session() as session:
            record = session.get(SessionMemoryRecord, session_id)
            if record is None or record.user_id != user_id:
                return False
            session.delete(record)
            session.commit()
            return True

    @staticmethod
    def context_text(memory: SessionMemoryView) -> str:
        parts = [
            f"上一轮产品：{'、'.join(memory.products)}" if memory.products else "",
            f"上一轮型号或错误码：{'、'.join(memory.model_codes)}" if memory.model_codes else "",
            f"上一轮意图：{memory.intent}" if memory.intent else "",
            f"上一轮回答摘要：{memory.answer_summary}" if memory.answer_summary else "",
        ]
        return "\n".join(item for item in parts if item)[:2200]

    @staticmethod
    def _aware(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

    @staticmethod
    def _dump(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _list(raw: str) -> list[str]:
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return []
        return [str(item) for item in value] if isinstance(value, list) else []

    def _view(self, record: SessionMemoryRecord) -> SessionMemoryView:
        visual = None
        if record.visual_context_json:
            try:
                visual = VisualContext.model_validate_json(record.visual_context_json)
            except ValueError:
                visual = None
        try:
            refs = [SessionEvidenceRef.model_validate(item) for item in json.loads(record.evidence_refs_json)]
        except (TypeError, ValueError, json.JSONDecodeError):
            refs = []
        return SessionMemoryView(
            session_id=record.session_id,
            user_id=record.user_id,
            turn_count=record.turn_count,
            last_question=record.last_question,
            products=self._list(record.products_json),
            model_codes=self._list(record.model_codes_json),
            intent=record.intent,
            answer_summary=record.answer_summary,
            evidence_refs=refs,
            visual_context=visual,
            missing_information=self._list(record.missing_information_json),
            risk_state=record.risk_state,
            created_at=self._aware(record.created_at),
            updated_at=self._aware(record.updated_at),
            expires_at=self._aware(record.expires_at),
        )
