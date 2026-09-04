from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from threading import RLock
from uuid import uuid4

from sqlalchemy.exc import IntegrityError
from sqlmodel import col, select

from app.contracts.models import (
    AgentResponse,
    ConversationDetail,
    ConversationStateView,
    ConversationSummary,
    ConversationTurnView,
)
from app.conversations.models import (
    ConversationRecord,
    ConversationStateRecord,
    ConversationTurnRecord,
)
from app.storage.database import Database
from app.runtime.session_memory import SessionMemoryRecord
from app.observability.traces import sanitize_trace_for_persistence


class ConversationStore:
    _LOCK_STRIPE_COUNT = 64

    def __init__(
        self,
        database: Database,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.database = database
        self.now = now or (lambda: datetime.now(UTC))
        self._lock_stripes = tuple(
            RLock() for _ in range(self._LOCK_STRIPE_COUNT)
        )

    @property
    def lock_stripe_count(self) -> int:
        return len(self._lock_stripes)

    def create(
        self,
        owner_id: str,
        *,
        conversation_id: str | None = None,
        title: str | None = None,
    ) -> ConversationSummary:
        now = self.now()
        resolved_id = (conversation_id or str(uuid4())).strip()
        if not resolved_id:
            raise ValueError("conversation id cannot be empty")
        normalized_title = (title or "新对话").strip()
        if not normalized_title:
            raise ValueError("conversation title cannot be empty")
        if conversation_id is not None:
            with self.conversation_lock(resolved_id):
                return self._create_record(
                    resolved_id,
                    owner_id,
                    normalized_title,
                    now,
                )
        return self._create_record(
            resolved_id,
            owner_id,
            normalized_title,
            now,
        )

    def _create_record(
        self,
        conversation_id: str,
        owner_id: str,
        title: str,
        now: datetime,
    ) -> ConversationSummary:
        try:
            with self.database.session() as session:
                existing = session.get(ConversationRecord, conversation_id)
                if existing is not None:
                    if existing.owner_id != owner_id:
                        raise PermissionError("conversation owner mismatch")
                    return self._summary(existing)
                conversation = ConversationRecord(
                    id=conversation_id,
                    owner_id=owner_id,
                    title=title[:80],
                    created_at=now,
                    updated_at=now,
                )
                session.add(conversation)
                session.commit()
                session.refresh(conversation)
                return self._summary(conversation)
        except IntegrityError as exc:
            return self._read_create_conflict(
                conversation_id,
                owner_id,
                exc,
            )

    def _read_create_conflict(
        self,
        conversation_id: str,
        owner_id: str,
        conflict: IntegrityError,
    ) -> ConversationSummary:
        with self.database.session() as session:
            existing = session.get(ConversationRecord, conversation_id)
            if existing is None:
                raise conflict
            if existing.owner_id != owner_id:
                raise PermissionError("conversation owner mismatch") from conflict
            return self._summary(existing)

    def begin_turn(
        self,
        conversation_id: str,
        owner_id: str,
        request_id: str,
        user_text: str,
        attachment_metadata: list[dict[str, object]],
    ) -> ConversationTurnView:
        with self.conversation_lock(conversation_id):
            for attempt in range(3):
                try:
                    return self._begin_turn_locked(
                        conversation_id,
                        owner_id,
                        request_id,
                        user_text,
                        attachment_metadata,
                    )
                except IntegrityError:
                    if attempt == 2:
                        raise
            raise RuntimeError("unreachable conversation turn retry state")

    def _begin_turn_locked(
        self,
        conversation_id: str,
        owner_id: str,
        request_id: str,
        user_text: str,
        attachment_metadata: list[dict[str, object]],
    ) -> ConversationTurnView:
        now = self.now()
        with self.database.session() as session:
            existing = session.exec(
                select(ConversationTurnRecord).where(
                    ConversationTurnRecord.request_id == request_id
                )
            ).first()
            if existing is not None:
                existing_conversation = session.get(
                    ConversationRecord,
                    existing.conversation_id,
                )
                if (
                    existing.conversation_id != conversation_id
                    or existing_conversation is None
                    or existing_conversation.owner_id != owner_id
                ):
                    raise PermissionError("conversation owner mismatch")
                return self._turn_view(existing)

            conversation = session.exec(
                select(ConversationRecord)
                .where(ConversationRecord.id == conversation_id)
                .with_for_update()
            ).first()
            if conversation is None:
                conversation = ConversationRecord(
                    id=conversation_id,
                    owner_id=owner_id,
                    title=self._title(user_text),
                    created_at=now,
                    updated_at=now,
                )
            elif conversation.owner_id != owner_id:
                raise PermissionError("conversation owner mismatch")

            latest = session.exec(
                select(ConversationTurnRecord)
                .where(ConversationTurnRecord.conversation_id == conversation_id)
                .order_by(col(ConversationTurnRecord.ordinal).desc())
            ).first()
            ordinal = (latest.ordinal + 1) if latest is not None else 1
            turn = ConversationTurnRecord(
                conversation_id=conversation_id,
                ordinal=ordinal,
                request_id=request_id,
                user_text=user_text.strip(),
                attachment_metadata_json=self._dump(
                    self._safe_attachment_metadata(attachment_metadata)
                ),
                created_at=now,
            )
            conversation.message_count += 1
            conversation.last_message_preview = user_text.strip()[:160]
            conversation.updated_at = now
            session.add(conversation)
            session.add(turn)
            session.commit()
            session.refresh(turn)
            return self._turn_view(turn)

    def conversation_lock(self, conversation_id: str) -> RLock:
        digest = sha256(conversation_id.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % len(self._lock_stripes)
        return self._lock_stripes[index]

    def complete_turn(
        self,
        request_id: str,
        response: AgentResponse,
    ) -> ConversationTurnView:
        conversation_id = self._conversation_id_for_request(request_id)
        with self.conversation_lock(conversation_id):
            return self._complete_turn_locked(
                request_id,
                response,
                conversation_id,
            )

    def _complete_turn_locked(
        self,
        request_id: str,
        response: AgentResponse,
        conversation_id: str,
    ) -> ConversationTurnView:
        now = self.now()
        with self.database.session() as session:
            conversation = session.exec(
                select(ConversationRecord)
                .where(ConversationRecord.id == conversation_id)
                .with_for_update()
            ).first()
            if conversation is None:
                raise KeyError(f"unknown conversation: {conversation_id}")
            turn = session.exec(
                select(ConversationTurnRecord)
                .where(ConversationTurnRecord.request_id == request_id)
                .with_for_update()
            ).first()
            if turn is None:
                raise KeyError(f"unknown conversation request: {request_id}")
            if turn.conversation_id != conversation_id:
                raise RuntimeError("conversation turn mapping changed during completion")
            if turn.status == "completed":
                return self._turn_view(turn)

            routing = response.routing
            turn.assistant_text = response.answer
            turn.response_json = self._ledger_response(response).model_dump_json()
            turn.status = "completed"
            turn.error_code = None
            turn.initial_route = routing.initial_route if routing else None
            turn.final_route = routing.final_route if routing else response.route
            turn.route_reason = routing.route_reason if routing else None
            turn.coverage_status = (
                routing.coverage_status.value if routing else None
            )
            turn.completed_at = now
            conversation.message_count += 1
            conversation.last_message_preview = response.answer.strip()[:160]
            conversation.last_route = turn.final_route
            conversation.updated_at = now
            state = session.exec(
                select(ConversationStateRecord)
                .where(
                    ConversationStateRecord.conversation_id
                    == turn.conversation_id
                )
                .with_for_update()
            ).first()
            if state is None:
                state = ConversationStateRecord(
                    conversation_id=turn.conversation_id,
                )
            state.evidence_refs_json = self._dump(
                [
                    {
                        "evidence_id": item.evidence_id,
                        "source_type": item.source_type,
                        "title": item.title,
                        "dataset_id": item.dataset_id,
                        "document_id": item.document_id,
                        "file_id": item.file_id,
                        "document_version": item.document_version,
                        "parent_id": item.parent_id or item.section_id,
                        "page_start": item.page_start,
                        "page_end": item.page_end,
                        "locator_label": item.locator_label,
                        "image_chunk_ids": item.image_chunk_ids[:5],
                    }
                    for item in response.citations[:20]
                ]
            )
            state.updated_at = now
            session.add(turn)
            session.add(conversation)
            session.add(state)
            session.commit()
            session.refresh(turn)
            return self._turn_view(turn)

    def fail_turn(self, request_id: str, error_code: str) -> None:
        try:
            conversation_id = self._conversation_id_for_request(request_id)
        except KeyError:
            return
        with self.conversation_lock(conversation_id):
            self._fail_turn_locked(request_id, error_code, conversation_id)

    def _fail_turn_locked(
        self,
        request_id: str,
        error_code: str,
        conversation_id: str,
    ) -> None:
        now = self.now()
        with self.database.session() as session:
            conversation = session.exec(
                select(ConversationRecord)
                .where(ConversationRecord.id == conversation_id)
                .with_for_update()
            ).first()
            if conversation is None:
                return
            turn = session.exec(
                select(ConversationTurnRecord)
                .where(ConversationTurnRecord.request_id == request_id)
                .with_for_update()
            ).first()
            if turn is None or turn.status == "completed":
                return
            if turn.conversation_id != conversation_id:
                raise RuntimeError("conversation turn mapping changed during failure")
            turn.status = "failed"
            turn.error_code = error_code[:120]
            turn.completed_at = now
            conversation.updated_at = now
            session.add(conversation)
            session.add(turn)
            session.commit()

    def list(
        self,
        owner_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[ConversationSummary]:
        with self.database.session() as session:
            records = session.exec(
                select(ConversationRecord)
                .where(ConversationRecord.owner_id == owner_id)
                .order_by(
                    col(ConversationRecord.updated_at).desc(),
                    col(ConversationRecord.id).desc(),
                )
                .offset(offset)
                .limit(limit)
            ).all()
            return [self._summary(record) for record in records]

    def get(
        self,
        conversation_id: str,
        owner_id: str,
    ) -> ConversationDetail | None:
        with self.database.session() as session:
            conversation = session.get(ConversationRecord, conversation_id)
            if conversation is None or conversation.owner_id != owner_id:
                return None
            turns = session.exec(
                select(ConversationTurnRecord)
                .where(ConversationTurnRecord.conversation_id == conversation_id)
                .order_by(col(ConversationTurnRecord.ordinal))
            ).all()
            state = session.get(ConversationStateRecord, conversation_id)
            return ConversationDetail(
                **self._summary(conversation).model_dump(),
                turns=[self._turn_view(turn) for turn in turns],
                state=self._state_view(state) if state is not None else None,
            )

    def rename(
        self,
        conversation_id: str,
        owner_id: str,
        title: str,
    ) -> ConversationSummary | None:
        normalized = title.strip()
        if not normalized:
            raise ValueError("conversation title cannot be empty")
        with self.conversation_lock(conversation_id):
            with self.database.session() as session:
                conversation = session.exec(
                    select(ConversationRecord)
                    .where(ConversationRecord.id == conversation_id)
                    .with_for_update()
                ).first()
                if conversation is None or conversation.owner_id != owner_id:
                    return None
                conversation.title = normalized[:80]
                conversation.updated_at = self.now()
                session.add(conversation)
                session.commit()
                session.refresh(conversation)
                return self._summary(conversation)

    def delete(self, conversation_id: str, owner_id: str) -> bool:
        with self.conversation_lock(conversation_id):
            with self.database.session() as session:
                conversation = session.exec(
                    select(ConversationRecord)
                    .where(ConversationRecord.id == conversation_id)
                    .with_for_update()
                ).first()
                if conversation is None or conversation.owner_id != owner_id:
                    return False
                turns = session.exec(
                    select(ConversationTurnRecord)
                    .where(
                        ConversationTurnRecord.conversation_id == conversation_id
                    )
                    .with_for_update()
                ).all()
                for turn in turns:
                    session.delete(turn)
                state = session.exec(
                    select(ConversationStateRecord)
                    .where(
                        ConversationStateRecord.conversation_id == conversation_id
                    )
                    .with_for_update()
                ).first()
                if state is not None:
                    session.delete(state)
                legacy = session.exec(
                    select(SessionMemoryRecord)
                    .where(SessionMemoryRecord.session_id == conversation_id)
                    .with_for_update()
                ).first()
                if legacy is not None:
                    session.delete(legacy)
                session.delete(conversation)
                session.commit()
                return True

    def _conversation_id_for_request(self, request_id: str) -> str:
        with self.database.session() as session:
            conversation_id = session.exec(
                select(ConversationTurnRecord.conversation_id).where(
                    ConversationTurnRecord.request_id == request_id
                )
            ).first()
        if conversation_id is None:
            raise KeyError(f"unknown conversation request: {request_id}")
        return str(conversation_id)

    @staticmethod
    def _ledger_response(response: AgentResponse) -> AgentResponse:
        citations = [
            item.model_copy(
                update={
                    "text": "",
                    "score": None,
                    "score_breakdown": {},
                    "retrieval_stage": None,
                    "evidence_confidence": None,
                }
            )
            for item in response.citations[:20]
        ]
        return response.model_copy(
            update={
                "citations": citations,
                "trace": sanitize_trace_for_persistence(response.trace),
            }
        )

    @staticmethod
    def _title(question: str) -> str:
        normalized = " ".join(question.split())
        return normalized[:36] or "新对话"

    @staticmethod
    def _safe_attachment_metadata(
        items: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        allowed = ("name", "mime_type", "sha256", "size_bytes")
        return [
            {key: item[key] for key in allowed if key in item}
            for item in items[:3]
        ]

    @staticmethod
    def _dump(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _load_object(raw: str | None, fallback):
        if not raw:
            return fallback
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _aware_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @staticmethod
    def _summary(record: ConversationRecord) -> ConversationSummary:
        return ConversationSummary(
            id=record.id,
            owner_id=record.owner_id,
            title=record.title,
            message_count=record.message_count,
            last_message_preview=record.last_message_preview,
            last_route=record.last_route,
            created_at=ConversationStore._aware_utc(record.created_at),
            updated_at=ConversationStore._aware_utc(record.updated_at),
        )

    def _turn_view(self, record: ConversationTurnRecord) -> ConversationTurnView:
        response = None
        if record.response_json:
            try:
                response = AgentResponse.model_validate_json(record.response_json)
            except ValueError:
                response = None
        attachments = self._load_object(record.attachment_metadata_json, [])
        return ConversationTurnView(
            id=record.id,
            conversation_id=record.conversation_id,
            ordinal=record.ordinal,
            request_id=record.request_id,
            user_text=record.user_text,
            attachment_metadata=attachments if isinstance(attachments, list) else [],
            assistant_text=record.assistant_text,
            response=response,
            status=record.status,
            error_code=record.error_code,
            initial_route=record.initial_route,
            final_route=record.final_route,
            route_reason=record.route_reason,
            coverage_status=record.coverage_status,
            created_at=self._aware_utc(record.created_at),
            completed_at=self._aware_utc(record.completed_at),
        )

    def _state_view(self, record: ConversationStateRecord) -> ConversationStateView:
        slots = self._load_object(record.slots_json, {})
        pending = self._load_object(record.pending_clarification_json, None)
        return ConversationStateView(
            slots=slots if isinstance(slots, dict) else {},
            rolling_summary=record.rolling_summary,
            summary_through_ordinal=record.summary_through_ordinal,
            pending_clarification=pending if isinstance(pending, dict) else None,
            clarification_round=record.clarification_round,
            memory_version=record.memory_version,
            updated_at=self._aware_utc(record.updated_at),
        )
