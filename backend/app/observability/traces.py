from __future__ import annotations

import json
import re
from collections.abc import Collection
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel, select

from app.contracts.models import AgentTrace
from app.storage.database import Database


class TraceRecord(SQLModel, table=True):
    request_id: str = SQLField(primary_key=True)
    owner_id: str = SQLField(default="__legacy_anonymous__", index=True)
    session_id: str = SQLField(index=True)
    route: str
    selected_agents_json: str
    steps_json: str
    spans_json: str = "[]"
    fallback_reason: str | None = None
    total_latency_ms: int
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))


class FeedbackRecord(SQLModel, table=True):
    id: str = SQLField(default_factory=lambda: str(uuid4()), primary_key=True)
    request_id: str = SQLField(index=True)
    rating: str
    category: str
    comment: str = ""
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))


class FeedbackCreate(BaseModel):
    request_id: str
    rating: str = Field(pattern="^(up|down)$")
    category: str = Field(min_length=1, max_length=80)
    comment: str = Field(default="", max_length=1000)


_LABELLED_IDENTIFIER = re.compile(
    r"(?P<label>订单(?:号|编号|标识)?|设备序列号|序列号|serial(?:\s+number)?|s\s*/\s*n|sn)"
    r"(?P<separator>\s*[:：#]?\s*)"
    r"(?P<value>(?=[A-Za-z0-9._/-]{4,})(?=[A-Za-z0-9._/-]*\d)"
    r"[A-Za-z0-9][A-Za-z0-9._/-]{3,})",
    flags=re.IGNORECASE,
)
_LONG_NUMBER = re.compile(r"(?<!\d)\d{8,}(?!\d)")


def _redaction_token(value: str) -> str:
    digest = sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"[已脱敏:{digest}]"


def _labelled_identifiers(value: object) -> set[str]:
    if isinstance(value, str):
        return {
            match.group("value")
            for match in _LABELLED_IDENTIFIER.finditer(value)
        }
    if isinstance(value, list):
        return {
            identifier
            for item in value
            for identifier in _labelled_identifiers(item)
        }
    if isinstance(value, dict):
        return {
            identifier
            for item in value.values()
            for identifier in _labelled_identifiers(item)
        }
    return set()


def redact_trace_text(
    value: str,
    *,
    limit: int = 1000,
    sensitive_identifiers: Collection[str] = (),
) -> str:
    duplicate_redacted = value
    for identifier in sorted(sensitive_identifiers, key=len, reverse=True):
        duplicate_redacted = re.sub(
            re.escape(identifier),
            _redaction_token(identifier),
            duplicate_redacted,
            flags=re.IGNORECASE,
        )
    labelled = _LABELLED_IDENTIFIER.sub(
        lambda match: (
            f"{match.group('label')}{match.group('separator')}"
            f"{_redaction_token(match.group('value'))}"
        ),
        duplicate_redacted,
    )
    redacted = _LONG_NUMBER.sub(
        lambda match: _redaction_token(match.group(0)),
        labelled,
    )
    return redacted[:limit]


def _sanitize_trace_value(
    value: object,
    sensitive_identifiers: Collection[str],
) -> object:
    if isinstance(value, str):
        return redact_trace_text(
            value,
            sensitive_identifiers=sensitive_identifiers,
        )
    if isinstance(value, list):
        return [
            _sanitize_trace_value(item, sensitive_identifiers)
            for item in value[:100]
        ]
    if isinstance(value, dict):
        return {
            str(key)[:120]: _sanitize_trace_value(item, sensitive_identifiers)
            for key, item in list(value.items())[:100]
        }
    return value


def sanitize_trace_for_persistence(trace: AgentTrace) -> AgentTrace:
    """Create a redacted ledger copy without changing the live response trace."""

    sensitive_identifiers = _labelled_identifiers(
        trace.model_dump(mode="python")
    )
    return trace.model_copy(
        update={
            "steps": [
                step.model_copy(
                    update={
                        "label": redact_trace_text(
                            step.label,
                            limit=200,
                            sensitive_identifiers=sensitive_identifiers,
                        ),
                        "summary": redact_trace_text(
                            step.summary,
                            limit=500,
                            sensitive_identifiers=sensitive_identifiers,
                        ),
                    }
                )
                for step in trace.steps
            ],
            "spans": [
                span.model_copy(
                    update={
                        "input_summary": redact_trace_text(
                            span.input_summary,
                            limit=500,
                            sensitive_identifiers=sensitive_identifiers,
                        ),
                        "output_summary": redact_trace_text(
                            span.output_summary,
                            limit=500,
                            sensitive_identifiers=sensitive_identifiers,
                        ),
                        "attributes": _sanitize_trace_value(
                            span.attributes,
                            sensitive_identifiers,
                        ),
                    }
                )
                for span in trace.spans
            ],
            "fallback_reason": (
                redact_trace_text(
                    trace.fallback_reason,
                    limit=500,
                    sensitive_identifiers=sensitive_identifiers,
                )
                if trace.fallback_reason
                else None
            ),
        }
    )


class TraceStore:
    LEGACY_ANONYMOUS_OWNER = "__legacy_anonymous__"

    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, trace: AgentTrace, *, owner_id: str | None = None) -> None:
        resolved_owner = (owner_id or self.LEGACY_ANONYMOUS_OWNER).strip()
        persisted_trace = sanitize_trace_for_persistence(trace)
        record = TraceRecord(
            request_id=persisted_trace.request_id,
            owner_id=resolved_owner or self.LEGACY_ANONYMOUS_OWNER,
            session_id=persisted_trace.session_id,
            route=persisted_trace.route,
            selected_agents_json=json.dumps(persisted_trace.selected_agents, ensure_ascii=False),
            steps_json=json.dumps([item.model_dump(mode="json") for item in persisted_trace.steps], ensure_ascii=False),
            spans_json=json.dumps([item.model_dump(mode="json") for item in persisted_trace.spans], ensure_ascii=False),
            fallback_reason=persisted_trace.fallback_reason,
            total_latency_ms=persisted_trace.total_latency_ms,
            created_at=persisted_trace.created_at,
        )
        with self.database.session() as session:
            session.add(record)
            session.commit()

    def list(self, owner_id: str) -> list[AgentTrace]:
        self._reject_reserved_owner(owner_id)
        with self.database.session() as session:
            records = session.exec(
                select(TraceRecord)
                .where(TraceRecord.owner_id == owner_id)
                .order_by(TraceRecord.created_at.desc())
            ).all()
            return [self._to_contract(item) for item in records]

    def get(self, request_id: str, owner_id: str) -> AgentTrace:
        self._reject_reserved_owner(owner_id)
        with self.database.session() as session:
            record = session.get(TraceRecord, request_id)
            if record is None or record.owner_id != owner_id:
                raise HTTPException(status_code=404, detail="Trace 不存在")
            return self._to_contract(record)

    @classmethod
    def _reject_reserved_owner(cls, owner_id: str) -> None:
        if owner_id.strip() == cls.LEGACY_ANONYMOUS_OWNER:
            raise HTTPException(status_code=422, detail="owner_id 不可使用保留值")

    def add_feedback(self, payload: FeedbackCreate) -> dict[str, object]:
        with self.database.session() as session:
            if session.get(TraceRecord, payload.request_id) is None:
                raise HTTPException(status_code=404, detail="Trace 不存在")
            record = FeedbackRecord(**payload.model_dump())
            session.add(record)
            session.commit()
            session.refresh(record)
        return {
            "id": record.id,
            "status": "queued-for-offline-review",
            "knowledge_updated": False,
        }

    @staticmethod
    def _to_contract(record: TraceRecord) -> AgentTrace:
        return AgentTrace(
            request_id=record.request_id,
            session_id=record.session_id,
            route=record.route,
            selected_agents=json.loads(record.selected_agents_json),
            steps=json.loads(record.steps_json),
            spans=json.loads(record.spans_json or "[]"),
            fallback_reason=record.fallback_reason,
            total_latency_ms=record.total_latency_ms,
            created_at=record.created_at,
        )
