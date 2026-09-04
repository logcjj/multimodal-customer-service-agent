from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class ConversationRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    owner_id: str = Field(index=True)
    title: str
    message_count: int = 0
    last_message_preview: str = ""
    last_route: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)


class ConversationTurnRecord(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "ordinal",
            name="uq_conversation_turn_ordinal",
        ),
    )

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    conversation_id: str = Field(index=True)
    ordinal: int
    request_id: str = Field(index=True, unique=True)
    user_text: str
    attachment_metadata_json: str = "[]"
    assistant_text: str = ""
    response_json: str | None = None
    status: str = "running"
    error_code: str | None = None
    initial_route: str | None = None
    final_route: str | None = None
    route_reason: str | None = None
    coverage_status: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class ConversationStateRecord(SQLModel, table=True):
    conversation_id: str = Field(primary_key=True)
    slots_json: str = "{}"
    rolling_summary: str = ""
    summary_through_ordinal: int = 0
    pending_clarification_json: str | None = None
    clarification_round: int = 0
    evidence_refs_json: str = "[]"
    memory_version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
