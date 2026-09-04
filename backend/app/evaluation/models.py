from __future__ import annotations

import json
from datetime import UTC, datetime

from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField
from sqlmodel import SQLModel


class EvalCaseRecord(SQLModel, table=True):
    id: str = SQLField(primary_key=True)
    question: str
    dataset_ids_json: str = "[]"
    target_parent_ids_json: str = "[]"
    reference_answer: str = ""
    required_facts_json: str = "[]"
    forbidden_facts_json: str = "[]"
    image_required: bool = False
    locked: bool = True
    source: str = "manual"
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))


class EvalRunRecord(SQLModel, table=True):
    id: str = SQLField(primary_key=True)
    candidate_version: str
    case_ids_json: str = "[]"
    metrics_json: str = "{}"
    details_json: str = "[]"
    passed: bool = False
    status: str = "completed"
    created_at: datetime = SQLField(default_factory=lambda: datetime.now(UTC))
    approved_at: datetime | None = None


class EvalCaseCreate(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    dataset_ids: list[str] = Field(min_length=1)
    target_parent_ids: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    required_facts: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    image_required: bool = False
    locked: bool = True
    source: str = "manual"


class EvalCaseView(EvalCaseCreate):
    id: str
    created_at: datetime


class EvalRunCreate(BaseModel):
    candidate_version: str = Field(min_length=1, max_length=120)
    case_ids: list[str] = Field(min_length=1)


class EvalRunView(BaseModel):
    id: str
    candidate_version: str
    case_ids: list[str]
    metrics: dict[str, float]
    details: list[dict[str, object]]
    passed: bool
    status: str
    created_at: datetime
    approved_at: datetime | None

