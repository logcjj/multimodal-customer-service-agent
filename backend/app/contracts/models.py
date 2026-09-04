from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class ModelKind(StrEnum):
    LLM = "llm"
    EMBEDDING = "embedding"
    VLM = "vlm"
    RERANK = "rerank"
    ASR = "asr"
    TTS = "tts"
    OCR = "ocr"


class CoverageStatus(StrEnum):
    COVERED = "covered"
    CLARIFIABLE = "clarifiable"
    UNSAFE_UNCOVERED = "unsafe_uncovered"
    GENERAL_ALLOWED = "general_allowed"
    GENERAL_UNAVAILABLE = "general_unavailable"


class ClarificationRequest(BaseModel):
    case_id: Annotated[str, Field(min_length=1, max_length=120)]
    field: Annotated[str, Field(min_length=1, max_length=80)]
    question: Annotated[str, Field(min_length=1, max_length=1000)]
    round: Annotated[int, Field(ge=1, le=3)]
    max_rounds: Annotated[int, Field(ge=1, le=3)] = 3
    accepted_input_types: list[str] = Field(default_factory=lambda: ["text"], max_length=3)


class RoutingIntent(BaseModel):
    initial_route: Literal[
        "technical_candidate",
        "customer_service_candidate",
        "mixed_candidate",
        "general_candidate",
    ]
    risk_level: Literal["low", "medium", "high"]
    requires_knowledge_check: bool
    reason_code: Annotated[str, Field(min_length=1, max_length=80)]
    llm_used: bool = False
    model_used: str | None = None
    classification_source: Literal["deterministic", "model", "model_fallback"] = (
        "deterministic"
    )


class RoutingDecision(BaseModel):
    initial_route: str
    final_route: Literal[
        "technical_knowledge",
        "customer_service",
        "mixed",
        "evidence_clarification",
        "general_llm",
        "safe_handoff",
        "general_unavailable",
    ]
    route_label: Annotated[str, Field(min_length=1, max_length=80)]
    route_reason: Annotated[str, Field(min_length=1, max_length=500)]
    coverage_status: CoverageStatus
    knowledge_covered: bool
    risk_level: Literal["low", "medium", "high"]
    clarification: ClarificationRequest | None = None


class ModelConfigurationCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    provider: Annotated[str, Field(min_length=1, max_length=80)]
    name: Annotated[str, Field(min_length=1, max_length=160)]
    kind: ModelKind
    base_url: Annotated[str, Field(min_length=1, max_length=500)]
    api_key: SecretStr | None = Field(default=None, exclude=True)
    capabilities: list[str] = Field(default_factory=list, max_length=20)
    enabled: bool = True


class ModelConfigurationUpdate(BaseModel):
    kind: ModelKind


class ModelConfiguration(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    provider: str
    name: str
    kind: ModelKind
    base_url: str
    secret_configured: bool = False
    secret_hint: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    enabled: bool = True
    is_default: bool = False
    health: Literal["untested", "healthy", "unhealthy"] = "untested"
    latency_ms: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=8000)]
    images: list[str] = Field(default_factory=list, max_length=3)
    session_id: str | None = Field(default=None, max_length=120)
    user_id: str | None = Field(default=None, max_length=120)
    deadline_ms: Annotated[int, Field(gt=0, le=120_000)] = 20_000


class SubTask(BaseModel):
    task_id: str
    title: str
    route: Literal["technical", "customer_service", "multimodal", "verification"]
    depends_on: list[str] = Field(default_factory=list)
    assigned_agent: str


class TaskPlan(BaseModel):
    route: Literal["technical", "customer_service", "mixed"]
    subtasks: list[SubTask] = Field(min_length=1, max_length=8)
    selected_agents: list[str] = Field(min_length=1, max_length=6)
    max_tool_calls: Annotated[int, Field(gt=0, le=8)] = 4
    max_retries: Annotated[int, Field(ge=0, le=1)] = 1


class Evidence(BaseModel):
    evidence_id: str
    source_type: Literal["manual", "image", "ocr", "policy", "vision", "tool", "legacy"]
    title: str
    text: str
    product: str | None = None
    dataset_id: str | None = None
    document_id: str | None = None
    file_id: str | None = None
    document_name: str | None = None
    document_mime_type: str | None = None
    document_version: str | None = None
    section_id: str | None = None
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    image_chunk_ids: list[str] = Field(default_factory=list)
    chapter_title: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    locator_label: str | None = None
    asset_ids: list[str] = Field(default_factory=list)
    score: float | None = Field(default=None, ge=0)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    retrieval_stage: str | None = None
    evidence_confidence: float | None = Field(default=None, ge=0, le=1)


class Claim(BaseModel):
    text: Annotated[str, Field(min_length=1)]
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Annotated[float, Field(ge=0, le=1)] = 1.0
    risk_level: Literal["low", "medium", "high"] = "low"


class VisualContext(BaseModel):
    """Structured, evidence-safe facts extracted from user supplied images."""

    model_config = ConfigDict(frozen=True)

    image_hashes: list[str] = Field(default_factory=list)
    ocr_text: str = ""
    detected_codes: list[str] = Field(default_factory=list)
    detected_numbers: list[str] = Field(default_factory=list)
    detected_product: str | None = None
    detected_components: list[str] = Field(default_factory=list)
    visible_objects: list[str] = Field(default_factory=list)
    visual_summary: str = ""
    provider_status: dict[str, str] = Field(default_factory=dict)
    field_provenance: dict[str, Literal["ocr", "vlm"]] = Field(
        default_factory=dict
    )
    confidence: Annotated[float, Field(ge=0, le=1)] = 0.0


class SessionEvidenceRef(BaseModel):
    evidence_id: str
    source_type: str
    title: str
    dataset_id: str | None = None
    document_id: str | None = None
    parent_id: str | None = None
    image_chunk_ids: list[str] = Field(default_factory=list)


class SessionMemoryView(BaseModel):
    session_id: str
    user_id: str | None = None
    turn_count: int = Field(ge=0)
    last_question: str
    products: list[str] = Field(default_factory=list)
    model_codes: list[str] = Field(default_factory=list)
    intent: str
    answer_summary: str
    evidence_refs: list[SessionEvidenceRef] = Field(default_factory=list)
    visual_context: VisualContext | None = None
    missing_information: list[str] = Field(default_factory=list)
    risk_state: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class RetrievalTraceSnapshot(BaseModel):
    query: str
    mode: str
    result_count: int = Field(ge=0)
    stage_counts: dict[str, int] = Field(default_factory=dict)
    rejected_reason: str | None = None


class AgentResult(BaseModel):
    task_id: str
    agent_id: str
    status: Literal["completed", "needs_input", "failed", "handoff"]
    answer_fragment: str = ""
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    confidence: Annotated[float, Field(ge=0, le=1)]
    missing_information: list[str] = Field(default_factory=list)
    recommended_next_action: str | None = None
    latency_ms: int = Field(default=0, ge=0)
    llm_generated: bool = False
    model_used: str | None = None
    search_query: str | None = None
    query_rewrite_model: str | None = None
    routed_products: list[str] = Field(default_factory=list)
    product_route_reason: str | None = None
    visual_context: VisualContext | None = None
    retrieval_trace: RetrievalTraceSnapshot | None = None

    @model_validator(mode="after")
    def validate_claim_evidence(self) -> "AgentResult":
        evidence_ids = {item.evidence_id for item in self.evidence}
        for claim in self.claims:
            if not claim.evidence_ids:
                raise ValueError("factual claims require at least one evidence id")
            missing = set(claim.evidence_ids) - evidence_ids
            if missing:
                raise ValueError(f"claim references unknown evidence ids: {sorted(missing)}")
        return self


class VerificationIssue(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"]
    evidence_ids: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    passed: bool
    action: Literal["accept", "revise", "clarify", "fallback", "handoff"]
    verified_claims: list[Claim] = Field(default_factory=list)
    issues: list[VerificationIssue] = Field(default_factory=list)
    confidence: Annotated[float, Field(ge=0, le=1)]


class AgentStep(BaseModel):
    agent_id: str
    label: str
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    latency_ms: int = Field(default=0, ge=0)
    summary: str = ""


class TraceSpan(BaseModel):
    span_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    status: Literal["completed", "failed", "skipped"] = "completed"
    latency_ms: int = Field(default=0, ge=0)
    input_summary: str = ""
    output_summary: str = ""
    attributes: dict[str, object] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    request_id: str
    session_id: str
    route: str
    selected_agents: list[str]
    steps: list[AgentStep]
    spans: list[TraceSpan] = Field(default_factory=list)
    fallback_reason: str | None = None
    total_latency_ms: int = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    route: str
    citations: list[Evidence] = Field(default_factory=list)
    assets: list[str] = Field(default_factory=list)
    verification: VerificationReport
    trace: AgentTrace
    used_legacy: bool = False
    routing: RoutingDecision | None = None
    timestamp: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))


class ConversationCreate(BaseModel):
    id: str | None = Field(default=None, max_length=120)
    title: str | None = Field(default=None, max_length=80)


class ConversationUpdate(BaseModel):
    title: Annotated[str, Field(min_length=1, max_length=80)]


class ConversationSummary(BaseModel):
    id: str
    owner_id: str
    title: str
    message_count: int = Field(ge=0)
    last_message_preview: str
    last_route: str | None = None
    created_at: datetime
    updated_at: datetime


class ConversationTurnView(BaseModel):
    id: str
    conversation_id: str
    ordinal: int = Field(ge=1)
    request_id: str
    user_text: str
    attachment_metadata: list[dict[str, object]] = Field(default_factory=list)
    assistant_text: str
    response: AgentResponse | None = None
    status: Literal["running", "completed", "failed"]
    error_code: str | None = None
    initial_route: str | None = None
    final_route: str | None = None
    route_reason: str | None = None
    coverage_status: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class ConversationStateView(BaseModel):
    slots: dict[str, object] = Field(default_factory=dict)
    rolling_summary: str = ""
    summary_through_ordinal: int = 0
    pending_clarification: dict[str, object] | None = None
    clarification_round: int = 0
    memory_version: int = 1
    updated_at: datetime | None = None


class ConversationDetail(ConversationSummary):
    turns: list[ConversationTurnView] = Field(default_factory=list)
    state: ConversationStateView | None = None
