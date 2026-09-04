from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

from app.contracts.models import AgentRequest, AgentResult, TaskPlan


@dataclass
class RuntimeState:
    request: AgentRequest
    started_at: float = field(default_factory=monotonic)
    plan: TaskPlan | None = None
    results: list[AgentResult] = field(default_factory=list)
    fallback_reason: str | None = None
    tool_calls: int = 0
    retries: int = 0

    @property
    def elapsed_ms(self) -> int:
        return max(0, round((monotonic() - self.started_at) * 1000))

    @property
    def remaining_ms(self) -> int:
        return max(0, self.request.deadline_ms - self.elapsed_ms)

    @property
    def is_expired(self) -> bool:
        return self.remaining_ms == 0

