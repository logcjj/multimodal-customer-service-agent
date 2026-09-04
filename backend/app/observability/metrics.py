from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic


@dataclass
class _Aggregate:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, value: float | None) -> None:
        if value is None:
            return
        number = max(0.0, float(value))
        self.count += 1
        self.total += number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

    def snapshot(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "average": round(self.total / self.count, 3) if self.count else None,
            "minimum": round(self.minimum, 3) if self.minimum is not None else None,
            "maximum": round(self.maximum, 3) if self.maximum is not None else None,
        }


@dataclass
class _RequestSample:
    started_at: float
    first_event_at: float | None = None
    first_delta_at: float | None = None
    provider_latencies: list[float] = field(default_factory=list)
    retrieval_mode: str | None = None
    downgrade_count: int = 0
    completed: bool = False


class RequestMetricsTracker:
    def __init__(self, registry: "MetricsRegistry") -> None:
        self.registry = registry
        self.sample = _RequestSample(started_at=registry.clock())

    def observe_event(self, event: dict[str, object]) -> None:
        now = self.registry.clock()
        if self.sample.first_event_at is None:
            self.sample.first_event_at = now
        event_type = str(event.get("type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if event_type == "answer.delta" and self.sample.first_delta_at is None:
            self.sample.first_delta_at = now
        if event_type == "retrieval.completed":
            mode = payload.get("mode")
            if isinstance(mode, str) and mode:
                self.sample.retrieval_mode = mode
        if event_type == "generation.completed":
            latency = payload.get("provider_latency_ms")
            if isinstance(latency, int | float) and not isinstance(latency, bool):
                self.sample.provider_latencies.append(float(latency))
        if event_type in {"ocr.completed", "vlm.completed", "generation.completed"} and event.get("status") == "failed":
            self.sample.downgrade_count += 1

    def complete(self, *, success: bool) -> None:
        if self.sample.completed:
            return
        self.sample.completed = True
        self.registry._record(self.sample, success=success, finished_at=self.registry.clock())


class MetricsRegistry:
    def __init__(self, *, clock: Callable[[], float] = monotonic) -> None:
        self.clock = clock
        self._lock = Lock()
        self._request_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._downgrade_count = 0
        self._retrieval_modes: Counter[str] = Counter()
        self._total_latency = _Aggregate()
        self._first_event = _Aggregate()
        self._first_delta = _Aggregate()
        self._provider_latency = _Aggregate()

    def start_request(self) -> RequestMetricsTracker:
        with self._lock:
            self._request_count += 1
        return RequestMetricsTracker(self)

    def _record(self, sample: _RequestSample, *, success: bool, finished_at: float) -> None:
        with self._lock:
            if success:
                self._completed_count += 1
            else:
                self._failed_count += 1
            self._downgrade_count += sample.downgrade_count
            if sample.retrieval_mode:
                self._retrieval_modes[sample.retrieval_mode] += 1
            self._total_latency.add((finished_at - sample.started_at) * 1000)
            self._first_event.add(
                (sample.first_event_at - sample.started_at) * 1000
                if sample.first_event_at is not None
                else None
            )
            self._first_delta.add(
                (sample.first_delta_at - sample.started_at) * 1000
                if sample.first_delta_at is not None
                else None
            )
            for value in sample.provider_latencies:
                self._provider_latency.add(value)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "request_count": self._request_count,
                "completed_count": self._completed_count,
                "failed_count": self._failed_count,
                "downgrade_count": self._downgrade_count,
                "retrieval_modes": dict(sorted(self._retrieval_modes.items())),
                "total_latency_ms": self._total_latency.snapshot(),
                "time_to_first_event_ms": self._first_event.snapshot(),
                "time_to_first_answer_delta_ms": self._first_delta.snapshot(),
                "provider_latency_ms": self._provider_latency.snapshot(),
            }
