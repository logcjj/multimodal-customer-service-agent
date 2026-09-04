from __future__ import annotations

import re
from dataclasses import dataclass

from app.agents.verifier import VerifierAgent
from app.contracts.models import AgentResult, Claim, Evidence


@dataclass(frozen=True)
class VerifiedStreamResult:
    emitted_text: str
    withheld_text: str
    issue_codes: list[str]
    final_emitted: list[str]


class VerifiedSentenceBuffer:
    """Buffers provider tokens and only exposes deterministically supported units."""

    _BOUNDARY = re.compile(r"[。！？!?；;\n]")

    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence
        self._pending = ""
        self._emitted: list[str] = []
        self._withheld: list[str] = []
        self._issue_codes: list[str] = []
        self._verifier = VerifierAgent()

    def feed(self, delta: str) -> list[str]:
        self._pending += delta
        ready: list[str] = []
        while True:
            match = self._BOUNDARY.search(self._pending)
            if match is None:
                break
            end = match.end()
            unit, self._pending = self._pending[:end], self._pending[end:]
            ready.extend(self._check(unit))
        return ready

    def finish(self) -> VerifiedStreamResult:
        final_emitted = self._check(self._pending) if self._pending else []
        self._pending = ""
        return VerifiedStreamResult(
            emitted_text="".join(self._emitted),
            withheld_text="".join(self._withheld),
            issue_codes=list(dict.fromkeys(self._issue_codes)),
            final_emitted=final_emitted,
        )

    def _check(self, unit: str) -> list[str]:
        if not unit.strip():
            self._emitted.append(unit)
            return [unit]
        evidence_ids = [item.evidence_id for item in self.evidence]
        result = AgentResult(
            task_id="stream-buffer",
            agent_id="knowledge",
            status="completed",
            answer_fragment=unit,
            claims=[Claim(text=unit.strip(), evidence_ids=evidence_ids)],
            evidence=self.evidence,
            confidence=1.0,
        )
        report = self._verifier.verify([result], {"knowledge"})
        if report.passed:
            self._emitted.append(unit)
            return [unit]
        self._withheld.append(unit)
        self._issue_codes.extend(item.code for item in report.issues if item.severity == "error")
        return []
