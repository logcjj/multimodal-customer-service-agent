from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from time import perf_counter

from app.contracts.models import AgentResult, ModelKind
from app.models.llm_gateway import LLMGateway
from app.runtime.conversation_memory import ConversationMemoryService


class MemoryCuratorAgent:
    id = "memory-curator"

    def __init__(
        self,
        memory: ConversationMemoryService,
        llm_gateway: LLMGateway | None,
    ) -> None:
        self.memory = memory
        self.llm_gateway = llm_gateway
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="aka-memory-curator",
        )
        self._state_lock = Lock()
        self._failure_count = 0
        self._last_error: str | None = None

    @property
    def failure_count(self) -> int:
        with self._state_lock:
            return self._failure_count

    @property
    def last_error(self) -> str | None:
        with self._state_lock:
            return self._last_error

    def submit(
        self,
        conversation_id: str,
        owner_id: str,
    ) -> Future[AgentResult]:
        future = self._executor.submit(self.run, conversation_id, owner_id)
        future.add_done_callback(self._record_background_result)
        return future

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _record_background_result(self, future: Future[AgentResult]) -> None:
        error: str | None = None
        try:
            result = future.result()
        except Exception as exc:
            error = type(exc).__name__
        else:
            if result.status == "failed":
                error = result.recommended_next_action or "background-job-failed"
        if error is None:
            return
        with self._state_lock:
            self._failure_count += 1
            self._last_error = error[:160]

    def run(self, conversation_id: str, owner_id: str) -> AgentResult:
        started = perf_counter()
        detail = self.memory.conversations.get(conversation_id, owner_id)
        if detail is None:
            return AgentResult(
                task_id="memory-curator-1",
                agent_id=self.id,
                status="failed",
                confidence=0,
                recommended_next_action="conversation-not-found",
                latency_ms=round((perf_counter() - started) * 1000),
            )
        gateway = self.llm_gateway
        if gateway is None or not gateway.available(ModelKind.LLM) or not detail.turns:
            return AgentResult(
                task_id="memory-curator-1",
                agent_id=self.id,
                status="completed",
                confidence=0.7,
                recommended_next_action="structured-memory-only",
                latency_ms=round((perf_counter() - started) * 1000),
            )

        transcript = "\n".join(
            f"用户：{turn.user_text}\n助手：{turn.assistant_text}"
            for turn in detail.turns
        )[-8000:]
        output = gateway.generate(
            kind=ModelKind.LLM,
            system_prompt=(
                "你是客服会话 Memory Curator。把对话压缩成简短、可核对的事实摘要。"
                "只保留用户明确提供的产品、型号、错误码、现象、已尝试操作、售后诉求和未解决事项；"
                "不得把助手建议写成用户事实，不得补造内容，直接输出摘要。"
            ),
            user_prompt=transcript,
            temperature=0,
            max_tokens=500,
        )
        if output is None or not output.text.strip():
            return AgentResult(
                task_id="memory-curator-1",
                agent_id=self.id,
                status="completed",
                confidence=0.6,
                recommended_next_action="summary-deferred",
                latency_ms=round((perf_counter() - started) * 1000),
            )
        through = max(turn.ordinal for turn in detail.turns)
        self.memory.save_summary(
            conversation_id,
            owner_id,
            output.text,
            through_ordinal=through,
        )
        return AgentResult(
            task_id="memory-curator-1",
            agent_id=self.id,
            status="completed",
            confidence=0.82,
            latency_ms=round((perf_counter() - started) * 1000),
            llm_generated=True,
            model_used=output.model,
            recommended_next_action="summary-updated",
        )
