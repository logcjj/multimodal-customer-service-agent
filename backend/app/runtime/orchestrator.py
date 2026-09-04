from __future__ import annotations

import re
from hashlib import sha256
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import monotonic
from uuid import uuid4

from app.agents.customer_service import CustomerServiceAgent
from app.agents.evidence_gap import EvidenceGapAgent
from app.agents.general import GeneralAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.memory_curator import MemoryCuratorAgent
from app.agents.multimodal import MultimodalAgent
from app.agents.verifier import VerifierAgent
from app.compatibility.legacy_champion import FallbackPolicy, LegacyChampionAdapter
from app.config.runtime import RuntimeSettings
from app.contracts.models import (
    AgentRequest,
    AgentResponse,
    AgentStep,
    AgentTrace,
    CoverageStatus,
    RoutingDecision,
    RoutingIntent,
    TraceSpan,
    VerificationReport,
)
from app.knowledge.retrieval import HybridRetriever
from app.models.llm_gateway import LLMGateway
from app.conversations.store import ConversationStore
from app.multimodal.visual_context import (
    visual_context_for_answer,
    visual_search_text,
)
from app.observability.traces import TraceStore
from app.runtime.planner import Planner
from app.runtime.conversation_memory import ConversationContext, ConversationMemoryService
from app.runtime.dynamic_routing import CoverageAssessment, IntentRouter, KnowledgeCoverageGate
from app.runtime.error_codes import extract_normalized_error_codes
from app.runtime.session_memory import SessionMemoryStore
from app.runtime.state import RuntimeState


_PIC_PLACEHOLDER_PATTERN = re.compile(r"<\s*PIC\s*>", re.IGNORECASE)
_NUMBERED_PROCEDURE_STEP_PATTERN = re.compile(
    r"^\s*\d{1,2}(?:[.)、]|(?=\s))\s*.+$", re.MULTILINE
)
_MAX_INLINE_MANUAL_ASSETS = 3


class Orchestrator:
    def __init__(
        self,
        retriever: HybridRetriever,
        trace_store: TraceStore,
        legacy: LegacyChampionAdapter,
        rollout_mode: str = "champion_guarded",
        llm_gateway: LLMGateway | None = None,
        settings: RuntimeSettings | None = None,
        session_memory: SessionMemoryStore | None = None,
        conversations: ConversationStore | None = None,
        conversation_memory: ConversationMemoryService | None = None,
        intent_router: IntentRouter | None = None,
        coverage_gate: KnowledgeCoverageGate | None = None,
        general_agent: GeneralAgent | None = None,
        evidence_gap_agent: EvidenceGapAgent | None = None,
        memory_curator: MemoryCuratorAgent | None = None,
    ) -> None:
        self.settings = settings or RuntimeSettings.from_env()
        self.planner = Planner()
        self.llm_gateway = llm_gateway
        self.multimodal = MultimodalAgent(llm_gateway, settings=self.settings)
        self.knowledge = KnowledgeAgent(retriever, llm_gateway, settings=self.settings)
        self.customer_service = CustomerServiceAgent(llm_gateway)
        self.verifier = VerifierAgent()
        self.session_memory = session_memory
        self.conversations = conversations
        self.conversation_memory = conversation_memory
        self.intent_router = intent_router or IntentRouter(llm_gateway)
        self.coverage_gate = coverage_gate or KnowledgeCoverageGate(
            max_rounds=self.settings.max_clarification_rounds,
        )
        self.general = general_agent or GeneralAgent(llm_gateway)
        self.evidence_gap = evidence_gap_agent or EvidenceGapAgent(
            max_rounds=self.settings.max_clarification_rounds,
        )
        self.memory_curator = memory_curator
        self.trace_store = trace_store
        self.legacy = legacy
        if rollout_mode not in {"champion_guarded", "agent_first", "legacy_only"}:
            raise ValueError(f"unsupported rollout mode: {rollout_mode}")
        self.rollout_mode = rollout_mode

    def run(
        self,
        request: AgentRequest,
        event_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> AgentResponse:
        request_id = str(uuid4())
        session_id = request.session_id or f"session-{uuid4()}"
        conversation_context: ConversationContext | None = None
        preclassified_intent: RoutingIntent | None = None
        persisted_turn = False
        owner_id = request.user_id
        if (
            owner_id
            and self.conversations is not None
            and self.conversation_memory is not None
            and self.settings.is_enabled(self.settings.conversation_history)
        ):
            try:
                turn = self.conversations.begin_turn(
                    session_id,
                    owner_id,
                    request_id,
                    request.question,
                    self._attachment_metadata(request.images),
                )
                persisted_turn = True
                if self.settings.is_enabled(self.settings.layered_memory):
                    conversation_context = self.conversation_memory.load_context(
                        session_id,
                        owner_id,
                        request.question,
                    )
                    pending = (
                        conversation_context.pending_clarification
                        if conversation_context is not None
                        else None
                    )
                    if pending is not None:
                        candidate_intent = None
                        if not self.conversation_memory.pending_reply_has_expected_shape(
                            pending,
                            request.question,
                        ):
                            candidate_intent = self.intent_router.classify(
                                request,
                                context_text="",
                            )
                        if self.conversation_memory.should_end_pending_for_topic_switch(
                            pending,
                            request.question,
                            has_images=bool(request.images),
                            candidate_intent=candidate_intent,
                        ):
                            self.conversation_memory.clear_pending_clarification(
                                session_id,
                                owner_id,
                            )
                            preclassified_intent = candidate_intent
                    self.conversation_memory.record_user_turn(
                        session_id,
                        owner_id,
                        turn.id,
                        request.question,
                    )
                    conversation_context = self.conversation_memory.load_context(
                        session_id,
                        owner_id,
                        request.question,
                    )
            except Exception:
                if persisted_turn:
                    try:
                        self.conversations.fail_turn(
                            request_id,
                            "layered_memory_persistence_failed",
                        )
                    except Exception:
                        self._emit(
                            event_sink,
                            event_type="persistence.failed",
                            agent_id="memory-curator",
                            status="failed",
                            label="失败状态未写入对话历史",
                            summary="回答将继续按无记忆模式执行",
                        )
                self._emit(
                    event_sink,
                    event_type="persistence.failed",
                    agent_id="memory-curator",
                    status="failed",
                    label="对话历史未保存",
                    summary="会话身份或持久化校验失败",
                )
                persisted_turn = False
                conversation_context = None
                preclassified_intent = None

        try:
            response = self._execute(
                request,
                request_id=request_id,
                session_id=session_id,
                conversation_context=conversation_context,
                preclassified_intent=preclassified_intent,
                event_sink=event_sink,
            )
        except Exception as exc:
            if persisted_turn and self.conversations is not None:
                try:
                    self.conversations.fail_turn(request_id, type(exc).__name__)
                except Exception:
                    self._emit(
                        event_sink,
                        event_type="persistence.failed",
                        agent_id="memory-curator",
                        status="failed",
                        label="失败状态未写入对话历史",
                        summary="原始执行异常保持不变",
                    )
            raise

        if persisted_turn and self.conversations is not None:
            layered_memory_enabled = self.settings.is_enabled(
                self.settings.layered_memory
            )
            try:
                self.conversations.complete_turn(request_id, response)
            except Exception:
                try:
                    self.conversations.fail_turn(
                        request_id,
                        "complete_turn_persistence_failed",
                    )
                except Exception:
                    self._emit(
                        event_sink,
                        event_type="persistence.failed",
                        agent_id=(
                            "memory-curator"
                            if layered_memory_enabled
                            else "orchestrator"
                        ),
                        status="failed",
                        label="失败状态未写入对话历史",
                        summary="回答本身仍然有效",
                    )
                self._emit(
                    event_sink,
                    event_type="persistence.failed",
                    agent_id=(
                        "memory-curator"
                        if layered_memory_enabled
                        else "orchestrator"
                    ),
                    status="failed",
                    label="对话回答未写入历史",
                    summary="回答本身仍然有效",
                )
            else:
                self._emit(
                    event_sink,
                    event_type=(
                        "memory.updated"
                        if layered_memory_enabled
                        else "conversation.saved"
                    ),
                    agent_id=(
                        "memory-curator"
                        if layered_memory_enabled
                        else "orchestrator"
                    ),
                    status="completed",
                    label=(
                        "更新分层会话记忆"
                        if layered_memory_enabled
                        else "保存对话账本"
                    ),
                    summary=(
                        "完整账本与结构化状态已持久化"
                        if layered_memory_enabled
                        else "仅保存用户与助手消息"
                    ),
                    payload={"conversation_id": session_id},
                )
                if (
                    layered_memory_enabled
                    and self.memory_curator is not None
                    and owner_id
                ):
                    try:
                        self.memory_curator.submit(session_id, owner_id)
                    except Exception:
                        self._emit(
                            event_sink,
                            event_type="persistence.failed",
                            agent_id="memory-curator",
                            status="failed",
                            label="滚动摘要未提交",
                            summary="完整对话账本已保存",
                        )
        return response

    def _execute(
        self,
        request: AgentRequest,
        *,
        request_id: str,
        session_id: str,
        conversation_context: ConversationContext | None,
        event_sink: Callable[[dict[str, object]], None] | None,
        preclassified_intent: RoutingIntent | None = None,
    ) -> AgentResponse:
        state = RuntimeState(request=request)
        effective_request = request.model_copy(update={"session_id": session_id})
        router_request = effective_request
        answer_conversation_context = (
            conversation_context
            if self.settings.affects_answer(self.settings.layered_memory)
            else None
        )
        layered_context_text = ""
        if answer_conversation_context is not None:
            layered_context_text = answer_conversation_context.context_text()
            structured_context_text = (
                answer_conversation_context.structured_context_text()
            )
            standalone_question = request.question
            if answer_conversation_context.pending_clarification is not None:
                standalone_question = (
                    f"{answer_conversation_context.pending_clarification.original_question}\n"
                    f"用户本轮补充：{request.question}"
                )
                if structured_context_text:
                    standalone_question += f"\n{structured_context_text}"
                router_request = router_request.model_copy(
                    update={"question": standalone_question}
                )
            elif structured_context_text:
                standalone_question = (
                    f"{request.question}\n{structured_context_text}"
                )
            effective_request = effective_request.model_copy(
                update={"question": standalone_question}
            )
        loaded_memory = None
        if (
            self.session_memory is not None
            and self.settings.is_enabled(self.settings.session_memory)
            and request.session_id
        ):
            loaded_memory = self.session_memory.load_relevant(
                request.session_id,
                request.user_id,
                request.question,
            )
            if loaded_memory is not None:
                memory_text = self.session_memory.context_text(loaded_memory)
                if memory_text and self.settings.affects_answer(self.settings.session_memory):
                    effective_request = effective_request.model_copy(
                        update={"question": f"{effective_request.question}\n{memory_text}"}
                    )
                self._emit(
                    event_sink,
                    event_type="session.loaded",
                    agent_id="orchestrator",
                    status="completed",
                    label="恢复相关多轮会话上下文",
                    summary=f"第 {loaded_memory.turn_count + 1} 轮",
                    payload={
                        "turn_count": loaded_memory.turn_count,
                        "products": loaded_memory.products,
                        "model_codes": loaded_memory.model_codes,
                        "risk_state": loaded_memory.risk_state,
                    },
                )
        self._emit(
            event_sink,
            event_type="run.started",
            agent_id="orchestrator",
            status="running",
            label="接收请求并建立运行上下文",
            payload={"request_id": request_id, "session_id": session_id},
        )
        intent: RoutingIntent | None = None
        dynamic_routing_active = self.settings.is_enabled(self.settings.dynamic_routing)
        dynamic_routing_affects_answer = (
            self.settings.affects_answer(self.settings.dynamic_routing)
            and self.rollout_mode == "agent_first"
        )
        if dynamic_routing_active:
            intent = (
                preclassified_intent
                or self.intent_router.classify(
                    router_request,
                    context_text=layered_context_text,
                )
            )
            self._emit(
                event_sink,
                event_type="route.detected",
                agent_id="router",
                status="completed",
                label="识别用户意图与风险边界",
                summary=self._intent_label(intent.initial_route),
                payload={
                    "initial_route": intent.initial_route,
                    "risk_level": intent.risk_level,
                    "reason_code": intent.reason_code,
                    "llm_used": intent.llm_used,
                    "model_used": intent.model_used,
                    "classification_source": intent.classification_source,
                },
            )
            if intent.classification_source == "model_fallback":
                self._emit(
                    event_sink,
                    event_type="router.fallback",
                    agent_id="router",
                    status="completed",
                    label="路由模型不可用，采用确定性安全规则",
                    summary=intent.reason_code,
                    payload={
                        "reason_code": intent.reason_code,
                        "model_used": intent.model_used,
                    },
                )
            if (
                intent.initial_route == "general_candidate"
                and dynamic_routing_affects_answer
                and self.settings.affects_answer(self.settings.general_agent)
            ):
                general_coverage = self.coverage_gate.evaluate(
                    intent=intent,
                    question=request.question,
                    results=[],
                    active_slots=self._active_slots(answer_conversation_context),
                    clarification_round=0,
                )
                if general_coverage.status != CoverageStatus.GENERAL_ALLOWED:
                    return self._build_pre_domain_handoff(
                        request_id=request_id,
                        session_id=session_id,
                        owner_id=request.user_id,
                        state=state,
                        intent=intent,
                        coverage=general_coverage,
                        event_sink=event_sink,
                    )
                return self._execute_general(
                    request=router_request,
                    original_request=request,
                    request_id=request_id,
                    session_id=session_id,
                    intent=intent,
                    context_text=layered_context_text,
                    has_persistent_conversation=conversation_context is not None,
                    state=state,
                    event_sink=event_sink,
                )

        route_override = (
            {
                "technical_candidate": "technical",
                "customer_service_candidate": "customer_service",
                "mixed_candidate": "mixed",
            }.get(intent.initial_route)
            if intent is not None and dynamic_routing_affects_answer
            else None
        )
        state.plan = self.planner.create_plan(
            effective_request,
            route_override=route_override,
        )
        if intent is not None:
            state.plan.selected_agents.insert(1, "router")
        self._emit(
            event_sink,
            event_type="plan.completed",
            agent_id="orchestrator",
            status="completed",
            label="完成任务拆分与动态组队",
            summary=f"{len(state.plan.subtasks)} 个子任务 · {state.plan.route}",
            payload={
                "route": state.plan.route,
                "selected_agents": state.plan.selected_agents,
                "subtasks": [item.model_dump(mode="json") for item in state.plan.subtasks],
                "llm_configured": bool(self.llm_gateway and self.llm_gateway.available()),
                "llm_model": self.llm_gateway.model_name() if self.llm_gateway else None,
            },
        )
        steps = [
            AgentStep(
                agent_id="orchestrator",
                label="理解问题并组建专业智能体团队",
                status="completed",
                summary=f"路线：{state.plan.route}",
            )
        ]
        if intent is not None:
            steps.append(
                AgentStep(
                    agent_id="router",
                    label="识别意图、领域和风险等级",
                    status="completed",
                    summary=self._intent_label(intent.initial_route),
                )
            )

        domain_request = effective_request
        visual_context = None
        answer_visual_context = None
        ocr_affects_answer = self.settings.affects_answer(
            self.settings.ocr_pipeline
        )
        visual_affects_answer = False
        if "multimodal" in state.plan.selected_agents:
            self._emit_agent_started(event_sink, "multimodal", "分析用户图片与可见文字")
            result = self.multimodal.run(request, event_sink)
            state.results.append(result)
            visual_context = result.visual_context
            answer_visual_context = (
                visual_context_for_answer(
                    visual_context,
                    include_ocr=ocr_affects_answer,
                )
                if visual_context is not None
                else None
            )
            visual_affects_answer = bool(
                answer_visual_context is not None
                and visual_search_text(answer_visual_context)
            )
            steps.append(self._step_from_result(result, "提取图片中的可见信息"))
            self._emit_agent_completed(event_sink, result, "完成图片结构化观察")
            enrichment = (
                visual_search_text(
                    answer_visual_context,
                    include_ocr=True,
                )
                if answer_visual_context is not None and visual_affects_answer
                else ""
            )
            if enrichment:
                domain_request = request.model_copy(
                    update={
                        "session_id": session_id,
                        "question": f"{effective_request.question}\n图片可见信息：{enrichment}",
                    }
                )
            if (
                conversation_context is not None
                and self.conversation_memory is not None
                and request.user_id
                and answer_visual_context is not None
                and visual_affects_answer
            ):
                self.conversation_memory.record_user_turn(
                    session_id,
                    request.user_id,
                    request_id,
                    request.question,
                    answer_visual_context,
                )
                refreshed_context = self.conversation_memory.load_context(
                    session_id,
                    request.user_id,
                    request.question,
                )
                if refreshed_context is not None:
                    conversation_context = refreshed_context
                    if self.settings.affects_answer(self.settings.layered_memory):
                        answer_conversation_context = refreshed_context

        expected_domain_agents: set[str] = set()
        domain_jobs = []
        if "knowledge" in state.plan.selected_agents:
            expected_domain_agents.add("knowledge")
            domain_jobs.append(
                (
                    "knowledge",
                    "检索说明书、聚合 Parent Evidence 并生成答案",
                    "检索说明书并生成技术结论",
                    "完成技术证据与回答",
                    lambda: self.knowledge.run(domain_request, event_sink),
                )
            )

        if "customer-service" in state.plan.selected_agents:
            expected_domain_agents.add("customer-service")
            domain_jobs.append(
                (
                    "customer-service",
                    "核对售后政策、订单条件与风险边界",
                    "核对售后政策和业务条件",
                    "完成客服政策建议",
                    lambda: self.customer_service.run(domain_request),
                )
            )

        for agent_id, start_label, _, _, _ in domain_jobs:
            if agent_id == "knowledge":
                self._emit(
                    event_sink,
                    event_type="retrieval.started",
                    agent_id="knowledge",
                    status="running",
                    label="执行 BM25、Dense、RRF、Rerank 与 Parent 聚合",
                )
            self._emit_agent_started(event_sink, agent_id, start_label)

        if len(domain_jobs) == 1:
            _, _, step_label, completion_label, execute = domain_jobs[0]
            result = execute()
            state.results.append(result)
            steps.append(self._step_from_result(result, step_label))
            self._emit_agent_completed(event_sink, result, completion_label)
            if result.agent_id == "knowledge":
                self._emit_retrieval_completed(event_sink, result)
        elif domain_jobs:
            with ThreadPoolExecutor(max_workers=len(domain_jobs), thread_name_prefix="aka-agent") as executor:
                futures = {
                    executor.submit(execute): (step_label, completion_label)
                    for _, _, step_label, completion_label, execute in domain_jobs
                }
                for future in as_completed(futures):
                    step_label, completion_label = futures[future]
                    result = future.result()
                    state.results.append(result)
                    steps.append(self._step_from_result(result, step_label))
                    self._emit_agent_completed(event_sink, result, completion_label)
                    if result.agent_id == "knowledge":
                        self._emit_retrieval_completed(event_sink, result)

        coverage: CoverageAssessment | None = None
        routing_decision: RoutingDecision | None = None
        route_resolution_deferred = False
        if intent is not None and dynamic_routing_affects_answer:
            active_slots = self._active_slots(answer_conversation_context)
            pending = (
                answer_conversation_context.pending_clarification
                if answer_conversation_context is not None
                else None
            )
            clarification_round = pending.round if pending is not None else 0
            coverage = self.coverage_gate.evaluate(
                intent=intent,
                question=domain_request.question,
                results=state.results,
                active_slots=active_slots,
                clarification_round=clarification_round,
            )
            self._emit(
                event_sink,
                event_type="knowledge.coverage",
                agent_id="knowledge-coverage",
                status="completed",
                label="评估知识证据覆盖度",
                summary=coverage.reason,
                payload=coverage.model_dump(mode="json"),
            )

            if coverage.status == CoverageStatus.CLARIFIABLE and not (
                self.legacy.available and conversation_context is None
            ):
                gap_result, clarification = self.evidence_gap.run(
                    missing_fields=coverage.missing_fields,
                    round_number=clarification_round + 1,
                    case_id=pending.case_id if pending is not None else None,
                )
                state.results.append(gap_result)
                state.plan.selected_agents.append("evidence-gap")
                if conversation_context is not None:
                    state.plan.selected_agents.append("memory-curator")
                steps.append(
                    self._step_from_result(gap_result, "逐项收集缺失证据")
                )
                routing_decision = self._routing_decision(
                    intent,
                    coverage,
                    clarification=clarification,
                )
                if (
                    self.conversation_memory is not None
                    and request.user_id
                    and conversation_context is not None
                ):
                    self.conversation_memory.set_pending_clarification(
                        session_id,
                        request.user_id,
                        clarification,
                        original_question=(
                            pending.original_question
                            if pending is not None
                            else request.question
                        ),
                    )
                return self._build_gate_response(
                    request_id=request_id,
                    session_id=session_id,
                    owner_id=request.user_id,
                    state=state,
                    answer=gap_result.answer_fragment,
                    verification=VerificationReport(
                        passed=False,
                        action="clarify",
                        confidence=gap_result.confidence,
                    ),
                    intent=intent,
                    routing=routing_decision,
                    steps=steps,
                    coverage=coverage,
                    loaded_memory=loaded_memory,
                    event_sink=event_sink,
                )
            if (
                coverage.status == CoverageStatus.CLARIFIABLE
                and self.legacy.available
                and conversation_context is None
            ):
                route_resolution_deferred = True

            if coverage.status == CoverageStatus.UNSAFE_UNCOVERED:
                routing_decision = self._routing_decision(intent, coverage)
                if not (
                    self.legacy.available
                    and intent.risk_level != "high"
                ):
                    return self._build_gate_response(
                        request_id=request_id,
                        session_id=session_id,
                        owner_id=request.user_id,
                        state=state,
                        answer=self._safe_handoff_answer(intent.risk_level),
                        verification=VerificationReport(
                            passed=False,
                            action="handoff",
                            confidence=0.2,
                        ),
                        intent=intent,
                        routing=routing_decision,
                        steps=steps,
                        coverage=coverage,
                        loaded_memory=loaded_memory,
                        event_sink=event_sink,
                    )
                route_resolution_deferred = True

            routing_decision = self._routing_decision(intent, coverage)
            if (
                self.conversation_memory is not None
                and request.user_id
                and conversation_context is not None
                and pending is not None
            ):
                self.conversation_memory.clear_pending_clarification(
                    session_id,
                    request.user_id,
                )
            if not route_resolution_deferred:
                self._emit_route_resolved(event_sink, routing_decision)

        self._emit(
            event_sink,
            event_type="verification.started",
            agent_id="verifier",
            status="running",
            label="验证 Claim、Evidence、图片与问题覆盖",
        )
        verification, shadow_verification = self._verify_results(
            state.results,
            expected_domain_agents,
        )
        steps.append(
            AgentStep(
                agent_id="verifier",
                label="验证事实、图片、政策和覆盖率",
                status="completed" if verification.passed else "failed",
                summary="验证通过" if verification.passed else "需要回退或澄清",
            )
        )
        self._emit(
            event_sink,
            event_type="verification.completed",
            agent_id="verifier",
            status="completed" if verification.passed else "failed",
            label="验证完成",
            summary="通过" if verification.passed else "需要回退或澄清",
            payload={
                "passed": verification.passed,
                "confidence": verification.confidence,
                "action": verification.action,
                "enhanced_mode": self.settings.enhanced_verifier,
                "shadow_issue_codes": (
                    [item.code for item in shadow_verification.issues]
                    if shadow_verification is not None
                    else []
                ),
            },
        )

        revision_attempted = False
        revision_succeeded = False
        evidence_fallback_applied = False
        knowledge_result = next((item for item in state.results if item.agent_id == "knowledge"), None)
        initial_knowledge_answer = knowledge_result.answer_fragment if knowledge_result else ""

        if (
            not verification.passed
            and knowledge_result is not None
            and knowledge_result.evidence
            and coverage is not None
            and coverage.status == CoverageStatus.COVERED
            and intent is not None
            and intent.initial_route == "technical_candidate"
        ):
            evidence_result = self.knowledge.fallback_to_primary_evidence(knowledge_result)
            candidate_results = [
                evidence_result if item is knowledge_result else item
                for item in state.results
            ]
            evidence_verification, evidence_shadow = self._verify_results(
                candidate_results,
                expected_domain_agents,
            )
            if evidence_verification.passed:
                state.results = candidate_results
                knowledge_result = evidence_result
                verification = evidence_verification
                shadow_verification = evidence_shadow
                evidence_fallback_applied = True
                steps.append(
                    AgentStep(
                        agent_id="knowledge",
                        label="采用最高相关手册证据作为安全答案",
                        status="completed",
                        summary="跳过重复模型修订与 Legacy 回答",
                    )
                )
                self._emit(
                    event_sink,
                    event_type="answer.evidence_fallback",
                    agent_id="knowledge",
                    status="completed",
                    label="核验未通过，采用手册证据原文",
                    summary=knowledge_result.evidence[0].title,
                    payload={
                        "evidence_id": knowledge_result.evidence[0].evidence_id,
                        "avoided_model_calls": 2,
                    },
                )
        if (
            not verification.passed
            and knowledge_result is not None
            and knowledge_result.evidence
            and self.llm_gateway is not None
            and self.llm_gateway.available()
            and state.remaining_ms > 2_000
        ):
            revision_attempted = True
            self._emit(
                event_sink,
                event_type="answer.revision.started",
                agent_id="knowledge",
                status="running",
                label="根据 Verifier 问题修订答案",
                summary=f"{len(verification.issues)} 个核验问题",
            )
            revised = self.knowledge.revise(
                domain_request,
                knowledge_result,
                [item.message for item in verification.issues],
            )
            state.results = [revised if item is knowledge_result else item for item in state.results]
            verification, shadow_verification = self._verify_results(
                state.results,
                expected_domain_agents,
            )
            revision_succeeded = verification.passed
            steps.extend(
                [
                    AgentStep(
                        agent_id="knowledge",
                        label="依据核验意见修订证据答案",
                        status="completed",
                        latency_ms=max(0, revised.latency_ms - knowledge_result.latency_ms),
                        summary="已生成受证据约束的修订答案",
                    ),
                    AgentStep(
                        agent_id="verifier",
                        label="复核修订后的 Claim 与 Evidence",
                        status="completed" if verification.passed else "failed",
                        summary="复核通过" if verification.passed else "复核仍未通过",
                    ),
                ]
            )
            self._emit(
                event_sink,
                event_type="answer.revision.completed",
                agent_id="knowledge",
                status="completed" if verification.passed else "failed",
                label="答案修订与复核完成",
                summary="复核通过" if verification.passed else "仍有未支持事实",
                payload={
                    "passed": verification.passed,
                    "model_used": revised.model_used,
                    "issue_codes": [item.code for item in verification.issues],
                },
            )

        confidence = verification.confidence
        fallback_requested = FallbackPolicy.should_fallback(
            confidence=confidence,
            verification_passed=verification.passed,
            remaining_ms=state.remaining_ms,
        )
        legacy_reason: str | None = None
        if self.rollout_mode == "champion_guarded" and self.legacy.available:
            legacy_reason = "守护链路模式"
        elif self.rollout_mode == "legacy_only" and self.legacy.available:
            legacy_reason = "仅守护链路模式"
        elif fallback_requested and self.legacy.available:
            legacy_reason = "新链路证据不足"

        citations = [
            evidence
            for item in state.results
            for evidence in item.evidence
            if item.agent_id != "multimodal"
            or evidence.source_type == "vision"
            or (evidence.source_type == "ocr" and ocr_affects_answer)
        ]
        assets = list(
            dict.fromkeys(
                asset.strip()
                for item in state.results
                if item.agent_id != "multimodal" or visual_affects_answer
                for asset in item.asset_ids
                if asset.strip()
            )
        )
        for evidence in citations:
            for asset in evidence.asset_ids:
                normalized_asset = asset.strip()
                if normalized_asset and normalized_asset not in assets:
                    assets.append(normalized_asset)
        assets = assets[:5]
        use_legacy = False
        if legacy_reason:
            self._emit(
                event_sink,
                event_type="guard.started",
                agent_id="legacy-champion",
                status="running",
                label="运行守护链路与答案对照",
                summary=legacy_reason,
            )
        legacy_answer = self.legacy.answer(request.question, request.images) if legacy_reason else ""
        legacy_invocation = self.legacy.last_invocation if legacy_reason else None
        if (
            legacy_answer
            and self.rollout_mode != "legacy_only"
            and not self._legacy_answer_is_acceptable(
                question=request.question,
                route=state.plan.route,
                answer=legacy_answer,
                has_user_images=bool(request.images),
                has_visual_evidence=any(
                    item.source_type in {"vision", "ocr"}
                    for item in citations
                ),
            )
        ):
            steps.append(
                AgentStep(
                    agent_id="legacy-champion",
                    label="检查冻结守护链路输出",
                    status="skipped",
                    summary="低信息通用回复未通过守护门禁",
                )
            )
            legacy_answer = ""
            self._emit(
                event_sink,
                event_type="guard.completed",
                agent_id="legacy-champion",
                status="skipped",
                label="守护链路输出未通过低信息门禁",
                summary="采用已验证的新链路答案",
            )
        if legacy_answer.strip():
            use_legacy = True
            answer = legacy_answer
            state.fallback_reason = legacy_reason
            steps.append(
                AgentStep(
                    agent_id="legacy-champion",
                    label="调用冻结守护链路",
                    status="completed",
                    summary=(
                        f"{legacy_reason or '保留当前高分行为'} · "
                        f"{legacy_invocation.model_used}"
                        if legacy_invocation is not None and legacy_invocation.model_used
                        else legacy_reason or "保留当前高分行为"
                    ),
                )
            )
            verification = VerificationReport(
                passed=True,
                action="accept",
                verified_claims=[],
                issues=verification.issues,
                confidence=self._legacy_confidence(
                    previous=confidence,
                    citations=citations,
                    answer=legacy_answer,
                ),
            )
            self._emit(
                event_sink,
                event_type="guard.completed",
                agent_id="legacy-champion",
                status="completed",
                label="守护链路完成",
                summary=legacy_reason or "采用守护链路答案",
            )
            routing_decision = self._legacy_routing_decision(
                intent=intent,
                plan_route=state.plan.route,
                prior_routing=routing_decision,
                reason=legacy_reason or "守护链路接管最终答案",
            )
            self._emit_route_resolved(event_sink, routing_decision)
        elif verification.passed:
            answer = self._compose_answer(state.plan.route, state.results)
        elif intent is not None and intent.risk_level == "high":
            answer = self._safe_handoff_answer("high")
        else:
            answer = "当前证据不足以给出可靠结论。请补充产品型号、具体故障现象或相关图片后再试。"

        manual_assets = list(
            dict.fromkeys(
                asset.strip()
                for item in state.results
                if item.agent_id == "knowledge"
                for asset in item.asset_ids
                if asset.strip()
            )
        )
        for evidence in citations:
            if evidence.source_type not in {"manual", "image"}:
                continue
            for asset in evidence.asset_ids:
                normalized_asset = asset.strip()
                if normalized_asset and normalized_asset not in manual_assets:
                    manual_assets.append(normalized_asset)
        manual_assets = manual_assets[:5]
        manual_evidence_has_pictures = any(
            item.agent_id == "knowledge"
            and any(
                evidence.asset_ids or _PIC_PLACEHOLDER_PATTERN.search(evidence.text)
                for evidence in item.evidence
            )
            for item in state.results
        )
        non_manual_assets = [asset for asset in assets if asset not in manual_assets]
        answer, manual_assets = self._align_manual_assets(
            answer,
            manual_assets,
            inject_missing_placeholders=manual_evidence_has_pictures,
        )
        assets = [*manual_assets, *non_manual_assets]

        if (
            route_resolution_deferred
            and not use_legacy
            and routing_decision is not None
        ):
            self._emit_route_resolved(event_sink, routing_decision)

        if event_sink is not None and initial_knowledge_answer and answer != initial_knowledge_answer:
            self._emit(
                event_sink,
                event_type="answer.revised",
                agent_id="verifier" if revision_attempted else "orchestrator",
                status="completed",
                label="以最终核验答案替换临时正文",
                summary=(
                    "采用手册证据原文"
                    if evidence_fallback_applied
                    else "Verifier 修订"
                    if revision_attempted
                    else "最终门禁调整"
                ),
                payload={"answer": answer},
            )

        session_saved = False
        if self.session_memory is not None and self.settings.is_enabled(self.settings.session_memory):
            knowledge_for_memory = next(
                (item for item in state.results if item.agent_id == "knowledge"),
                None,
            )
            products = list(knowledge_for_memory.routed_products) if knowledge_for_memory else []
            memory_visual_context = (
                answer_visual_context if visual_affects_answer else None
            )
            if (
                not products
                and memory_visual_context
                and memory_visual_context.detected_product
            ):
                products = [memory_visual_context.detected_product]
            model_codes = list(
                dict.fromkeys(
                    [
                        *re.findall(
                            r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]+\b",
                            request.question,
                        ),
                        *(
                            memory_visual_context.detected_codes
                            if memory_visual_context
                            else []
                        ),
                    ]
                )
            )
            missing_information = list(
                dict.fromkeys(
                    value
                    for result in state.results
                    for value in result.missing_information
                )
            )
            try:
                saved = self.session_memory.save_turn(
                    session_id=session_id,
                    user_id=request.user_id,
                    question=request.question,
                    products=products,
                    model_codes=model_codes,
                    intent=state.plan.route,
                    answer=answer,
                    evidence=citations,
                    visual_context=memory_visual_context,
                    missing_information=missing_information,
                    risk_state="verified" if verification.passed else "needs_review",
                )
                session_saved = True
                self._emit(
                    event_sink,
                    event_type="session.saved",
                    agent_id="orchestrator",
                    status="completed",
                    label="保存结构化会话记忆",
                    summary=f"已保存第 {saved.turn_count} 轮",
                    payload={"turn_count": saved.turn_count, "expires_at": saved.expires_at.isoformat()},
                )
            except (PermissionError, ValueError):
                self._emit(
                    event_sink,
                    event_type="session.saved",
                    agent_id="orchestrator",
                    status="failed",
                    label="会话记忆未保存",
                    summary="会话身份或数据校验失败",
                )

        knowledge_result = next(
            (item for item in state.results if item.agent_id == "knowledge"),
            None,
        )
        retrieval_trace = (
            knowledge_result.retrieval_trace
            if knowledge_result is not None
            else None
        )
        spans = [
            TraceSpan(
                name="query_understanding",
                output_summary=f"路线：{state.plan.route}；选择 {len(state.plan.selected_agents)} 个 Agent",
                attributes={"selected_agents": state.plan.selected_agents},
            )
        ]
        if intent is not None:
            spans.append(
                TraceSpan(
                    name="intent_routing",
                    output_summary=self._intent_label(intent.initial_route),
                    attributes={
                        "initial_route": intent.initial_route,
                        "risk_level": intent.risk_level,
                        "reason_code": intent.reason_code,
                        "llm_used": intent.llm_used,
                        "model_used": intent.model_used,
                        "classification_source": intent.classification_source,
                    },
                )
            )
            if intent.classification_source == "model_fallback":
                spans.append(self._router_fallback_span(intent))
        if coverage is not None:
            spans.append(
                TraceSpan(
                    name="knowledge_coverage",
                    status=(
                        "completed"
                        if coverage.status == CoverageStatus.COVERED
                        else "failed"
                    ),
                    output_summary=coverage.reason,
                    attributes=coverage.model_dump(mode="json"),
                )
            )
        if legacy_invocation is not None:
            spans.append(
                TraceSpan(
                    name="legacy_champion",
                    status="completed" if use_legacy else "skipped",
                    output_summary=(
                        legacy_invocation.model_used
                        if legacy_invocation.llm_used and legacy_invocation.model_used
                        else "确定性守护链路"
                    ),
                    attributes={
                        "llm_used": legacy_invocation.llm_used,
                        "model_used": legacy_invocation.model_used,
                        "fallback_reason": legacy_invocation.fallback_reason,
                        "answer_adopted": use_legacy,
                    },
                )
            )
        if visual_context is not None:
            spans.append(
                TraceSpan(
                    name="visual_context",
                    status=(
                        "completed"
                        if any(value == "ok" for value in visual_context.provider_status.values())
                        else "skipped"
                    ),
                    input_summary=f"{len(request.images)} 张用户图片",
                    output_summary=(
                        f"OCR {visual_context.provider_status.get('ocr', 'unavailable')}；"
                        f"VLM {visual_context.provider_status.get('vlm', 'unavailable')}"
                    ),
                    attributes={
                        "provider_status": visual_context.provider_status,
                        "confidence": visual_context.confidence,
                        "mode": self.settings.ocr_pipeline,
                        "detected_code_count": len(visual_context.detected_codes),
                        "detected_component_count": len(visual_context.detected_components),
                    },
                )
            )
        if loaded_memory is not None or session_saved:
            spans.append(
                TraceSpan(
                    name="session_memory",
                    status="completed",
                    input_summary=(
                        f"恢复 {loaded_memory.turn_count} 轮" if loaded_memory else "新会话"
                    ),
                    output_summary="结构化上下文已保存" if session_saved else "只读恢复",
                    attributes={
                        "loaded": loaded_memory is not None,
                        "saved": session_saved,
                        "prior_evidence_count": len(loaded_memory.evidence_refs) if loaded_memory else 0,
                    },
                )
            )
        if retrieval_trace is not None:
            spans.extend(
                [
                    TraceSpan(
                        name="product_routing",
                        status="completed" if knowledge_result and knowledge_result.routed_products else "skipped",
                        input_summary=request.question[:200],
                        output_summary=(
                            "、".join(knowledge_result.routed_products)
                            if knowledge_result and knowledge_result.routed_products
                            else "保持全库检索"
                        ),
                        attributes={
                            "products": knowledge_result.routed_products if knowledge_result else [],
                            "reason": knowledge_result.product_route_reason if knowledge_result else None,
                        },
                    ),
                    TraceSpan(
                        name="query_rewrite",
                        status="completed" if knowledge_result and knowledge_result.query_rewrite_model else "skipped",
                        input_summary=request.question[:200],
                        output_summary=(
                            knowledge_result.search_query[:300]
                            if knowledge_result and knowledge_result.search_query
                            else request.question[:200]
                        ),
                        attributes={"model": knowledge_result.query_rewrite_model if knowledge_result else None},
                    ),
                    TraceSpan(
                        name="lexical_retrieval",
                        output_summary=(
                            f"候选 {retrieval_trace.stage_counts.get('lexical', 0)} 条"
                        ),
                        attributes={
                            "candidate_count": retrieval_trace.stage_counts.get(
                                "lexical", 0
                            )
                        },
                    ),
                    TraceSpan(
                        name="dense_retrieval",
                        status=(
                            "completed"
                            if retrieval_trace.stage_counts.get("dense", 0)
                            else "skipped"
                        ),
                        output_summary=(
                            f"候选 {retrieval_trace.stage_counts.get('dense', 0)} 条"
                        ),
                        attributes={"mode": retrieval_trace.mode},
                    ),
                    TraceSpan(
                        name="rrf_fusion",
                        output_summary=(
                            f"融合 {retrieval_trace.stage_counts.get('rrf', 0)} 条"
                        ),
                    ),
                    TraceSpan(
                        name="child_rerank",
                        status=(
                            "completed"
                            if retrieval_trace.stage_counts.get("rerank", 0)
                            else "skipped"
                        ),
                        output_summary=(
                            f"精排 {retrieval_trace.stage_counts.get('rerank', 0)} 条"
                        ),
                    ),
                    TraceSpan(
                        name="parent_aggregation",
                        output_summary=(
                            f"返回 {retrieval_trace.result_count} 个 Parent"
                        ),
                        attributes={
                            "rejected_reason": retrieval_trace.rejected_reason,
                            "query": retrieval_trace.query,
                        },
                    ),
                ]
            )
        spans.append(
            TraceSpan(
                name="answer_revision",
                status=("completed" if revision_succeeded else "failed") if revision_attempted else "skipped",
                output_summary=(
                    "Verifier 复核通过"
                    if revision_succeeded
                    else "未触发修订" if not revision_attempted else "修订后仍未通过"
                ),
            )
        )
        spans.append(
            TraceSpan(
                name="claim_verification",
                status="completed" if verification.passed else "failed",
                output_summary="验证通过" if verification.passed else "验证失败",
                attributes={
                    "issue_codes": [item.code for item in verification.issues],
                    "enhanced_mode": self.settings.enhanced_verifier,
                    "shadow_issue_codes": (
                        [item.code for item in shadow_verification.issues]
                        if shadow_verification is not None
                        else []
                    ),
                },
            )
        )
        if conversation_context is not None:
            if "memory-curator" not in state.plan.selected_agents:
                state.plan.selected_agents.append("memory-curator")
            spans.append(
                TraceSpan(
                    name="layered_memory",
                    status="completed",
                    input_summary=(
                        f"保留 {len(conversation_context.prompt.included_ordinals)} 个完整轮次"
                    ),
                    output_summary="结构化 Slots 与完整账本已更新",
                    attributes={
                        "estimated_tokens": conversation_context.prompt.estimated_tokens,
                        "has_pending_clarification": (
                            conversation_context.pending_clarification is not None
                        ),
                    },
                )
            )
        trace = AgentTrace(
            request_id=request_id,
            session_id=session_id,
            route=(
                routing_decision.final_route
                if routing_decision is not None
                else state.plan.route
            ),
            selected_agents=state.plan.selected_agents,
            steps=steps,
            spans=spans,
            fallback_reason=state.fallback_reason,
            total_latency_ms=max(1, round((monotonic() - state.started_at) * 1000)),
        )
        self.trace_store.save(trace, owner_id=request.user_id)
        return AgentResponse(
            request_id=request_id,
            session_id=session_id,
            answer=answer,
            route=(
                routing_decision.final_route
                if routing_decision is not None
                else state.plan.route
            ),
            citations=citations,
            assets=assets,
            verification=verification,
            trace=trace,
            used_legacy=use_legacy,
            routing=routing_decision,
        )

    def _build_pre_domain_handoff(
        self,
        *,
        request_id: str,
        session_id: str,
        owner_id: str | None,
        state: RuntimeState,
        intent: RoutingIntent,
        coverage: CoverageAssessment,
        event_sink: Callable[[dict[str, object]], None] | None,
    ) -> AgentResponse:
        self._emit(
            event_sink,
            event_type="knowledge.coverage",
            agent_id="knowledge-coverage",
            status="completed",
            label="评估知识证据覆盖度",
            summary=coverage.reason,
            payload=coverage.model_dump(mode="json"),
        )
        routing = self._routing_decision(intent, coverage)
        self._emit_route_resolved(event_sink, routing)
        answer = self._safe_handoff_answer("high")
        self._emit(
            event_sink,
            event_type="answer.delta",
            agent_id="orchestrator",
            status="running",
            label="输出安全交接建议",
            payload={"delta": answer},
        )
        verification = VerificationReport(
            passed=False,
            action="handoff",
            confidence=0.2,
        )
        spans = [
            TraceSpan(
                name="intent_routing",
                output_summary=self._intent_label(intent.initial_route),
                attributes={
                    "initial_route": intent.initial_route,
                    "risk_level": intent.risk_level,
                    "reason_code": intent.reason_code,
                    "llm_used": intent.llm_used,
                    "model_used": intent.model_used,
                    "classification_source": intent.classification_source,
                },
            ),
            TraceSpan(
                name="knowledge_coverage",
                status="failed",
                output_summary=coverage.reason,
                attributes=coverage.model_dump(mode="json"),
            ),
        ]
        if intent.classification_source == "model_fallback":
            spans.insert(1, self._router_fallback_span(intent))
        trace = AgentTrace(
            request_id=request_id,
            session_id=session_id,
            route=routing.final_route,
            selected_agents=["orchestrator", "router"],
            steps=[
                AgentStep(
                    agent_id="orchestrator",
                    label="建立运行上下文",
                    status="completed",
                    summary="通用路由安全门禁",
                ),
                AgentStep(
                    agent_id="router",
                    label="阻止越界请求进入通用模型",
                    status="completed",
                    summary=coverage.reason,
                ),
            ],
            spans=spans,
            total_latency_ms=max(1, round((monotonic() - state.started_at) * 1000)),
        )
        self.trace_store.save(trace, owner_id=owner_id)
        return AgentResponse(
            request_id=request_id,
            session_id=session_id,
            answer=answer,
            route=routing.final_route,
            citations=[],
            assets=[],
            verification=verification,
            trace=trace,
            routing=routing,
        )

    def _execute_general(
        self,
        *,
        request: AgentRequest,
        original_request: AgentRequest,
        request_id: str,
        session_id: str,
        intent: RoutingIntent,
        context_text: str,
        has_persistent_conversation: bool,
        state: RuntimeState,
        event_sink: Callable[[dict[str, object]], None] | None,
    ) -> AgentResponse:
        coverage = CoverageAssessment(
            status=CoverageStatus.GENERAL_ALLOWED,
            final_route="general_llm",
            reason="问题不依赖产品手册或客服政策",
            knowledge_covered=False,
        )
        self._emit(
            event_sink,
            event_type="knowledge.coverage",
            agent_id="knowledge-coverage",
            status="completed",
            label="评估知识证据覆盖度",
            summary=coverage.reason,
            payload=coverage.model_dump(mode="json"),
        )
        self._emit_agent_started(event_sink, "general", "调用通用大模型处理开放问题")
        result = self.general.run(request, context_text=context_text)
        self._emit_agent_completed(event_sink, result, "通用大模型回答完成")
        verification = self.verifier.verify_general(original_request, result)
        self._emit(
            event_sink,
            event_type="verification.started",
            agent_id="verifier",
            status="running",
            label="核验通用回答的安全与职责边界",
        )
        self._emit(
            event_sink,
            event_type="verification.completed",
            agent_id="verifier",
            status="completed" if verification.passed else "failed",
            label="通用回答边界核验完成",
            summary="通过" if verification.passed else "需要安全交接",
            payload={
                "passed": verification.passed,
                "confidence": verification.confidence,
                "action": verification.action,
            },
        )

        if result.status != "completed":
            coverage = coverage.model_copy(
                update={
                    "status": CoverageStatus.GENERAL_UNAVAILABLE,
                    "final_route": "general_unavailable",
                    "reason": "通用大模型当前不可用",
                }
            )
            answer = "通用大模型当前不可用，请稍后重试或检查模型配置。"
        elif verification.passed:
            answer = result.answer_fragment
        else:
            coverage = coverage.model_copy(
                update={
                    "status": CoverageStatus.UNSAFE_UNCOVERED,
                    "final_route": "safe_handoff",
                    "reason": "通用回答越过产品安全或官方承诺边界",
                }
            )
            answer = self._safe_handoff_answer(intent.risk_level)

        routing = self._routing_decision(intent, coverage)
        self._emit_route_resolved(event_sink, routing)
        if answer:
            self._emit(
                event_sink,
                event_type="answer.delta",
                agent_id="general",
                status="running",
                label="输出已核验的通用回答",
                payload={"delta": answer},
            )

        selected_agents = ["orchestrator", "router", "general", "verifier"]
        if has_persistent_conversation:
            selected_agents.append("memory-curator")
        steps = [
            AgentStep(
                agent_id="orchestrator",
                label="建立运行上下文",
                status="completed",
                summary="开放问题路由",
            ),
            AgentStep(
                agent_id="router",
                label="识别为通用开放问题",
                status="completed",
                summary=intent.reason_code,
            ),
            self._step_from_result(result, "调用通用大模型生成回答"),
            AgentStep(
                agent_id="verifier",
                label="核验通用回答职责边界",
                status="completed" if verification.passed else "failed",
                summary="核验通过" if verification.passed else "安全交接",
            ),
        ]
        spans = [
            TraceSpan(
                name="intent_routing",
                output_summary=self._intent_label(intent.initial_route),
                attributes={
                    "initial_route": intent.initial_route,
                    "risk_level": intent.risk_level,
                    "reason_code": intent.reason_code,
                    "llm_used": intent.llm_used,
                    "model_used": intent.model_used,
                    "classification_source": intent.classification_source,
                },
            ),
            TraceSpan(
                name="knowledge_coverage",
                status="completed",
                output_summary=coverage.reason,
                attributes=coverage.model_dump(mode="json"),
            ),
            TraceSpan(
                name="general_answer",
                status=(
                    "completed" if result.status == "completed" else "failed"
                ),
                output_summary=(
                    result.model_used or result.recommended_next_action
                ),
                attributes={
                    "llm_used": result.llm_generated,
                    "model_used": result.model_used,
                },
            ),
            TraceSpan(
                name="general_verification",
                status="completed" if verification.passed else "failed",
                output_summary=("职责边界核验通过" if verification.passed else "已安全交接"),
                attributes={"issue_codes": [item.code for item in verification.issues]},
            ),
        ]
        if intent.classification_source == "model_fallback":
            spans.insert(1, self._router_fallback_span(intent))
        if has_persistent_conversation:
            spans.append(
                TraceSpan(
                    name="layered_memory",
                    output_summary="回答完成后异步整理可追溯摘要",
                )
            )
        trace = AgentTrace(
            request_id=request_id,
            session_id=session_id,
            route=routing.final_route,
            selected_agents=selected_agents,
            steps=steps,
            spans=spans,
            total_latency_ms=max(1, round((monotonic() - state.started_at) * 1000)),
        )
        self.trace_store.save(trace, owner_id=original_request.user_id)
        return AgentResponse(
            request_id=request_id,
            session_id=session_id,
            answer=answer,
            route=routing.final_route,
            verification=verification,
            trace=trace,
            routing=routing,
        )

    def _build_gate_response(
        self,
        *,
        request_id: str,
        session_id: str,
        owner_id: str | None,
        state: RuntimeState,
        answer: str,
        verification: VerificationReport,
        intent: RoutingIntent,
        routing: RoutingDecision,
        steps: list[AgentStep],
        coverage: CoverageAssessment,
        loaded_memory=None,
        event_sink: Callable[[dict[str, object]], None] | None,
    ) -> AgentResponse:
        if routing.clarification is not None:
            self._emit(
                event_sink,
                event_type="clarification.required",
                agent_id="evidence-gap",
                status="completed",
                label="等待用户补充一项关键信息",
                summary=routing.clarification.question,
                payload=routing.clarification.model_dump(mode="json"),
            )
        self._emit_route_resolved(event_sink, routing)
        self._emit(
            event_sink,
            event_type="answer.delta",
            agent_id=(
                "evidence-gap"
                if routing.final_route == "evidence_clarification"
                else "orchestrator"
            ),
            status="running",
            label=(
                "请求补充一项关键信息"
                if routing.final_route == "evidence_clarification"
                else "输出安全交接建议"
            ),
            payload={"delta": answer},
        )
        selected_agents = [
            agent
            for agent in state.plan.selected_agents
            if agent != "verifier"
        ]
        spans = [
            TraceSpan(
                name="intent_routing",
                output_summary=self._intent_label(routing.initial_route),
                attributes={"risk_level": routing.risk_level},
            ),
            TraceSpan(
                name="knowledge_coverage",
                status="failed",
                output_summary=coverage.reason,
                attributes=coverage.model_dump(mode="json"),
            ),
        ]
        if intent.classification_source == "model_fallback":
            spans.insert(1, self._router_fallback_span(intent))
        knowledge_result = next(
            (item for item in state.results if item.agent_id == "knowledge"),
            None,
        )
        retrieval_trace = (
            knowledge_result.retrieval_trace
            if knowledge_result is not None
            else None
        )
        if retrieval_trace is not None:
            spans.extend(
                [
                    TraceSpan(
                        name="product_routing",
                        status=(
                            "completed"
                            if knowledge_result and knowledge_result.routed_products
                            else "skipped"
                        ),
                        output_summary=(
                            "、".join(knowledge_result.routed_products)
                            if knowledge_result and knowledge_result.routed_products
                            else "保持全库检索"
                        ),
                    ),
                    TraceSpan(
                        name="query_rewrite",
                        status=(
                            "completed"
                            if knowledge_result and knowledge_result.query_rewrite_model
                            else "skipped"
                        ),
                        input_summary=state.request.question[:200],
                        output_summary=(
                            knowledge_result.search_query[:300]
                            if knowledge_result and knowledge_result.search_query
                            else state.request.question[:200]
                        ),
                        attributes={
                            "model": (
                                knowledge_result.query_rewrite_model
                                if knowledge_result
                                else None
                            )
                        },
                    ),
                    TraceSpan(
                        name="parent_aggregation",
                        status=(
                            "completed" if retrieval_trace.result_count else "failed"
                        ),
                        output_summary=(
                            f"返回 {retrieval_trace.result_count} 个 Parent"
                        ),
                        attributes={
                            "rejected_reason": retrieval_trace.rejected_reason,
                            "query": retrieval_trace.query,
                        },
                    ),
                ]
            )
        if loaded_memory is not None:
            spans.append(
                TraceSpan(
                    name="session_memory",
                    status="completed",
                    input_summary=f"恢复 {loaded_memory.turn_count} 轮",
                    output_summary="只读恢复",
                    attributes={
                        "loaded": True,
                        "saved": False,
                        "prior_evidence_count": len(loaded_memory.evidence_refs),
                    },
                )
            )
        trace = AgentTrace(
            request_id=request_id,
            session_id=session_id,
            route=routing.final_route,
            selected_agents=list(dict.fromkeys(selected_agents)),
            steps=steps,
            spans=spans,
            total_latency_ms=max(1, round((monotonic() - state.started_at) * 1000)),
        )
        self.trace_store.save(trace, owner_id=owner_id)
        return AgentResponse(
            request_id=request_id,
            session_id=session_id,
            answer=answer,
            route=routing.final_route,
            citations=[],
            assets=[],
            verification=verification,
            trace=trace,
            routing=routing,
        )

    @staticmethod
    def _active_slots(
        context: ConversationContext | None,
    ) -> dict[str, str]:
        if context is None:
            return {}
        return {
            name: value
            for name in context.slots
            if (value := context.active_value(name)) is not None
        }

    @staticmethod
    def _routing_decision(
        intent: RoutingIntent,
        coverage: CoverageAssessment,
        *,
        clarification=None,
    ) -> RoutingDecision:
        return RoutingDecision(
            initial_route=intent.initial_route,
            final_route=coverage.final_route,
            route_label=Orchestrator._route_label(coverage.final_route),
            route_reason=coverage.reason,
            coverage_status=coverage.status,
            knowledge_covered=coverage.knowledge_covered,
            risk_level=intent.risk_level,
            clarification=clarification,
        )

    @staticmethod
    def _legacy_routing_decision(
        *,
        intent: RoutingIntent | None,
        plan_route: str,
        prior_routing: RoutingDecision | None,
        reason: str,
    ) -> RoutingDecision:
        initial_route = (
            intent.initial_route
            if intent is not None
            else {
                "technical": "technical_candidate",
                "customer_service": "customer_service_candidate",
                "mixed": "mixed_candidate",
            }.get(plan_route, "technical_candidate")
        )
        final_route = {
            "technical_candidate": "technical_knowledge",
            "customer_service_candidate": "customer_service",
            "mixed_candidate": "mixed",
            "general_candidate": "general_llm",
        }[initial_route]
        coverage_status = (
            CoverageStatus.GENERAL_ALLOWED
            if final_route == "general_llm"
            else (
                prior_routing.coverage_status
                if prior_routing is not None
                and prior_routing.coverage_status == CoverageStatus.COVERED
                else CoverageStatus.UNSAFE_UNCOVERED
            )
        )
        return RoutingDecision(
            initial_route=initial_route,
            final_route=final_route,
            route_label=(
                f"{Orchestrator._route_label(final_route)} · 守护链路"
            ),
            route_reason=(
                f"守护链路接管最终答案（{reason}）；"
                "保持原领域路由口径"
            ),
            coverage_status=coverage_status,
            knowledge_covered=(
                prior_routing.knowledge_covered
                if prior_routing is not None
                else False
            ),
            risk_level=intent.risk_level if intent is not None else "low",
            clarification=None,
        )

    @staticmethod
    def _router_fallback_span(intent: RoutingIntent) -> TraceSpan:
        return TraceSpan(
            name="router_fallback",
            status="completed",
            output_summary="采用确定性安全路由",
            attributes={
                "reason_code": intent.reason_code,
                "model_used": intent.model_used,
            },
        )

    def _emit_route_resolved(
        self,
        event_sink: Callable[[dict[str, object]], None] | None,
        routing: RoutingDecision,
    ) -> None:
        self._emit(
            event_sink,
            event_type="route.resolved",
            agent_id="router",
            status="completed",
            label=f"确定最终路由：{routing.route_label}",
            summary=routing.route_reason,
            payload=routing.model_dump(mode="json"),
        )

    @staticmethod
    def _intent_label(route: str) -> str:
        return {
            "technical_candidate": "技术知识候选",
            "customer_service_candidate": "客服政策候选",
            "mixed_candidate": "技术与客服混合候选",
            "general_candidate": "通用大模型候选",
        }.get(route, route)

    @staticmethod
    def _route_label(route: str) -> str:
        return {
            "technical_knowledge": "技术知识库",
            "customer_service": "客服政策",
            "mixed": "技术与客服协同",
            "evidence_clarification": "证据补全",
            "general_llm": "通用大模型",
            "general_unavailable": "通用模型不可用",
            "safe_handoff": "安全交接",
        }.get(route, route)

    @staticmethod
    def _safe_handoff_answer(risk_level: str) -> str:
        if risk_level == "high":
            return (
                "当前缺少能够支持安全处理的官方证据。请立即停止使用并断开电源，"
                "不要自行拆开或继续操作；请联系品牌官方售后或具备资质的维修人员处理。"
            )
        return (
            "当前证据不足，知识库没有足够依据支持具体结论。为避免误导，我不会生成未经手册支持的操作步骤；"
            "请补充准确型号、故障码或联系官方售后核验。"
        )

    @staticmethod
    def _legacy_confidence(*, previous: float, citations, answer: str) -> float:
        """Keep guard-path confidence tied to evidence instead of a fixed floor."""

        evidence_confidences = [
            item.evidence_confidence
            if item.evidence_confidence is not None
            else min(0.85, max(0.35, float(item.score or 0.0)))
            for item in citations
        ]
        evidence_support = (
            sum(evidence_confidences) / len(evidence_confidences)
            if evidence_confidences
            else 0.0
        )
        answer_detail = min(0.12, max(0, len(answer.strip()) - 20) / 900 * 0.12)
        confidence = (
            max(0.0, previous) * 0.4
            + evidence_support * 0.33
            + answer_detail
            + 0.07
        )
        return round(min(0.93, max(0.15, confidence)), 4)

    @staticmethod
    def _attachment_metadata(images: list[str]) -> list[dict[str, object]]:
        metadata: list[dict[str, object]] = []
        for index, image in enumerate(images[:3], start=1):
            header, _, payload = image.partition(",")
            mime_type = (
                header[5:].split(";", 1)[0]
                if header.startswith("data:")
                else "application/octet-stream"
            )
            metadata.append(
                {
                    "name": f"image-{index}",
                    "mime_type": mime_type,
                    "sha256": sha256(image.encode("utf-8")).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
        return metadata

    def _verify_results(self, results, expected_domain_agents):
        mode = self.settings.enhanced_verifier
        if mode == "off":
            return (
                self.verifier.verify(results, expected_domain_agents, enhanced=False),
                None,
            )
        enhanced = self.verifier.verify(results, expected_domain_agents, enhanced=True)
        if mode == "on":
            return enhanced, None
        basic = self.verifier.verify(results, expected_domain_agents, enhanced=False)
        return basic, enhanced

    def _emit_retrieval_completed(
        self,
        event_sink: Callable[[dict[str, object]], None] | None,
        result,
    ) -> None:
        retrieval_trace = result.retrieval_trace
        if retrieval_trace is None:
            return
        self._emit(
            event_sink,
            event_type="retrieval.completed",
            agent_id="knowledge",
            status="completed" if retrieval_trace.result_count else "failed",
            label="检索与 Parent 聚合完成",
            summary=(
                f"{retrieval_trace.mode} · {retrieval_trace.result_count} 个 Parent"
                if retrieval_trace.result_count
                else retrieval_trace.rejected_reason or "没有证据"
            ),
            payload={
                "query": retrieval_trace.query,
                "mode": retrieval_trace.mode,
                "result_count": retrieval_trace.result_count,
                "stage_counts": retrieval_trace.stage_counts,
                "rejected_reason": retrieval_trace.rejected_reason,
            },
        )

    @staticmethod
    def _emit(
        event_sink: Callable[[dict[str, object]], None] | None,
        *,
        event_type: str,
        agent_id: str,
        status: str,
        label: str,
        summary: str = "",
        payload: dict[str, object] | None = None,
    ) -> None:
        if event_sink is not None:
            event_sink(
                {
                    "type": event_type,
                    "agent_id": agent_id,
                    "status": status,
                    "label": label,
                    "summary": summary,
                    "payload": payload or {},
                }
            )

    def _emit_agent_started(
        self,
        event_sink: Callable[[dict[str, object]], None] | None,
        agent_id: str,
        label: str,
    ) -> None:
        self._emit(
            event_sink,
            event_type="agent.started",
            agent_id=agent_id,
            status="running",
            label=label,
        )

    def _emit_agent_completed(
        self,
        event_sink: Callable[[dict[str, object]], None] | None,
        result,
        label: str,
    ) -> None:
        self._emit(
            event_sink,
            event_type="agent.completed",
            agent_id=result.agent_id,
            status="completed" if result.status == "completed" else "failed",
            label=label,
            summary=(
                f"{result.model_used} · LLM 生成 · {result.latency_ms} ms"
                if result.llm_generated
                else f"确定性工具链 · {result.latency_ms} ms"
            ),
            payload={
                "llm_generated": result.llm_generated,
                "model_used": result.model_used,
                "confidence": result.confidence,
                "evidence_count": len(result.evidence),
                "search_query": result.search_query,
                "query_rewrite_model": result.query_rewrite_model,
                "routed_products": result.routed_products,
                "product_route_reason": result.product_route_reason,
            },
        )

    @staticmethod
    def _step_from_result(result, label: str) -> AgentStep:
        return AgentStep(
            agent_id=result.agent_id,
            label=label,
            status="completed" if result.status == "completed" else "failed",
            latency_ms=result.latency_ms,
            summary=(result.answer_fragment[:80] if result.answer_fragment else result.recommended_next_action or ""),
        )

    @staticmethod
    def _compose_answer(route: str, results) -> str:
        fragments = {item.agent_id: item.answer_fragment for item in results if item.answer_fragment}
        if route == "mixed":
            return f"技术处理\n{fragments.get('knowledge', '')}\n\n售后建议\n{fragments.get('customer-service', '')}".strip()
        if route == "customer_service":
            return fragments.get("customer-service", "")
        return fragments.get("knowledge", "")

    @staticmethod
    def _align_manual_assets(
        answer: str,
        assets: list[str],
        *,
        inject_missing_placeholders: bool = False,
    ) -> tuple[str, list[str]]:
        """Keep manual assets strictly aligned with visible PIC placeholders."""
        unique_assets = list(dict.fromkeys(asset.strip() for asset in assets if asset.strip()))
        if (
            inject_missing_placeholders
            and unique_assets
            and not _PIC_PLACEHOLDER_PATTERN.search(answer)
        ):
            available = min(len(unique_assets), _MAX_INLINE_MANUAL_ASSETS)
            inserted = 0

            def add_step_placeholder(match: re.Match[str]) -> str:
                nonlocal inserted
                if inserted >= available:
                    return match.group(0)
                inserted += 1
                return f"{match.group(0).rstrip()} <PIC>"

            answer = _NUMBERED_PROCEDURE_STEP_PATTERN.sub(
                add_step_placeholder,
                answer,
            )
            if inserted == 0:
                paragraphs = re.split(r"\n\s*\n", answer, maxsplit=1)
                if len(paragraphs) == 2:
                    answer = (
                        f"{paragraphs[0].rstrip()}\n\n<PIC>\n\n"
                        f"{paragraphs[1].lstrip()}"
                    )
                else:
                    first_sentence = re.search(
                        r"^(.+?[。！？!?])(\s+.+)$",
                        answer,
                        flags=re.DOTALL,
                    )
                    if first_sentence is not None:
                        answer = (
                            f"{first_sentence.group(1)}\n\n<PIC>\n\n"
                            f"{first_sentence.group(2).lstrip()}"
                        )
                    else:
                        answer = f"{answer.rstrip()}\n\n<PIC>"
        supported_count = min(
            len(unique_assets),
            _MAX_INLINE_MANUAL_ASSETS,
            len(_PIC_PLACEHOLDER_PATTERN.findall(answer)),
        )
        seen = 0

        def keep_supported_placeholder(match: re.Match[str]) -> str:
            nonlocal seen
            seen += 1
            return "<PIC>" if seen <= supported_count else ""

        normalized_answer = _PIC_PLACEHOLDER_PATTERN.sub(
            keep_supported_placeholder,
            answer,
        )
        return normalized_answer.strip(), unique_assets[:supported_count]

    @staticmethod
    def _legacy_answer_is_acceptable(
        *,
        question: str,
        route: str,
        answer: str,
        has_user_images: bool = False,
        has_visual_evidence: bool = False,
    ) -> bool:
        if has_user_images and has_visual_evidence:
            return False
        if route not in {"technical", "mixed"}:
            return True
        identifier_pattern = re.compile(
            r"(?<![A-Za-z0-9-])(?=[A-Za-z0-9-]*[A-Za-z])"
            r"(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]+(?![A-Za-z0-9-])"
        )
        requested_identifiers = {
            item.lower() for item in identifier_pattern.findall(question)
        }
        answered_identifiers = {
            item.lower() for item in identifier_pattern.findall(answer)
        }
        requested_error_codes = set(extract_normalized_error_codes(question))
        answered_error_codes = set(extract_normalized_error_codes(answer))
        if requested_error_codes - answered_error_codes:
            return False
        if requested_identifiers - answered_identifiers:
            return False
        normalized = "".join(answer.split())
        generic_markers = (
            "请提供订单号、商品型号、问题现象",
            "请提供订单号和商品当前状态",
            "需要结合商品品类、收货地址和当前订单规则",
            "售后维修需要结合故障现象、购买时间和商品状态判断",
        )
        if any(marker in normalized for marker in generic_markers):
            return False
        service_markers = (
            "订单号",
            "外包装",
            "快递面单",
            "开箱视频",
            "运输破损",
            "漏发",
            "错发",
            "补发",
            "换货",
            "退货退款",
        )
        return sum(marker in normalized for marker in service_markers) < 3
