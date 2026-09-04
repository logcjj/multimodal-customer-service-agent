from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.capability_routes import router as capability_router
from app.api.chat_routes import router as chat_router
from app.api.conversation_routes import router as conversation_router
from app.api.feedback_routes import router as feedback_router
from app.api.evaluation_routes import router as evaluation_router
from app.api.knowledge_routes import router as knowledge_router
from app.api.index_routes import router as index_router
from app.api.mcp_routes import router as mcp_router
from app.api.model_routes import router as model_router
from app.api.session_routes import router as session_router
from app.api.retrieval_routes import router as retrieval_router
from app.api.trace_routes import router as trace_router
from app.agents.memory_curator import MemoryCuratorAgent
from app.compatibility.legacy_champion import LegacyChampionAdapter, LegacyLLMRuntime
from app.config.runtime import RuntimeSettings
from app.contracts.models import ModelKind
from app.conversations.store import ConversationStore
from app.knowledge.service import KnowledgeService, LiveKnowledgeRetriever
from app.knowledge.providers import ModelGateway
from app.evaluation.service import EvaluationService
from app.models.service import ModelService
from app.models.llm_gateway import GenerateOverride, LLMGateway
from app.observability.traces import FeedbackRecord, TraceRecord, TraceStore
from app.observability.metrics import MetricsRegistry
from app.runtime.orchestrator import Orchestrator
from app.runtime.conversation_memory import ConversationMemoryService
from app.runtime.dynamic_routing import IntentRouter, KnowledgeCoverageGate
from app.runtime.session_memory import SessionMemoryStore
from app.storage.database import Database

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:5176",
    "http://127.0.0.1:5176",
)


def configured_cors_origins() -> list[str]:
    extra_origins = [
        origin.strip()
        for origin in os.getenv("AKA_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    return list(dict.fromkeys((*DEFAULT_CORS_ORIGINS, *extra_origins)))


def create_app(
    data_dir: Path | None = None,
    legacy_answer_func: Callable[[str, list[str]], str] | None = None,
    rollout_mode: str | None = None,
    llm_generate_func: GenerateOverride | None = None,
) -> FastAPI:
    resolved_data_dir = data_dir or Path(__file__).resolve().parents[1] / "data"
    settings = RuntimeSettings.from_env()
    database = Database(resolved_data_dir, database_url=settings.database_url)

    app = FastAPI(title="Multimodal Customer Service Agent", version="0.7.0")
    app.state.settings = settings
    app.state.database = database
    app.state.model_service = ModelService(database)
    app.state.metrics = MetricsRegistry()
    app.state.session_memory = SessionMemoryStore(
        database,
        ttl_seconds=settings.session_ttl_seconds,
    )
    app.state.conversations = ConversationStore(database)
    app.state.conversation_memory = ConversationMemoryService(
        database,
        app.state.conversations,
        context_tokens=settings.conversation_context_tokens,
    )
    app.state.knowledge_service = KnowledgeService(database, settings=settings)
    app.state.evaluation_service = EvaluationService(database, app.state.knowledge_service)
    app.state.llm_gateway = LLMGateway(
        app.state.model_service,
        generate_override=llm_generate_func,
    )
    app.state.intent_router = IntentRouter(app.state.llm_gateway)
    app.state.coverage_gate = KnowledgeCoverageGate(
        max_rounds=settings.max_clarification_rounds,
    )
    app.state.memory_curator = MemoryCuratorAgent(
        app.state.conversation_memory,
        app.state.llm_gateway,
    )
    app.state.model_gateway = ModelGateway(app.state.model_service)
    app.state.knowledge_service.embed_override = app.state.model_gateway.embed
    app.state.knowledge_service.rerank_override = app.state.model_gateway.rerank

    def embedding_model_name() -> str:
        runtime = app.state.model_service.get_default_runtime(ModelKind.EMBEDDING)
        return runtime[0].name if runtime else "embedding-unconfigured"

    app.state.knowledge_service.embedding_model_provider = embedding_model_name
    app.state.knowledge_service.embedding_configured_provider = lambda: bool(
        app.state.model_service.get_default_runtime(ModelKind.EMBEDDING)
    )

    def legacy_llm_runtime() -> LegacyLLMRuntime | None:
        runtime = app.state.model_service.get_runtime_by_name(
            ModelKind.LLM,
            settings.legacy_llm_model,
        )
        if runtime is None:
            return None
        record, secret = runtime
        if not secret:
            return None
        return LegacyLLMRuntime(
            model=settings.legacy_llm_model,
            base_url=record.base_url,
            api_key=secret,
            timeout_seconds=settings.legacy_llm_timeout_seconds,
        )

    def shutdown_knowledge_service() -> None:
        app.state.knowledge_service.shutdown()

    def shutdown_memory_curator() -> None:
        app.state.memory_curator.shutdown()

    def preload_active_index_bundles() -> None:
        app.state.knowledge_service.preload_active_bundles()

    app.router.add_event_handler("startup", preload_active_index_bundles)
    app.router.add_event_handler("shutdown", shutdown_knowledge_service)
    app.router.add_event_handler("shutdown", shutdown_memory_curator)
    app.state.trace_store = TraceStore(database)
    resolved_rollout_mode = rollout_mode or "agent_first"
    app.state.orchestrator = Orchestrator(
        retriever=LiveKnowledgeRetriever(app.state.knowledge_service),
        trace_store=app.state.trace_store,
        legacy=LegacyChampionAdapter(
            answer_func=legacy_answer_func,
            llm_runtime_provider=legacy_llm_runtime,
        ),
        rollout_mode=resolved_rollout_mode,
        llm_gateway=app.state.llm_gateway,
        settings=settings,
        session_memory=app.state.session_memory,
        conversations=app.state.conversations,
        conversation_memory=app.state.conversation_memory,
        intent_router=app.state.intent_router,
        coverage_gate=app.state.coverage_gate,
        memory_curator=app.state.memory_curator,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configured_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
    app.include_router(capability_router)
    app.include_router(chat_router)
    app.include_router(conversation_router)
    app.include_router(feedback_router)
    app.include_router(evaluation_router)
    app.include_router(knowledge_router)
    app.include_router(index_router)
    app.include_router(model_router)
    app.include_router(mcp_router)
    app.include_router(retrieval_router)
    app.include_router(session_router)
    app.include_router(trace_router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "aka-multi-agent"}

    @app.get("/api/metrics")
    def metrics() -> dict[str, object]:
        return app.state.metrics.snapshot()

    @app.get("/api/readiness")
    def readiness() -> dict[str, object]:
        llm_configured = app.state.llm_gateway.available()
        vlm_configured = app.state.llm_gateway.available(ModelKind.VLM)
        embedding_runtime = app.state.model_service.get_default_runtime(ModelKind.EMBEDDING)
        rerank_runtime = app.state.model_service.get_default_runtime(ModelKind.RERANK)
        ocr_runtime = app.state.model_service.get_default_runtime(ModelKind.OCR)
        embedding_configured = bool(embedding_runtime and embedding_runtime[0].health == "healthy")
        rerank_configured = bool(rerank_runtime and rerank_runtime[0].health == "healthy")
        ocr_configured = bool(ocr_runtime and ocr_runtime[0].health == "healthy")
        return {
            "status": "ready",
            "rollout_mode": app.state.orchestrator.rollout_mode,
            "legacy_available": app.state.orchestrator.legacy.available,
            "model_registry": "ready",
            "trace_store": "ready",
            "llm_configured": llm_configured,
            "llm_model": app.state.llm_gateway.model_name() if llm_configured else None,
            "vlm_configured": vlm_configured,
            "embedding_configured": embedding_configured,
            "rerank_configured": rerank_configured,
            "ocr_configured": ocr_configured,
            "dynamic_routing": settings.dynamic_routing,
            "conversation_history": settings.conversation_history,
            "layered_memory": settings.layered_memory,
            "general_agent": settings.general_agent,
            "image_chunk_retrieval": settings.image_chunk_retrieval,
            "ocr_pipeline": settings.ocr_pipeline,
            "caption_embedding": settings.caption_embedding,
            "legacy_llm_model": settings.legacy_llm_model,
            "legacy_llm_configured": legacy_llm_runtime() is not None,
        }

    return app


app = create_app(rollout_mode=os.getenv("AKA_ROLLOUT_MODE", "champion_guarded"))
