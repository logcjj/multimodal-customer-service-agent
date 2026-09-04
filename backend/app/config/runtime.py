from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, cast


FeatureMode = Literal["off", "shadow", "on"]
_FEATURE_MODES = {"off", "shadow", "on"}
_MIN_SESSION_TTL_SECONDS = 300
_MAX_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60


def _feature_mode(name: str, default: FeatureMode) -> FeatureMode:
    value = os.getenv(name, default).strip().lower()
    if value not in _FEATURE_MODES:
        raise ValueError(f"{name} must be one of: off, shadow, on")
    return cast(FeatureMode, value)


def _session_ttl_seconds() -> int:
    raw = os.getenv("AKA_SESSION_TTL_SECONDS", "604800").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("AKA_SESSION_TTL_SECONDS must be an integer") from exc
    if not _MIN_SESSION_TTL_SECONDS <= value <= _MAX_SESSION_TTL_SECONDS:
        raise ValueError(
            "AKA_SESSION_TTL_SECONDS must be between "
            f"{_MIN_SESSION_TTL_SECONDS} and {_MAX_SESSION_TTL_SECONDS}"
        )
    return value


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class RuntimeSettings:
    offline_index_mode: FeatureMode
    image_chunk_retrieval: FeatureMode
    ocr_pipeline: FeatureMode
    caption_embedding: FeatureMode
    verified_streaming: FeatureMode
    session_memory: FeatureMode
    enhanced_verifier: FeatureMode
    session_ttl_seconds: int
    database_url: str | None
    dynamic_routing: FeatureMode = "on"
    conversation_history: FeatureMode = "on"
    layered_memory: FeatureMode = "on"
    general_agent: FeatureMode = "on"
    max_clarification_rounds: int = 3
    conversation_context_tokens: int = 6000
    legacy_llm_model: str = "qwen3-max"
    legacy_llm_timeout_seconds: int = 20

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        database_url = os.getenv("AKA_DATABASE_URL", "").strip() or None
        return cls(
            offline_index_mode=_feature_mode("AKA_OFFLINE_INDEX_MODE", "on"),
            image_chunk_retrieval=_feature_mode("AKA_IMAGE_CHUNK_RETRIEVAL", "on"),
            ocr_pipeline=_feature_mode("AKA_OCR_PIPELINE", "on"),
            caption_embedding=_feature_mode("AKA_CAPTION_EMBEDDING", "on"),
            verified_streaming=_feature_mode("AKA_VERIFIED_STREAMING", "on"),
            session_memory=_feature_mode("AKA_SESSION_MEMORY", "on"),
            enhanced_verifier=_feature_mode("AKA_ENHANCED_VERIFIER", "on"),
            session_ttl_seconds=_session_ttl_seconds(),
            database_url=database_url,
            dynamic_routing=_feature_mode("AKA_DYNAMIC_ROUTING", "on"),
            conversation_history=_feature_mode("AKA_CONVERSATION_HISTORY", "on"),
            layered_memory=_feature_mode("AKA_LAYERED_MEMORY", "on"),
            general_agent=_feature_mode("AKA_GENERAL_AGENT", "on"),
            max_clarification_rounds=_bounded_int(
                "AKA_MAX_CLARIFICATION_ROUNDS",
                3,
                1,
                3,
            ),
            conversation_context_tokens=_bounded_int(
                "AKA_CONVERSATION_CONTEXT_TOKENS",
                6000,
                512,
                131_072,
            ),
            legacy_llm_model=(
                os.getenv("AKA_LEGACY_LLM_MODEL", "qwen3-max").strip()
                or "qwen3-max"
            ),
            legacy_llm_timeout_seconds=_bounded_int(
                "AKA_LEGACY_LLM_TIMEOUT_SECONDS",
                20,
                1,
                120,
            ),
        )

    @staticmethod
    def is_enabled(mode: FeatureMode) -> bool:
        return mode in {"shadow", "on"}

    @staticmethod
    def affects_answer(mode: FeatureMode) -> bool:
        return mode == "on"
