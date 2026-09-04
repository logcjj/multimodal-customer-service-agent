from __future__ import annotations

import pytest

from app.config.runtime import RuntimeSettings


def test_released_multimodal_feature_modes_default_to_on(monkeypatch) -> None:
    for name in (
        "AKA_OFFLINE_INDEX_MODE",
        "AKA_IMAGE_CHUNK_RETRIEVAL",
        "AKA_OCR_PIPELINE",
        "AKA_CAPTION_EMBEDDING",
        "AKA_VERIFIED_STREAMING",
        "AKA_SESSION_MEMORY",
        "AKA_ENHANCED_VERIFIER",
        "AKA_DYNAMIC_ROUTING",
        "AKA_CONVERSATION_HISTORY",
        "AKA_LAYERED_MEMORY",
        "AKA_GENERAL_AGENT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("AKA_MAX_CLARIFICATION_ROUNDS", raising=False)
    monkeypatch.delenv("AKA_CONVERSATION_CONTEXT_TOKENS", raising=False)
    monkeypatch.delenv("AKA_LEGACY_LLM_MODEL", raising=False)
    monkeypatch.delenv("AKA_LEGACY_LLM_TIMEOUT_SECONDS", raising=False)

    settings = RuntimeSettings.from_env()

    assert settings.offline_index_mode == "on"
    assert settings.image_chunk_retrieval == "on"
    assert settings.ocr_pipeline == "on"
    assert settings.caption_embedding == "on"
    assert settings.verified_streaming == "on"
    assert settings.session_memory == "on"
    assert settings.enhanced_verifier == "on"
    assert settings.dynamic_routing == "on"
    assert settings.conversation_history == "on"
    assert settings.layered_memory == "on"
    assert settings.general_agent == "on"
    assert settings.max_clarification_rounds == 3
    assert settings.conversation_context_tokens == 6000
    assert settings.legacy_llm_model == "qwen3-max"
    assert settings.legacy_llm_timeout_seconds == 20


def test_feature_modes_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("AKA_IMAGE_CHUNK_RETRIEVAL", "on")
    monkeypatch.setenv("AKA_OCR_PIPELINE", "off")
    monkeypatch.setenv("AKA_SESSION_TTL_SECONDS", "7200")
    monkeypatch.setenv("AKA_DATABASE_URL", "postgresql+psycopg://app@db/aka")
    monkeypatch.setenv("AKA_LEGACY_LLM_MODEL", "qwen3-max-2026-01-23")
    monkeypatch.setenv("AKA_LEGACY_LLM_TIMEOUT_SECONDS", "45")

    settings = RuntimeSettings.from_env()

    assert settings.image_chunk_retrieval == "on"
    assert settings.ocr_pipeline == "off"
    assert settings.session_ttl_seconds == 7200
    assert settings.database_url == "postgresql+psycopg://app@db/aka"
    assert settings.legacy_llm_model == "qwen3-max-2026-01-23"
    assert settings.legacy_llm_timeout_seconds == 45


def test_invalid_feature_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("AKA_IMAGE_CHUNK_RETRIEVAL", "sometimes")

    with pytest.raises(ValueError, match="off, shadow, on"):
        RuntimeSettings.from_env()


@pytest.mark.parametrize("value", ["299", str(30 * 24 * 60 * 60 + 1), "abc"])
def test_invalid_session_ttl_is_rejected(monkeypatch, value: str) -> None:
    monkeypatch.setenv("AKA_SESSION_TTL_SECONDS", value)

    with pytest.raises(ValueError, match="AKA_SESSION_TTL_SECONDS"):
        RuntimeSettings.from_env()


@pytest.mark.parametrize("value", ["0", "4", "abc"])
def test_invalid_max_clarification_rounds_is_rejected(
    monkeypatch,
    value: str,
) -> None:
    monkeypatch.setenv("AKA_MAX_CLARIFICATION_ROUNDS", value)

    with pytest.raises(ValueError, match="AKA_MAX_CLARIFICATION_ROUNDS"):
        RuntimeSettings.from_env()


@pytest.mark.parametrize("value", ["511", "131073", "abc"])
def test_invalid_conversation_context_tokens_is_rejected(
    monkeypatch,
    value: str,
) -> None:
    monkeypatch.setenv("AKA_CONVERSATION_CONTEXT_TOKENS", value)

    with pytest.raises(ValueError, match="AKA_CONVERSATION_CONTEXT_TOKENS"):
        RuntimeSettings.from_env()


@pytest.mark.parametrize("value", ["0", "121", "abc"])
def test_invalid_legacy_llm_timeout_is_rejected(monkeypatch, value: str) -> None:
    monkeypatch.setenv("AKA_LEGACY_LLM_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="AKA_LEGACY_LLM_TIMEOUT_SECONDS"):
        RuntimeSettings.from_env()
