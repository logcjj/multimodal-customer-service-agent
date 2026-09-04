from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from app.compatibility.legacy_retrieval import install_compact_legacy_bm25


@dataclass(frozen=True)
class LegacyLLMRuntime:
    model: str
    base_url: str
    api_key: str
    timeout_seconds: int = 20


@dataclass(frozen=True)
class LegacyInvocation:
    llm_used: bool = False
    model_used: str | None = None
    fallback_reason: str | None = None


class _TrackedLLMClient:
    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.config = getattr(delegate, "config", None)
        self.called = False
        self.succeeded = False
        self.failure_type: str | None = None

    def chat(self, *args: Any, **kwargs: Any) -> str:
        self.called = True
        try:
            answer = self.delegate.chat(*args, **kwargs)
        except Exception as exc:
            self.failure_type = type(exc).__name__
            raise
        self.succeeded = True
        return str(answer)


class FallbackPolicy:
    @staticmethod
    def should_fallback(*, confidence: float, verification_passed: bool, remaining_ms: int) -> bool:
        return confidence < 0.58 or not verification_passed or remaining_ms < 1000


class LegacyChampionAdapter:
    def __init__(
        self,
        answer_func: Callable[[str, list[str]], str] | None = None,
        source_root: Path | None = None,
        manual_dir: Path | None = None,
        llm_runtime_provider: Callable[[], LegacyLLMRuntime | None] | None = None,
        llm_client_factory: Callable[[LegacyLLMRuntime], object] | None = None,
    ) -> None:
        self._answer_func = answer_func
        discovered_source, discovered_manual = self._discover_workspace_paths()
        self.source_root = source_root or self._path_from_env("AKA_LEGACY_SRC") or discovered_source
        self.manual_dir = manual_dir or self._path_from_env("AKA_LEGACY_MANUAL_DIR") or discovered_manual
        self._engine: Any | None = None
        self._load_error: str | None = None
        self._engine_lock = RLock()
        self._llm_runtime_provider = llm_runtime_provider
        self._llm_client_factory = llm_client_factory
        self._last_invocation: ContextVar[LegacyInvocation] = ContextVar(
            f"legacy_invocation_{id(self)}",
            default=LegacyInvocation(),
        )

    @staticmethod
    def _path_from_env(name: str) -> Path | None:
        value = os.getenv(name, "").strip()
        return Path(value).expanduser() if value else None

    @staticmethod
    def _discover_workspace_paths() -> tuple[Path | None, Path | None]:
        relative_backend = Path(
            "项目对比分析_V1/aka_submit解压副本/submit/backend"
        )
        for parent in Path(__file__).resolve().parents:
            backend = parent / relative_backend
            source = backend / "src"
            manuals = backend / "手册"
            if source.exists() and manuals.exists():
                return source, manuals
        return None, None

    @property
    def available(self) -> bool:
        if self._answer_func is not None or self._engine is not None:
            return True
        return bool(self.source_root and self.source_root.exists() and self.manual_dir and self.manual_dir.exists())

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def last_invocation(self) -> LegacyInvocation:
        return self._last_invocation.get()

    def answer(self, question: str, images: list[str]) -> str:
        self._last_invocation.set(LegacyInvocation())
        try:
            if self._answer_func is not None:
                return self._answer_func(question, images)
            engine = self._get_engine()
            if engine is None:
                return ""
            with self._engine_lock:
                self._set_menu_flag(engine, question)
                return self._answer_with_optional_llm(engine, question, images)
        except Exception as exc:
            self._load_error = type(exc).__name__
            return ""

    @staticmethod
    def _set_menu_flag(engine: object, question: str) -> None:
        engine_module = sys.modules.get(engine.__class__.__module__)
        if engine_module is not None:
            setattr(
                engine_module,
                "menu_requested",
                bool(re.search(r"\bmenu\b", question, re.IGNORECASE)),
            )

    def _answer_with_optional_llm(
        self,
        engine: object,
        question: str,
        images: list[str],
    ) -> str:
        try:
            runtime = self._llm_runtime_provider() if self._llm_runtime_provider else None
        except Exception as exc:
            self._last_invocation.set(
                LegacyInvocation(fallback_reason=f"legacy_runtime_error:{type(exc).__name__}")
            )
            return self._answer_deterministically(engine, question, images)

        if runtime is None or not runtime.api_key.strip():
            self._last_invocation.set(
                LegacyInvocation(fallback_reason="legacy_llm_unavailable")
            )
            return self._answer_deterministically(engine, question, images)

        try:
            delegate = (
                self._llm_client_factory(runtime)
                if self._llm_client_factory is not None
                else self._build_llm_client(runtime)
            )
        except Exception as exc:
            self._last_invocation.set(
                LegacyInvocation(
                    model_used=runtime.model,
                    fallback_reason=f"qwen_init_error:{type(exc).__name__}",
                )
            )
            return self._answer_deterministically(engine, question, images)

        tracked = _TrackedLLMClient(delegate)
        self._enable_llm(engine, tracked)
        try:
            answer = self._answer_with_qwen_profile(engine, question, images)
        except Exception as exc:
            tracked.failure_type = tracked.failure_type or type(exc).__name__
            answer = ""

        if tracked.failure_type is not None:
            self._last_invocation.set(
                LegacyInvocation(
                    llm_used=tracked.called,
                    model_used=runtime.model,
                    fallback_reason=f"qwen_error:{tracked.failure_type}",
                )
            )
            return self._answer_deterministically(engine, question, images)

        self._last_invocation.set(
            LegacyInvocation(
                llm_used=tracked.called and tracked.succeeded,
                model_used=runtime.model if tracked.called else None,
            )
        )
        return str(answer)

    @staticmethod
    def _enable_llm(engine: object, client: object) -> None:
        setattr(engine, "use_llm", True)
        # Keep technical Builder answers deterministic; Qwen is reserved for
        # the legacy customer-service generator, where no manual Builder exists.
        setattr(engine, "use_llm_manual_polish", False)
        setattr(engine, "use_llm_query_frame", False)
        setattr(engine, "use_llm_query_rewrite", False)
        setattr(engine, "use_ann", False)
        setattr(engine, "llm", client)
        setattr(engine, "customer_llm", client)

    @staticmethod
    def _answer_with_qwen_profile(
        engine: object,
        question: str,
        images: list[str],
    ) -> str:
        customer_module = sys.modules.get("df_kefu_baseline.customer_llm")
        had_profile = bool(
            customer_module is not None
            and hasattr(customer_module, "customer_llm_profile")
        )
        original_profile = (
            getattr(customer_module, "customer_llm_profile", None)
            if customer_module is not None
            else None
        )
        if customer_module is not None:
            setattr(customer_module, "customer_llm_profile", lambda: "qwen_original")
        try:
            return str(engine.answer(question, images=images))
        finally:
            if customer_module is not None:
                if had_profile:
                    setattr(customer_module, "customer_llm_profile", original_profile)
                else:
                    delattr(customer_module, "customer_llm_profile")

    @staticmethod
    def _answer_deterministically(
        engine: object,
        question: str,
        images: list[str],
    ) -> str:
        names = (
            "use_llm",
            "use_llm_manual_polish",
            "use_llm_query_frame",
            "use_llm_query_rewrite",
            "use_ann",
            "llm",
            "customer_llm",
        )
        previous = {name: getattr(engine, name, None) for name in names}
        setattr(engine, "use_llm", False)
        setattr(engine, "use_llm_manual_polish", False)
        setattr(engine, "use_llm_query_frame", False)
        setattr(engine, "use_llm_query_rewrite", False)
        setattr(engine, "use_ann", False)
        setattr(engine, "llm", None)
        setattr(engine, "customer_llm", None)
        try:
            return str(engine.answer(question, images=images))
        finally:
            for name, value in previous.items():
                setattr(engine, name, value)

    @staticmethod
    def _build_llm_client(runtime: LegacyLLMRuntime) -> object:
        from df_kefu_baseline.llm_client import LLMConfig, OpenAICompatibleClient

        config = LLMConfig(
            api_key=runtime.api_key,
            base_url=runtime.base_url.rstrip("/"),
            model=runtime.model,
            timeout=runtime.timeout_seconds,
            max_tokens=0,
            enable_thinking=False,
            temperature=0.0,
        )
        return OpenAICompatibleClient(config)

    def _get_engine(self) -> Any | None:
        if self._engine is not None:
            return self._engine
        if not self.available or self.source_root is None or self.manual_dir is None:
            return None
        try:
            source = str(self.source_root)
            if source not in sys.path:
                sys.path.insert(0, source)
            from df_kefu_baseline import answer_evidence, answer_manual_polish, answer_text, retrieval
            from df_kefu_baseline.answer_engine import AnswerEngine

            install_compact_legacy_bm25(retrieval)

            # The frozen submission split one large module into three files and
            # expects their public helpers to share a namespace. Bridge those
            # namespaces at runtime without mutating the original source tree.
            split_modules = (answer_text, answer_manual_polish, answer_evidence)
            public_symbols = {
                name: value
                for module in split_modules
                for name, value in vars(module).items()
                if not name.startswith("__")
            }
            for module in split_modules:
                for name, value in public_symbols.items():
                    module.__dict__.setdefault(name, value)

            self._engine = AnswerEngine(
                use_llm=False,
                use_ann=False,
                use_llm_query_frame=False,
                use_llm_query_rewrite=False,
                use_llm_manual_polish=False,
                manual_dir=self.manual_dir,
                illustration_dir=self.manual_dir / "插图",
            )
            return self._engine
        except Exception as exc:
            self._load_error = type(exc).__name__
            return None
