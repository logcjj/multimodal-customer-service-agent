from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import httpx

from app.contracts.models import ModelKind
from app.models.service import ModelService
from app.providers.errors import (
    ProviderFailure,
    classify_provider_failure,
    failure_from_response,
)


@dataclass(frozen=True)
class ModelInvocation:
    kind: str
    provider: str
    model: str
    latency_ms: int
    status: str
    item_count: int


class ModelGateway:
    def __init__(
        self,
        model_service: ModelService,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.model_service = model_service
        self.transport = transport
        self.timeout = timeout
        self.last_invocation: ModelInvocation | None = None
        self.last_error: str | None = None
        self.last_failure: ProviderFailure | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.last_failure = None
        runtime = self.model_service.get_default_runtime(ModelKind.EMBEDDING)
        if runtime is None or not texts:
            self.last_error = "embedding-not-configured"
            return []
        record, secret = runtime
        if record.health == "unhealthy":
            self._unavailable("embedding-model-unhealthy")
            return []
        if not secret:
            self._authentication_failure("embedding-credential-missing")
            return []
        started = perf_counter()
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout, trust_env=False) as client:
                response = client.post(
                    f"{record.base_url.rstrip('/')}/embeddings",
                    headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
                    json={"model": record.name, "input": texts, "encoding_format": "float"},
                )
                response.raise_for_status()
            body = response.json()
            data = sorted(body.get("data", []), key=lambda item: int(item.get("index", 0)))
            vectors = [[float(value) for value in item["embedding"]] for item in data]
            if len(vectors) != len(texts):
                raise ValueError("embedding-count-mismatch")
            self._success("embedding", record.provider, record.name, started, len(texts))
            return vectors
        except httpx.HTTPStatusError as exc:
            self._failure("embedding", record.provider, record.name, started, failure_from_response(exc.response))
            return []
        except Exception as exc:
            body = {"error_code": str(exc)} if str(exc) == "embedding-count-mismatch" else None
            failure = classify_provider_failure(status_code=None, body=body, exception=exc)
            self._failure("embedding", record.provider, record.name, started, failure)
            return []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.last_failure = None
        runtime = self.model_service.get_default_runtime(ModelKind.RERANK)
        if runtime is None or not documents:
            self.last_error = "rerank-not-configured"
            return []
        record, secret = runtime
        if record.health == "unhealthy":
            self._unavailable("rerank-model-unhealthy")
            return []
        if not secret:
            self._authentication_failure("rerank-credential-missing")
            return []
        started = perf_counter()
        try:
            endpoint = "reranks" if record.name.startswith("qwen3-") or "compatible-api" in record.base_url else "rerank"
            with httpx.Client(transport=self.transport, timeout=self.timeout, trust_env=False) as client:
                response = client.post(
                    f"{record.base_url.rstrip('/')}/{endpoint}",
                    headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
                    json={"model": record.name, "query": query, "documents": documents, "top_n": len(documents)},
                )
                response.raise_for_status()
            body = response.json()
            results = body.get("results", [])
            if not isinstance(results, list) or not results:
                raise ValueError("rerank-results-missing")
            scores = [0.0] * len(documents)
            for item in results:
                index = int(item.get("index", -1))
                if 0 <= index < len(scores):
                    scores[index] = float(item.get("relevance_score", item.get("score", 0.0)))
            self._success("rerank", record.provider, record.name, started, len(documents))
            return scores
        except httpx.HTTPStatusError as exc:
            self._failure("rerank", record.provider, record.name, started, failure_from_response(exc.response))
            return []
        except Exception as exc:
            failure = classify_provider_failure(status_code=None, body=None, exception=exc)
            self._failure("rerank", record.provider, record.name, started, failure)
            return []

    def _success(self, kind: str, provider: str, model: str, started: float, count: int) -> None:
        self.last_error = None
        self.last_failure = None
        self.last_invocation = ModelInvocation(
            kind=kind,
            provider=provider,
            model=model,
            latency_ms=max(1, round((perf_counter() - started) * 1000)),
            status="success",
            item_count=count,
        )

    def _failure(
        self,
        kind: str,
        provider: str,
        model: str,
        started: float,
        failure: ProviderFailure,
    ) -> None:
        self.last_error = failure.code
        self.last_failure = failure
        self.last_invocation = ModelInvocation(
            kind=kind,
            provider=provider,
            model=model,
            latency_ms=max(1, round((perf_counter() - started) * 1000)),
            status="failed",
            item_count=0,
        )

    def _unavailable(self, internal_code: str) -> None:
        self.last_error = internal_code
        self.last_failure = ProviderFailure(
            code="provider_unavailable",
            message="模型已被标记为不可用，请重新执行健康检查。",
            retryable=True,
        )

    def _authentication_failure(self, internal_code: str) -> None:
        self.last_error = internal_code
        self.last_failure = ProviderFailure(
            code="authentication_error",
            message="模型访问凭据未配置。",
            retryable=False,
        )
