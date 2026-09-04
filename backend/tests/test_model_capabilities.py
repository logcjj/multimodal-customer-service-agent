from __future__ import annotations

import json

import httpx

from app.contracts.models import ModelConfigurationCreate, ModelKind
from app.knowledge.providers import ModelGateway
from app.models.service import ModelService
from app.storage.database import Database


def test_embedding_gateway_posts_input_and_returns_vectors(tmp_path) -> None:
    service = ModelService(Database(tmp_path))
    service.create_model(
        ModelConfigurationCreate(
            provider="OpenAI-API-Compatible",
            name="embedding-test",
            kind=ModelKind.EMBEDDING,
            base_url="https://example.test/v1",
            api_key="unit-test-secret",
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert json.loads(request.content)["input"] == ["E03 排水故障"]
        assert request.headers["Authorization"] == "Bearer unit-test-secret"
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

    gateway = ModelGateway(service, transport=httpx.MockTransport(handler))

    assert gateway.embed(["E03 排水故障"]) == [[0.1, 0.2]]
    assert gateway.last_invocation is not None
    assert gateway.last_invocation.kind == "embedding"


def test_rerank_gateway_restores_input_order(tmp_path) -> None:
    service = ModelService(Database(tmp_path))
    service.create_model(
        ModelConfigurationCreate(
            provider="Tongyi-Qianwen",
            name="rerank-test",
            kind=ModelKind.RERANK,
            base_url="https://example.test/v1",
            api_key="unit-test-secret",
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}]},
        )

    gateway = ModelGateway(service, transport=httpx.MockTransport(handler))

    assert gateway.rerank("排水", ["表带", "排水过滤器"]) == [0.4, 0.9]


def test_gateway_never_exposes_raw_provider_error_or_secret(tmp_path) -> None:
    service = ModelService(Database(tmp_path))
    service.create_model(
        ModelConfigurationCreate(
            provider="OpenAI-API-Compatible",
            name="embedding-test",
            kind=ModelKind.EMBEDDING,
            base_url="https://example.test/v1",
            api_key="private-unit-test-secret",
        )
    )

    gateway = ModelGateway(
        service,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text="Authorization private-unit-test-secret")
        ),
    )

    assert gateway.embed(["text"]) == []
    assert gateway.last_error == "authentication_error"
    assert gateway.last_failure is not None
    assert gateway.last_failure.code == "authentication_error"
    assert "private-unit-test-secret" not in repr(gateway.last_invocation)
    assert "private-unit-test-secret" not in repr(gateway.last_failure)
