from __future__ import annotations

import json

import httpx

from app.contracts.models import ModelConfigurationCreate, ModelKind
from app.models.llm_gateway import LLMGateway
from app.models.service import ModelService
from app.storage.database import Database


class FragmentedStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


def _gateway(tmp_path, handler) -> LLMGateway:
    service = ModelService(Database(tmp_path))
    model = service.create_model(
        ModelConfigurationCreate(
            provider="deepseek",
            name="deepseek-chat",
            kind=ModelKind.LLM,
            base_url="https://llm.example/v1",
            api_key="stream-test-secret",
        )
    )
    service.set_default(model.id)
    return LLMGateway(service, transport=httpx.MockTransport(handler))


def test_openai_stream_decodes_fragmented_chinese_sse_and_done(tmp_path) -> None:
    captured: dict[str, object] = {}
    raw = (
        'data: {"choices":[{"delta":{"content":"请先"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"断电。"}}]}\n\n'
        'data: [DONE]\n\n'
    ).encode("utf-8")
    split = raw.index("请".encode("utf-8")) + 1
    stream = FragmentedStream([raw[:split], raw[split : split + 2], raw[split + 2 :]])

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, stream=stream)

    gateway = _gateway(tmp_path, handler)

    deltas = list(
        gateway.generate_stream(
            kind=ModelKind.LLM,
            system_prompt="只根据证据回答",
            user_prompt="问题和证据",
        )
    )

    assert deltas == ["请先", "断电。"]
    assert captured["body"]["stream"] is True
    assert stream.closed is True


def test_openai_stream_classifies_provider_error_without_leaking_body(tmp_path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"code": "invalid_api_key", "message": "bad stream-test-secret"}},
        )

    gateway = _gateway(tmp_path, handler)

    assert list(
        gateway.generate_stream(
            kind=ModelKind.LLM,
            system_prompt="system",
            user_prompt="user",
        )
    ) == []
    assert gateway.last_failure is not None
    assert gateway.last_failure.code == "authentication_error"
    assert "stream-test-secret" not in gateway.last_failure.message
