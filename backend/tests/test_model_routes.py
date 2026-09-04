from __future__ import annotations

from fastapi.testclient import TestClient

from app.contracts.models import ModelKind
from app.main import create_app


def build_client(tmp_path) -> TestClient:
    return TestClient(create_app(data_dir=tmp_path))


def test_provider_catalog_exposes_supported_capabilities(tmp_path) -> None:
    client = build_client(tmp_path)

    response = client.get("/api/providers")

    assert response.status_code == 200
    providers = response.json()
    openai_compatible = next(
        item for item in providers if item["id"] == "openai-compatible"
    )
    assert openai_compatible["name"] == "OpenAI-Compatible（自定义接口）"
    assert any("vlm" in item["capabilities"] for item in providers)
    openai = next(item for item in providers if item["id"] == "openai")
    assert "vlm" in openai["capabilities"]
    tongyi = next(item for item in providers if item["id"] == "tongyi-qianwen")
    assert "ocr" in tongyi["capabilities"]
    assert "asr" in tongyi["capabilities"]
    assert "tts" in tongyi["capabilities"]
    ocr_presets = tongyi["model_presets"]["ocr"]
    assert [item["name"] for item in ocr_presets] == [
        "qwen3.5-ocr",
        "qwen-vl-ocr-latest",
        "qwen-vl-ocr-2025-11-20",
        "qwen-vl-ocr",
    ]
    assert [item["name"] for item in tongyi["model_presets"]["vlm"]] == [
        "qwen3-vl-flash",
        "qwen3-vl-plus",
    ]
    assert [item["name"] for item in tongyi["model_presets"]["rerank"]] == [
        "qwen3-rerank",
    ]
    assert all(
        "recommended" not in preset
        for presets in tongyi["model_presets"].values()
        for preset in presets
    )


def test_create_model_masks_and_persists_secret(tmp_path) -> None:
    client = build_client(tmp_path)

    created = client.post(
        "/api/models",
        json={
            "provider": "OpenAI-API-Compatible",
            "name": "qwen3-max",
            "kind": "llm",
            "base_url": "https://example.com/v1",
            "api_key": "secret-model-token",
            "capabilities": ["tools", "json"],
        },
    )

    assert created.status_code == 201
    body = created.json()
    assert body["secret_configured"] is True
    assert body["secret_hint"].startswith("••••")
    assert body["is_default"] is True
    assert "secret-model-token" not in created.text

    listed = client.get("/api/models")
    assert listed.status_code == 200
    assert listed.json()[0]["name"] == "qwen3-max"
    assert "secret-model-token" not in listed.text


def test_default_model_cannot_be_deleted_until_replaced(tmp_path) -> None:
    client = build_client(tmp_path)
    model_id = client.post(
        "/api/models",
        json={
            "provider": "DeepSeek",
            "name": "deepseek-chat",
            "kind": "llm",
            "base_url": "https://api.deepseek.com/v1",
        },
    ).json()["id"]

    set_default = client.post(f"/api/models/{model_id}/default")
    deleted = client.delete(f"/api/models/{model_id}")

    assert set_default.status_code == 200
    assert set_default.json()["is_default"] is True
    assert deleted.status_code == 409


def test_named_runtime_selects_non_default_llm_without_replacing_default(tmp_path) -> None:
    client = build_client(tmp_path)
    deepseek_id = client.post(
        "/api/models",
        json={
            "provider": "DeepSeek",
            "name": "deepseek-v4-flash",
            "kind": "llm",
            "base_url": "https://api.deepseek.com",
            "api_key": "deepseek-secret",
        },
    ).json()["id"]
    qwen_id = client.post(
        "/api/models",
        json={
            "provider": "Tongyi-Qianwen",
            "name": "qwen3-max",
            "kind": "llm",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "qwen-secret",
        },
    ).json()["id"]

    runtime = client.app.state.model_service.get_runtime_by_name(
        ModelKind.LLM,
        "QWEN3-MAX",
    )

    assert runtime is not None
    record, secret = runtime
    assert record.id == qwen_id
    assert record.is_default is False
    assert record.name == "qwen3-max"
    assert secret == "qwen-secret"
    models = {item["id"]: item for item in client.get("/api/models").json()}
    assert models[deepseek_id]["is_default"] is True
    assert models[qwen_id]["is_default"] is False


def test_app_injects_named_qwen_runtime_into_legacy_adapter(tmp_path) -> None:
    client = build_client(tmp_path)
    client.post(
        "/api/models",
        json={
            "provider": "DeepSeek",
            "name": "deepseek-v4-flash",
            "kind": "llm",
            "base_url": "https://api.deepseek.com",
            "api_key": "deepseek-secret",
        },
    )
    client.post(
        "/api/models",
        json={
            "provider": "Tongyi-Qianwen",
            "name": "qwen3-max",
            "kind": "llm",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "qwen-secret",
        },
    )

    class FakeEngine:
        use_llm = False
        use_llm_manual_polish = False
        use_llm_query_frame = False
        use_llm_query_rewrite = False
        use_ann = False
        llm = None
        customer_llm = None

        def answer(self, question, images):
            if self.use_llm and self.llm is not None:
                return self.llm.chat([{"role": "user", "content": question}])
            return "deterministic"

    class FakeClient:
        def __init__(self, model: str) -> None:
            self.model = model

        def chat(self, messages, temperature=None):
            return f"{self.model}:{messages[-1]['content']}"

    adapter = client.app.state.orchestrator.legacy
    adapter._engine = FakeEngine()
    adapter._llm_client_factory = lambda runtime: FakeClient(runtime.model)

    answer = adapter.answer("冠军回退", [])

    assert answer == "qwen3-max:冠军回退"
    assert adapter.last_invocation.llm_used is True
    assert adapter.last_invocation.model_used == "qwen3-max"
    default_runtime = client.app.state.model_service.get_default_runtime(ModelKind.LLM)
    assert default_runtime is not None
    assert default_runtime[0].name == "deepseek-v4-flash"


def test_connection_test_returns_unhealthy_without_secret(tmp_path) -> None:
    client = build_client(tmp_path)
    model_id = client.post(
        "/api/models",
        json={
            "provider": "OpenAI",
            "name": "gpt-test",
            "kind": "llm",
            "base_url": "https://example.com/v1",
        },
    ).json()["id"]

    response = client.post(f"/api/models/{model_id}/test")

    assert response.status_code == 200
    assert response.json()["health"] == "unhealthy"
    assert response.json()["message"] == "未配置访问凭据"


def test_model_kind_can_be_corrected_without_replacing_the_llm_default(tmp_path) -> None:
    client = build_client(tmp_path)
    llm_id = client.post(
        "/api/models",
        json={
            "provider": "DeepSeek",
            "name": "deepseek-chat",
            "kind": "llm",
            "base_url": "https://api.deepseek.com/v1",
        },
    ).json()["id"]
    visual_model_id = client.post(
        "/api/models",
        json={
            "provider": "Tongyi-Qianwen",
            "name": "qwen3-vl-flash",
            "kind": "llm",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "capabilities": ["llm"],
        },
    ).json()["id"]

    response = client.patch(
        f"/api/models/{visual_model_id}",
        json={"kind": "vlm"},
    )

    assert response.status_code == 200
    assert response.json()["kind"] == "vlm"
    assert response.json()["capabilities"] == ["vlm"]
    assert response.json()["is_default"] is True
    assert response.json()["health"] == "untested"

    models = {item["id"]: item for item in client.get("/api/models").json()}
    assert models[llm_id]["kind"] == "llm"
    assert models[llm_id]["is_default"] is True
    assert models[visual_model_id]["kind"] == "vlm"
