from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_secret_never_appears_in_model_readiness_or_trace_responses(tmp_path) -> None:
    secret = "private-redaction-test-token"
    client = TestClient(create_app(data_dir=tmp_path, llm_generate_func=lambda kind, system, user, images: "请先断电。"))
    created = client.post(
        "/api/models",
        json={
            "provider": "DeepSeek",
            "name": "deepseek-v4-flash",
            "kind": "llm",
            "base_url": "https://api.deepseek.com",
            "api_key": secret,
        },
    )
    assert created.status_code == 201
    client.post(
        "/api/chat",
        json={"question": "洗衣机 E03 怎么处理？", "user_id": "owner-a"},
    )

    payloads = [
        created.text,
        client.get("/api/models").text,
        client.get("/api/readiness").text,
        client.get("/api/traces", params={"user_id": "owner-a"}).text,
    ]

    assert all(secret not in payload for payload in payloads)
