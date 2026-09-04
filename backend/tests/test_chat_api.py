from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import create_app
from tests.knowledge_fixtures import create_client_with_manuals


def test_technical_chat_returns_evidence_and_agent_trace(tmp_path) -> None:
    client = create_client_with_manuals(tmp_path)

    response = client.post(
        "/api/chat",
        json={"question": "空气净化器的滤网应该怎么清洁？", "session_id": "demo-tech"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "technical_knowledge"
    assert body["routing"]["final_route"] == "technical_knowledge"
    assert body["trace"]["route"] == "technical_knowledge"
    assert "拔下插头" in body["answer"]
    citation = body["citations"][0]
    document = client.get("/api/datasets").json()[0]
    linked_document = client.get(f"/api/datasets/{document['id']}/documents").json()[0]
    assert citation["source_type"] == "manual"
    assert citation["file_id"] == linked_document["file_id"]
    assert citation["document_name"] == "test-manual.md"
    assert citation["document_mime_type"] == "text/markdown"
    assert citation["chapter_title"] == "空气净化器滤网清洁"
    assert citation["parent_id"] == citation["section_id"]
    assert citation["retrieval_stage"] in {"lexical-only", "hybrid", "hybrid-rerank"}
    assert 0 <= citation["evidence_confidence"] <= 1
    assert body["trace"]["selected_agents"] == [
        "orchestrator",
        "router",
        "knowledge",
        "verifier",
    ]
    assert body["verification"]["passed"] is True


def test_mixed_chat_uses_domain_agents_without_exposing_hidden_reasoning(tmp_path) -> None:
    client = create_client_with_manuals(tmp_path)

    response = client.post(
        "/api/chat",
        json={"question": "洗衣机出现 E03 报错，处理不好可以退货吗？", "session_id": "demo-mixed"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "mixed"
    assert "knowledge" in body["trace"]["selected_agents"]
    assert "customer-service" in body["trace"]["selected_agents"]
    assert "退货" in body["answer"]
    assert "chain_of_thought" not in response.text
    assert "reasoning" not in response.text


def test_unknown_question_uses_legacy_champion_when_available(tmp_path) -> None:
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            legacy_answer_func=lambda question, images: f"旧冠军链路回答：{question}",
        )
    )

    response = client.post("/api/chat", json={"question": "一个完全未知的产品问题"})

    assert response.status_code == 200
    body = response.json()
    assert body["used_legacy"] is True
    assert body["answer"].startswith("旧冠军链路回答")
    assert body["trace"]["fallback_reason"] == "新链路证据不足"


def test_trace_is_persisted_and_queryable(tmp_path) -> None:
    client = create_client_with_manuals(tmp_path)
    created = client.post(
        "/api/chat",
        json={"question": "健身追踪器表带怎么安装？", "user_id": "owner-a"},
    ).json()

    listed = client.get("/api/traces", params={"user_id": "owner-a"})
    detail = client.get(
        f"/api/traces/{created['request_id']}",
        params={"user_id": "owner-a"},
    )

    assert listed.status_code == 200
    assert listed.json()[0]["request_id"] == created["request_id"]
    assert detail.status_code == 200
    assert detail.json()["session_id"] == created["session_id"]


def test_feedback_is_linked_to_request_without_rewriting_knowledge(tmp_path) -> None:
    client = create_client_with_manuals(tmp_path)
    created = client.post("/api/chat", json={"question": "健身追踪器表带怎么安装？"}).json()

    response = client.post(
        "/api/feedback",
        json={"request_id": created["request_id"], "rating": "down", "category": "image", "comment": "图片不相关"},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "queued-for-offline-review"
    assert response.json()["knowledge_updated"] is False


def test_champion_guarded_mode_keeps_legacy_answer_while_recording_v3_evidence(tmp_path) -> None:
    client = create_client_with_manuals(
        tmp_path,
        legacy_answer_func=lambda question, images: "冠军链路：E03 请先断电并清理排水过滤器。",
        rollout_mode="champion_guarded",
    )

    response = client.post("/api/chat", json={"question": "洗衣机 E03 怎么处理？"})

    assert response.status_code == 200
    body = response.json()
    assert body["used_legacy"] is True
    assert body["answer"].startswith("冠军链路")
    assert body["citations"]
    assert body["trace"]["fallback_reason"] == "守护链路模式"


def test_champion_guard_rejects_legacy_answer_that_drops_requested_error_code(
    tmp_path,
) -> None:
    client = create_client_with_manuals(
        tmp_path,
        legacy_answer_func=lambda question, images: (
            "冠军链路：请先断电并清理排水过滤器。"
        ),
        rollout_mode="champion_guarded",
    )

    response = client.post("/api/chat", json={"question": "洗衣机 E03 怎么处理？"})

    assert response.status_code == 200
    body = response.json()
    assert body["used_legacy"] is False
    assert "E03" in body["answer"]
    assert body["verification"]["passed"] is True


def test_readiness_reports_rollout_and_legacy_status(tmp_path) -> None:
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            legacy_answer_func=lambda question, images: "ok",
            rollout_mode="agent_first",
        )
    )

    response = client.get("/api/readiness")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "rollout_mode": "agent_first",
        "legacy_available": True,
        "model_registry": "ready",
        "trace_store": "ready",
        "llm_configured": False,
        "llm_model": None,
        "vlm_configured": False,
        "embedding_configured": False,
        "rerank_configured": False,
        "ocr_configured": False,
        "dynamic_routing": "on",
        "conversation_history": "on",
        "layered_memory": "on",
        "general_agent": "on",
        "image_chunk_retrieval": "on",
        "ocr_pipeline": "on",
        "caption_embedding": "on",
        "legacy_llm_model": "qwen3-max",
        "legacy_llm_configured": False,
    }


def test_champion_guard_rejects_generic_legacy_reply_when_verified_manual_evidence_exists(tmp_path) -> None:
    client = create_client_with_manuals(
        tmp_path,
        legacy_answer_func=lambda question, images: "您好，请提供订单号、商品型号、问题现象和您的具体诉求。",
        rollout_mode="champion_guarded",
    )

    response = client.post("/api/chat", json={"question": "洗衣机出现 E03 怎么处理？"})

    assert response.status_code == 200
    body = response.json()
    assert body["used_legacy"] is False
    assert "E03" in body["answer"]
    assert body["verification"]["passed"] is True


def test_technical_no_evidence_rejects_generic_service_legacy_reply(tmp_path) -> None:
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
            legacy_answer_func=lambda question, images: (
                "您好，售后维修需要结合故障现象、购买时间和商品状态判断。"
                "请提供订单号、故障描述、照片或视频。"
            ),
        )
    )

    body = client.post(
        "/api/chat",
        json={"question": "火星牌量子咖啡机出现 ZX999 应该怎么维修？"},
    ).json()

    assert body["used_legacy"] is False
    assert body["citations"] == []
    assert "当前证据不足" in body["answer"]


def test_technical_no_evidence_rejects_legacy_reply_that_drops_requested_error_code(
    tmp_path,
) -> None:
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
            legacy_answer_func=lambda question, images: (
                "您好，非常抱歉影响您的使用体验。请先保留商品、外包装、快递面单和异常位置照片或视频，"
                "并提供订单号。我们会核实是否属于运输破损、漏发错发或商品异常，再安排补发、换货、"
                "退货退款或维修。"
            ),
        )
    )

    body = client.post(
        "/api/chat",
        json={"question": "洗衣机出现 E03 排水故障应该怎么处理？"},
    ).json()

    assert body["used_legacy"] is False
    assert body["citations"] == []
    assert "当前证据不足" in body["answer"]


def test_legacy_only_mode_preserves_frozen_answer_without_challenger_takeover(tmp_path) -> None:
    frozen_answer = (
        "您好，请提供订单号、商品型号、问题现象和您的具体诉求。"
    )
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="legacy_only",
            legacy_answer_func=lambda question, images: frozen_answer,
        )
    )

    body = client.post(
        "/api/chat",
        json={"question": "火星牌量子咖啡机出现 ZX999 应该怎么维修？"},
    ).json()

    assert body["used_legacy"] is True
    assert body["answer"] == frozen_answer
    assert body["trace"]["fallback_reason"] == "仅守护链路模式"


def test_streaming_chat_exposes_bounded_agent_status_events_and_final_response(tmp_path) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=lambda kind, system, user, images: "请断电后检查排水管，并清理排水过滤器。",
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "洗衣机出现 E03 怎么处理？", "session_id": "stream-demo"},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    assert response.status_code == 200
    event_types = [item["type"] for item in events]
    assert event_types[0] == "run.started"
    assert "plan.completed" in event_types
    assert any(item["type"] == "agent.started" and item["agent_id"] == "knowledge" for item in events)
    assert any(item["type"] == "agent.completed" and item["agent_id"] == "knowledge" for item in events)
    assert "verification.started" in event_types
    assert "verification.completed" in event_types
    assert event_types[-1] == "run.completed"
    assert events[-1]["payload"]["response"]["answer"].startswith("请断电后")
    assert events[-1]["payload"]["response"]["trace"]["selected_agents"] == [
        "orchestrator",
        "router",
        "knowledge",
        "verifier",
    ]


def test_mixed_route_starts_independent_specialists_before_collecting_results(tmp_path) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=lambda kind, system, user, images: "根据现有证据，需要先排查故障，再核对退货条件。",
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "洗衣机出现 E03，修不好可以退货吗？"},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    knowledge_started = next(index for index, item in enumerate(events) if item["type"] == "agent.started" and item["agent_id"] == "knowledge")
    service_started = next(index for index, item in enumerate(events) if item["type"] == "agent.started" and item["agent_id"] == "customer-service")
    first_domain_completed = min(
        index
        for index, item in enumerate(events)
        if item["type"] == "agent.completed" and item["agent_id"] in {"knowledge", "customer-service"}
    )

    assert knowledge_started < first_domain_completed
    assert service_started < first_domain_completed


def test_visual_context_enriches_retrieval_and_citations_when_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AKA_OCR_PIPELINE", "on")
    seen_prompts: list[tuple[str, str, str]] = []

    def generate(kind, system, user, images):
        seen_prompts.append((kind, system, user))
        if kind == "ocr":
            return '{"visible_text":"ERROR E03","codes":["E03"],"numbers":[],"confidence":0.95}'
        if kind == "vlm":
            return (
                '{"product":"洗衣机","components":["显示屏"],"visible_objects":["显示屏"],'
                '"summary":"显示屏显示 E03","confidence":0.9}'
            )
        if "检索查询改写器" in system:
            return "中文：洗衣机 E03 排水 | English: washing machine E03 drain"
        return "洗衣机出现 E03 时，请先关闭电源，检查排水管并清理排水过滤器。"

    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=generate,
    )
    body = client.post(
        "/api/chat",
        json={
            "question": "这个错误怎么处理？",
            "images": ["data:image/png;base64,aW1hZ2U="],
        },
    ).json()

    retrieval_prompts = [
        user
        for kind, _, user in seen_prompts
        if kind == "llm" and "用户问题" in user
    ]
    answer_system_prompts = [
        system
        for kind, system, _ in seen_prompts
        if kind == "llm" and "产品技术支持智能体" in system
    ]
    assert any("E03" in prompt and "洗衣机" in prompt for prompt in retrieval_prompts)
    assert any(
        "图中编号、箭头或操作顺序" in prompt
        and "同图 ImageChunk" in prompt
        and "排除项" in prompt
        and "不得扩写为 USB-C 或 USB-A" in prompt
        for prompt in answer_system_prompts
    )
    assert {item["source_type"] for item in body["citations"]} >= {"manual", "ocr", "vision"}
    assert any(item["name"] == "visual_context" for item in body["trace"]["spans"])
