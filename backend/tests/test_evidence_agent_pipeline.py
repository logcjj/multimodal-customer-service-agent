from __future__ import annotations

import json

from app.agents.verifier import VerifierAgent
from tests.knowledge_fixtures import create_client_with_manuals


def test_stream_exposes_real_retrieval_rerank_and_verification_events(tmp_path) -> None:
    client = create_client_with_manuals(tmp_path, rollout_mode="agent_first")

    with client.stream("POST", "/api/chat/stream", json={"question": "洗衣机 E03 怎么处理？"}) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    event_types = [item["type"] for item in events]
    assert "retrieval.started" in event_types
    assert "retrieval.completed" in event_types
    retrieval = next(item for item in events if item["type"] == "retrieval.completed")
    assert retrieval["payload"]["result_count"] >= 1
    assert retrieval["payload"]["mode"] in {"lexical-only", "hybrid", "hybrid-rerank"}
    assert event_types.index("retrieval.started") < event_types.index("agent.completed")


def test_trace_contains_retrieval_and_verification_spans(tmp_path) -> None:
    client = create_client_with_manuals(tmp_path, rollout_mode="agent_first")
    response = client.post(
        "/api/chat",
        json={"question": "空气净化器滤网怎么清洁？", "user_id": "owner-a"},
    ).json()

    trace = client.get(
        f"/api/traces/{response['request_id']}",
        params={"user_id": "owner-a"},
    ).json()

    names = {item["name"] for item in trace["spans"]}
    assert {"query_understanding", "lexical_retrieval", "parent_aggregation", "claim_verification"} <= names
    assert all("api_key" not in json.dumps(item) for item in trace["spans"])


def test_verifier_replaces_unsupported_number_with_primary_evidence(tmp_path) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=lambda kind, system, user, images: "请等待 99 分钟再启动洗衣机。",
        legacy_answer_func=lambda question, images: "",
    )

    body = client.post("/api/chat", json={"question": "洗衣机 E03 怎么处理？"}).json()

    assert body["verification"]["passed"] is True
    assert "99 分钟" not in body["answer"]
    assert "检查排水管并清理排水过滤器" in body["answer"]
    assert body["used_legacy"] is False


def test_verifier_accepts_localized_value_inside_evidence_range() -> None:
    assert VerifierAgent._measurement_supported("15分钟", "soakfor10-15minutes") is True
    assert VerifierAgent._measurement_supported("99分钟", "soakfor10-15minutes") is False


def test_verifier_accepts_chinese_date_parts_from_iso_and_slash_dates() -> None:
    support = "徐江涛-2026-07-23工作日报，整理了7/21-7/23的连续工作线"

    assert VerifierAgent._measurement_supported("2026年", support) is True
    assert VerifierAgent._measurement_supported("7月", support) is True
    assert VerifierAgent._measurement_supported("23日", support) is True
    assert VerifierAgent._measurement_supported("8月", support) is False


def test_verifier_feedback_uses_evidence_without_second_llm_call(tmp_path) -> None:
    prompts: list[str] = []

    def generate(kind, system, user, images):
        prompts.append(system)
        if "检索查询改写器" in system:
            return "中文：洗衣机 E03 排水 | English: washing machine E03 drain"
        if "审校智能体" in system:
            return "请先关闭电源，检查排水管并清理排水过滤器。"
        return "请等待 99 分钟，再检查洗衣机。"

    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=generate,
        legacy_answer_func=lambda question, images: "",
    )

    with client.stream("POST", "/api/chat/stream", json={"question": "洗衣机 E03 怎么处理？"}) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    final = events[-1]["payload"]["response"]
    event_types = [item["type"] for item in events]
    assert final["verification"]["passed"] is True
    assert "检查排水管并清理排水过滤器" in final["answer"]
    assert not any("审校智能体" in prompt for prompt in prompts)
    assert "answer.evidence_fallback" in event_types
    assert "answer.revision.started" not in event_types
    assert "answer.revision.completed" not in event_types
    revised = next(item for item in events if item["type"] == "answer.revised")
    assert revised["payload"]["answer"] == final["answer"]


def test_shadow_verifier_records_enhanced_issues_without_changing_answer(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AKA_ENHANCED_VERIFIER", "shadow")
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=lambda kind, system, user, images: "请等待 99 分钟再启动洗衣机。",
        legacy_answer_func=lambda question, images: "",
    )

    body = client.post("/api/chat", json={"question": "洗衣机 E03 怎么处理？"}).json()
    verification_span = next(
        item for item in body["trace"]["spans"] if item["name"] == "claim_verification"
    )

    assert body["verification"]["passed"] is True
    assert "99 分钟" in body["answer"]
    assert verification_span["attributes"]["enhanced_mode"] == "shadow"
    assert "unsupported-number-or-model" in verification_span["attributes"][
        "shadow_issue_codes"
    ]
