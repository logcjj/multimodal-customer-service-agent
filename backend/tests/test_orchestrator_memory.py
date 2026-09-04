from __future__ import annotations

import json

from app.agents.knowledge import KnowledgeAgent
from app.contracts.models import AgentRequest, Evidence
from tests.knowledge_fixtures import create_client_with_manuals


class RecordingRetriever:
    def __init__(self) -> None:
        self.query = ""

    def search(self, query, products=None, top_k=5):
        self.query = query
        return [
            Evidence(
                evidence_id="manual:blower-minor",
                source_type="manual",
                title="吹风机人身安全",
                text="禁止未成年人操作吹风机。",
            )
        ]


def test_followup_restores_product_and_code_but_product_switch_drops_old_context(tmp_path) -> None:
    client = create_client_with_manuals(tmp_path, rollout_mode="agent_first")

    first = client.post(
        "/api/chat",
        json={"question": "洗衣机 E03 怎么处理？", "session_id": "followup-session"},
    )
    assert first.status_code == 200

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "那还能继续使用吗？", "session_id": "followup-session"},
    ) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]
    second = events[-1]["payload"]["response"]

    assert any(item["type"] == "session.loaded" for item in events)
    assert second["citations"]
    assert "E03" in second["citations"][0]["text"]

    switched = client.post(
        "/api/chat",
        json={"question": "空气净化器滤网怎么清洁？", "session_id": "followup-session"},
    ).json()
    assert switched["citations"][0]["title"] == "空气净化器滤网清洁"
    assert "E03" not in switched["citations"][0]["text"]


def test_followup_retrieval_keeps_structured_product_but_excludes_previous_answer_summary() -> None:
    retriever = RecordingRetriever()
    agent = KnowledgeAgent(retriever)

    result = agent.run(
        AgentRequest(
            question=(
                "那未成年人可以操作吗？\n"
                "上一轮产品：吹风机手册\n"
                "上一轮意图：technical\n"
                "上一轮回答摘要：严禁将吹风机喷口对准人或动物。"
            )
        )
    )

    assert result.status == "completed"
    assert "未成年人" in retriever.query
    assert "上一轮产品：吹风机手册" in retriever.query
    assert "上一轮回答摘要" not in retriever.query
    assert "喷口对准人或动物" not in retriever.query


def test_shadow_session_memory_records_context_without_injecting_it_into_retrieval(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AKA_SESSION_MEMORY", "shadow")
    client = create_client_with_manuals(tmp_path, rollout_mode="agent_first")
    client.post(
        "/api/chat",
        json={"question": "洗衣机 E03 怎么处理？", "session_id": "shadow-memory"},
    )

    second = client.post(
        "/api/chat",
        json={"question": "那还能继续使用吗？", "session_id": "shadow-memory"},
    ).json()
    rewrite = next(item for item in second["trace"]["spans"] if item["name"] == "query_rewrite")

    assert any(
        item["name"] == "session_memory" and item["attributes"]["loaded"]
        for item in second["trace"]["spans"]
    )
    assert "上一轮产品" not in rewrite["output_summary"]
    assert "上一轮型号或错误码" not in rewrite["output_summary"]
