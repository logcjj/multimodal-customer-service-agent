from __future__ import annotations

import json

from app.contracts.models import Evidence
from app.runtime.verified_stream import VerifiedSentenceBuffer
from tests.knowledge_fixtures import create_client_with_manuals


def test_verified_sentence_buffer_emits_supported_and_withholds_unsupported_promise() -> None:
    buffer = VerifiedSentenceBuffer(
        [
            Evidence(
                evidence_id="manual:1",
                source_type="manual",
                title="安全说明",
                text="请先断电，再检查排水管。",
            )
        ]
    )

    emitted = buffer.feed("请先断电。我们保证免费维修。")
    result = buffer.finish()

    assert emitted == ["请先断电。"]
    assert result.emitted_text == "请先断电。"
    assert result.withheld_text == "我们保证免费维修。"
    assert "unsupported-service-commitment" in result.issue_codes


def test_chat_stream_emits_answer_deltas_before_verification_and_canonical_result(tmp_path) -> None:
    client = create_client_with_manuals(
        tmp_path,
        rollout_mode="agent_first",
        llm_generate_func=lambda kind, system, user, images: "请先关闭电源。检查排水管并清理排水过滤器。",
    )

    with client.stream("POST", "/api/chat/stream", json={"question": "洗衣机 E03 怎么处理？"}) as response:
        events = [json.loads(line) for line in response.iter_lines() if line]

    assert response.headers.get("content-encoding") == "identity"
    event_types = [item["type"] for item in events]
    assert "generation.started" in event_types
    assert "answer.delta" in event_types
    assert event_types.index("answer.delta") < event_types.index("verification.started")
    deltas = "".join(item["payload"]["delta"] for item in events if item["type"] == "answer.delta")
    final = events[-1]["payload"]["response"]
    assert deltas == final["answer"]
    assert events[-1]["type"] == "run.completed"
