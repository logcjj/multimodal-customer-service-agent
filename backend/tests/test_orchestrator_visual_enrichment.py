from __future__ import annotations

from fastapi.testclient import TestClient

from app.compatibility.legacy_champion import LegacyChampionAdapter
from app.config.runtime import RuntimeSettings
from app.contracts.models import AgentRequest, AgentResult, Claim, Evidence, VisualContext
from app.observability.traces import TraceStore
from app.main import create_app
from app.runtime.orchestrator import Orchestrator
from app.storage.database import Database


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, products=None, top_k: int = 5):
        self.queries.append(query)
        if "烤架" not in query:
            return []
        return [
            Evidence(
                evidence_id="manual:oven-rack",
                source_type="manual",
                title="烤箱配件 / 烤架",
                text="烤架用于承托烤盘或直接放置适合烧烤的食材。",
                product="烤箱",
                score=5,
                retrieval_stage="test",
                evidence_confidence=0.9,
            )
        ]


class StubMultimodalAgent:
    def run(self, request: AgentRequest, event_sink=None) -> AgentResult:
        vision = Evidence(
            evidence_id="vision:user-image",
            source_type="vision",
            title="用户图片视觉观察",
            text="图片中可见烤箱烤架。",
            product="烤箱",
            retrieval_stage="user_image_vlm",
            evidence_confidence=0.9,
        )
        ocr = Evidence(
            evidence_id="ocr:user-image",
            source_type="ocr",
            title="用户图片 OCR",
            text="E99",
            retrieval_stage="user_image_ocr",
            evidence_confidence=0.8,
        )
        return AgentResult(
            task_id="vision-1",
            agent_id="multimodal",
            status="completed",
            claims=[
                Claim(
                    text="已从图片提取可见信息。",
                    evidence_ids=[vision.evidence_id, ocr.evidence_id],
                )
            ],
            evidence=[vision, ocr],
            asset_ids=["shadow-visual-asset"],
            confidence=0.9,
            visual_context=VisualContext(
                ocr_text="E99",
                detected_codes=["E99"],
                detected_product="烤箱",
                detected_components=["烤架"],
                visible_objects=["金属网架"],
                visual_summary="图片中可见烤箱烤架。",
                provider_status={"ocr": "ok", "vlm": "ok"},
                confidence=0.9,
            ),
        )


def _settings(*, ocr_pipeline: str = "shadow") -> RuntimeSettings:
    return RuntimeSettings(
        offline_index_mode="on",
        image_chunk_retrieval="shadow",
        ocr_pipeline=ocr_pipeline,
        caption_embedding="shadow",
        verified_streaming="on",
        session_memory="off",
        enhanced_verifier="on",
        session_ttl_seconds=3600,
        database_url=None,
    )


def test_ocr_shadow_excludes_ocr_but_keeps_healthy_vlm_in_answer_path(
    tmp_path,
) -> None:
    database = Database(tmp_path)
    retriever = RecordingRetriever()
    orchestrator = Orchestrator(
        retriever=retriever,
        trace_store=TraceStore(database),
        legacy=LegacyChampionAdapter(answer_func=lambda question, images: ""),
        rollout_mode="agent_first",
        llm_gateway=None,
        settings=_settings(),
        session_memory=None,
    )
    orchestrator.multimodal = StubMultimodalAgent()

    response = orchestrator.run(
        AgentRequest(
            question="这个配件是做什么的？",
            images=["data:image/png;base64,aW1hZ2U="],
        )
    )

    assert "烤架" in retriever.queries[0]
    assert "E99" not in retriever.queries[0]
    assert "烤架用于" in response.answer
    assert {item.source_type for item in response.citations} == {
        "manual",
        "vision",
    }
    assert response.assets == ["shadow-visual-asset"]
    visual_span = next(
        span for span in response.trace.spans if span.name == "visual_context"
    )
    assert visual_span.attributes["mode"] == "shadow"


def test_on_visual_pipeline_can_enrich_retrieval_and_answer(tmp_path) -> None:
    database = Database(tmp_path)
    retriever = RecordingRetriever()
    orchestrator = Orchestrator(
        retriever=retriever,
        trace_store=TraceStore(database),
        legacy=LegacyChampionAdapter(answer_func=lambda question, images: ""),
        rollout_mode="agent_first",
        llm_gateway=None,
        settings=_settings(ocr_pipeline="on"),
        session_memory=None,
    )
    orchestrator.multimodal = StubMultimodalAgent()

    response = orchestrator.run(
        AgentRequest(
            question="这个配件是做什么的？",
            images=["data:image/png;base64,aW1hZ2U="],
        )
    )

    assert "烤架" in retriever.queries[0]
    assert "E99" in retriever.queries[0]
    assert "烤架用于" in response.answer
    assert {item.source_type for item in response.citations} == {
        "manual",
        "vision",
        "ocr",
    }
    assert response.assets == ["shadow-visual-asset"]


def test_verified_visual_answer_is_not_replaced_by_legacy_text_only_reply(
    tmp_path,
) -> None:
    database = Database(tmp_path)
    retriever = RecordingRetriever()
    orchestrator = Orchestrator(
        retriever=retriever,
        trace_store=TraceStore(database),
        legacy=LegacyChampionAdapter(
            answer_func=lambda question, images: (
                "无法识别图片中的具体内容，请重新上传清晰图片。"
            )
        ),
        rollout_mode="champion_guarded",
        llm_gateway=None,
        settings=_settings(ocr_pipeline="on"),
        session_memory=None,
    )
    orchestrator.multimodal = StubMultimodalAgent()

    response = orchestrator.run(
        AgentRequest(
            question="这个配件是做什么的？",
            images=["data:image/png;base64,aW1hZ2U="],
        )
    )

    assert response.used_legacy is False
    assert "烤架用于" in response.answer
    legacy_span = next(
        span for span in response.trace.spans if span.name == "legacy_champion"
    )
    assert legacy_span.status == "skipped"
    assert legacy_span.attributes["answer_adopted"] is False


def test_visual_evidence_never_delegates_back_to_text_only_legacy() -> None:
    acceptable = Orchestrator._legacy_answer_is_acceptable(
        question="图中的两个箭头分别表示什么？",
        route="technical",
        answer="图中展示了电源线连接方法。",
        has_user_images=True,
        has_visual_evidence=True,
    )

    assert acceptable is False


def test_legacy_guard_preserves_spaced_error_code_variants() -> None:
    acceptable = Orchestrator._legacy_answer_is_acceptable(
        question="相机显示 Err 02 应该怎么处理？",
        route="technical",
        answer="请关闭相机后重新插入存储卡。",
    )

    assert acceptable is False


def test_ocr_shadow_keeps_vlm_slots_but_excludes_ocr_derived_slots(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AKA_OCR_PIPELINE", "shadow")
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            rollout_mode="agent_first",
        )
    )
    client.app.state.orchestrator.multimodal = StubMultimodalAgent()

    body = client.post(
        "/api/chat",
        json={
            "question": "设备一直报警怎么办",
            "images": ["data:image/png;base64,aW1hZ2U="],
            "session_id": "shadow-visual-slots",
            "user_id": "owner-a",
        },
    ).json()
    detail = client.get(
        "/api/conversations/shadow-visual-slots",
        params={"user_id": "owner-a"},
    ).json()

    assert body["routing"]["final_route"] == "evidence_clarification"
    assert body["routing"]["clarification"]["field"] == "model"
    assert body["routing"]["clarification"]["round"] == 1
    assert detail["state"]["slots"]["product"][-1]["value"] == "烤箱"
    assert "error_code" not in detail["state"]["slots"]
