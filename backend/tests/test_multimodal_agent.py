from __future__ import annotations

from app.agents.multimodal import MultimodalAgent
from app.config.runtime import RuntimeSettings
from app.contracts.models import AgentRequest, ModelKind
from app.models.llm_gateway import LLMOutput


class FakeVisualGateway:
    def available(self, kind=ModelKind.LLM):
        return kind in {ModelKind.OCR, ModelKind.VLM}

    def generate(self, *, kind, **kwargs):
        if kind == ModelKind.OCR:
            return LLMOutput(
                text='{"visible_text":"ERROR E03","codes":["E03"],"numbers":[],"confidence":0.95}',
                provider="test",
                model="ocr-test",
                latency_ms=5,
            )
        return LLMOutput(
            text='{"product":"洗衣机","components":["排水过滤器"],"visible_objects":["显示屏"],'
            '"summary":"显示屏显示 E03","confidence":0.9}',
            provider="test",
            model="vlm-test",
            latency_ms=7,
        )


def _settings(ocr: str = "on") -> RuntimeSettings:
    return RuntimeSettings(
        offline_index_mode="on",
        image_chunk_retrieval="shadow",
        ocr_pipeline=ocr,
        caption_embedding="shadow",
        verified_streaming="on",
        session_memory="on",
        enhanced_verifier="on",
        session_ttl_seconds=3600,
        database_url=None,
    )


def test_multimodal_agent_runs_independent_ocr_and_vlm() -> None:
    agent = MultimodalAgent(FakeVisualGateway(), settings=_settings())

    result = agent.run(
        AgentRequest(
            question="这个错误怎么处理？",
            images=["data:image/png;base64,aW1hZ2U="],
        )
    )

    assert result.status == "completed"
    assert result.visual_context is not None
    assert result.visual_context.detected_codes == ["E03"]
    assert result.visual_context.detected_product == "洗衣机"
    assert result.visual_context.field_provenance == {
        "ocr_text": "ocr",
        "detected_codes": "ocr",
        "detected_numbers": "ocr",
        "detected_product": "vlm",
        "detected_components": "vlm",
        "visible_objects": "vlm",
        "visual_summary": "vlm",
    }
    assert {item.source_type for item in result.evidence} == {"ocr", "vision"}
    assert result.model_used == "ocr-test + vlm-test"


def test_multimodal_agent_honestly_degrades_when_ocr_is_off() -> None:
    agent = MultimodalAgent(FakeVisualGateway(), settings=_settings(ocr="off"))

    result = agent.run(
        AgentRequest(question="看图", images=["data:image/png;base64,aW1hZ2U="])
    )

    assert result.visual_context is not None
    assert result.visual_context.ocr_text == ""
    assert result.visual_context.provider_status["ocr"] == "disabled"
    assert result.visual_context.detected_product == "洗衣机"
    assert result.visual_context.provider_status["vlm"] == "ok"
    assert all(item.source_type != "ocr" for item in result.evidence)
    assert any(item.source_type == "vision" for item in result.evidence)


def test_vlm_prompt_requests_owning_product_category_instead_of_accessory_shape() -> None:
    class ProductCategoryGateway(FakeVisualGateway):
        def generate(self, *, kind, **kwargs):
            if kind == ModelKind.VLM:
                assert "所属主设备品类" in kwargs["system_prompt"]
                assert "疑似功能名称" in kwargs["system_prompt"]
                assert "编号、箭头或流程图" in kwargs["system_prompt"]
                assert "箭头的起点、终点和移动方向" in kwargs["system_prompt"]
                return LLMOutput(
                    text=(
                        '{"product":"洗碗机","components":["餐具篮"],'
                        '"visible_objects":["灰色塑料提篮"],'
                        '"summary":"图片显示一个灰色塑料提篮","confidence":0.9}'
                    ),
                    provider="test",
                    model="vlm-test",
                    latency_ms=7,
                )
            return super().generate(kind=kind, **kwargs)

    result = MultimodalAgent(ProductCategoryGateway(), settings=_settings()).run(
        AgentRequest(
            question="这个小提篮是干什么用的？",
            images=["data:image/png;base64,aW1hZ2U="],
        )
    )

    assert result.visual_context is not None
    assert result.visual_context.detected_product == "洗碗机"
    assert result.visual_context.detected_components == ["餐具篮"]
