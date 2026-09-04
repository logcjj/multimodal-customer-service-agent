from __future__ import annotations

from app.multimodal.visual_context import (
    merge_visual_context,
    parse_ocr_output,
    parse_vlm_output,
)


def test_visual_context_preserves_ocr_codes_numbers_and_vlm_objects() -> None:
    context = merge_visual_context(
        image_hashes=["hash-1"],
        ocr=parse_ocr_output(
            '{"visible_text":"ERROR E03 220V","codes":["E03"],"numbers":["220V"],'
            '"label_fields":{"model":"WM-100"},"confidence":0.98}'
        ),
        vision=parse_vlm_output(
            '{"product":"洗衣机","components":["排水过滤器","排水过滤器"],'
            '"visible_objects":["显示屏","过滤器"],"summary":"屏幕显示 E03","confidence":0.9}'
        ),
    )

    assert context.ocr_text == "ERROR E03 220V"
    assert context.detected_codes == ["E03", "WM-100"]
    assert context.detected_numbers == ["220V"]
    assert context.detected_product == "洗衣机"
    assert context.detected_components == ["排水过滤器"]
    assert context.visible_objects == ["显示屏", "过滤器"]
    assert context.confidence == 0.9


def test_invalid_ocr_json_does_not_create_synthetic_text() -> None:
    ocr = parse_ocr_output("not-json")
    context = merge_visual_context(
        image_hashes=["hash-1"],
        ocr=ocr,
        vision=parse_vlm_output("{}"),
    )

    assert context.ocr_text == ""
    assert context.detected_codes == []
    assert context.provider_status["ocr"] == "invalid_response"


def test_markdown_json_fence_is_supported_without_accepting_extra_prose() -> None:
    parsed = parse_ocr_output(
        '```json\n{"visible_text":"SN 12345","codes":[],"numbers":["12345"],"confidence":0.8}\n```'
    )
    invalid = parse_ocr_output('结果如下：{"visible_text":"SN 12345"}')

    assert parsed.visible_text == "SN 12345"
    assert parsed.status == "ok"
    assert invalid.status == "invalid_response"


def test_ocr_ignores_non_object_label_fields_without_crashing() -> None:
    parsed = parse_ocr_output(
        '{"visible_text":"遥控器屏幕", "codes":[], "numbers":[], '
        '"label_fields":["screen"], "confidence":0.8}'
    )

    assert parsed.status == "ok"
    assert parsed.visible_text == "遥控器屏幕"
    assert parsed.label_fields == {}
