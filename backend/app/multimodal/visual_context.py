from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.contracts.models import VisualContext


ProviderStatus = Literal[
    "ok",
    "empty",
    "disabled",
    "unavailable",
    "provider_error",
    "invalid_response",
]


class ParsedOCR(BaseModel):
    model_config = ConfigDict(frozen=True)

    visible_text: str = ""
    codes: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    label_fields: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0, le=1)
    status: ProviderStatus = "empty"


class ParsedVision(BaseModel):
    model_config = ConfigDict(frozen=True)

    product: str | None = None
    components: list[str] = Field(default_factory=list)
    visible_objects: list[str] = Field(default_factory=list)
    summary: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    status: ProviderStatus = "empty"


_FENCED_JSON = re.compile(r"\A```(?:json)?\s*\n(?P<body>\{.*\})\s*\n```\Z", re.DOTALL | re.IGNORECASE)
_CODE_TOKEN = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9][A-Za-z0-9._/-]*\Z")


def _strict_json_object(raw: str) -> dict[str, object] | None:
    text = raw.strip()
    fenced = _FENCED_JSON.fullmatch(text)
    if fenced:
        text = fenced.group("body")
    elif not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _text(value: object, *, limit: int = 4000) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _strings(value: object, *, limit: int = 30) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        candidates = []
    return list(dict.fromkeys(item.strip() for item in candidates if item.strip()))[:limit]


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def parse_ocr_output(raw: str) -> ParsedOCR:
    value = _strict_json_object(raw)
    if value is None:
        return ParsedOCR(status="invalid_response")
    raw_label_fields = value.get("label_fields")
    label_fields_source = raw_label_fields if isinstance(raw_label_fields, dict) else {}
    label_fields = {
        str(key).strip(): str(item).strip()
        for key, item in label_fields_source.items()
        if str(key).strip()
        and isinstance(item, str)
        and item.strip()
    }
    visible_text = _text(value.get("visible_text"))
    codes = _strings(value.get("codes"))
    numbers = _strings(value.get("numbers"))
    populated = bool(visible_text or codes or numbers or label_fields)
    return ParsedOCR(
        visible_text=visible_text,
        codes=codes,
        numbers=numbers,
        label_fields=label_fields,
        confidence=_confidence(value.get("confidence")),
        status="ok" if populated else "empty",
    )


def parse_vlm_output(raw: str) -> ParsedVision:
    value = _strict_json_object(raw)
    if value is None:
        return ParsedVision(status="invalid_response")
    product = _text(value.get("product"), limit=160) or None
    components = _strings(value.get("components"))
    visible_objects = _strings(value.get("visible_objects"))
    summary = _text(value.get("summary"))
    populated = bool(product or components or visible_objects or summary)
    return ParsedVision(
        product=product,
        components=components,
        visible_objects=visible_objects,
        summary=summary,
        confidence=_confidence(value.get("confidence")),
        status="ok" if populated else "empty",
    )


def empty_ocr(status: ProviderStatus) -> ParsedOCR:
    return ParsedOCR(status=status)


def empty_vision(status: ProviderStatus) -> ParsedVision:
    return ParsedVision(status=status)


def image_hashes(images: list[str]) -> list[str]:
    hashes: list[str] = []
    for image in images:
        payload = image.encode("utf-8")
        if image.startswith("data:") and "," in image:
            encoded = image.split(",", 1)[1]
            try:
                payload = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                payload = image.encode("utf-8")
        hashes.append(hashlib.sha256(payload).hexdigest())
    return hashes


def merge_visual_context(
    *,
    image_hashes: list[str],
    ocr: ParsedOCR,
    vision: ParsedVision,
) -> VisualContext:
    label_codes = [value for value in ocr.label_fields.values() if _CODE_TOKEN.fullmatch(value)]
    confidences = [
        confidence
        for status, confidence in ((ocr.status, ocr.confidence), (vision.status, vision.confidence))
        if status == "ok" and confidence > 0
    ]
    return VisualContext(
        image_hashes=list(dict.fromkeys(image_hashes)),
        ocr_text=ocr.visible_text,
        detected_codes=list(dict.fromkeys([*ocr.codes, *label_codes])),
        detected_numbers=list(dict.fromkeys(ocr.numbers)),
        detected_product=vision.product,
        detected_components=list(dict.fromkeys(vision.components)),
        visible_objects=list(dict.fromkeys(vision.visible_objects)),
        visual_summary=vision.summary,
        provider_status={"ocr": ocr.status, "vlm": vision.status},
        field_provenance={
            "ocr_text": "ocr",
            "detected_codes": "ocr",
            "detected_numbers": "ocr",
            "detected_product": "vlm",
            "detected_components": "vlm",
            "visible_objects": "vlm",
            "visual_summary": "vlm",
        },
        confidence=min(confidences) if confidences else 0.0,
    )


def visual_context_for_answer(
    context: VisualContext,
    *,
    include_ocr: bool,
) -> VisualContext:
    """Return only fields whose source is enabled for the live answer path."""

    if include_ocr:
        return context
    provenance = {
        "ocr_text": "ocr",
        "detected_codes": "ocr",
        "detected_numbers": "ocr",
        "detected_product": "vlm",
        "detected_components": "vlm",
        "visible_objects": "vlm",
        "visual_summary": "vlm",
        **context.field_provenance,
    }
    empty_values: dict[str, object] = {
        "ocr_text": "",
        "detected_codes": [],
        "detected_numbers": [],
        "detected_product": None,
        "detected_components": [],
        "visible_objects": [],
        "visual_summary": "",
    }
    updates = {
        field_name: empty_value
        for field_name, empty_value in empty_values.items()
        if provenance.get(field_name) == "ocr"
    }
    updates["field_provenance"] = {
        field_name: source
        for field_name, source in provenance.items()
        if source != "ocr"
    }
    return context.model_copy(update=updates)


def visual_search_text(
    context: VisualContext,
    *,
    include_ocr: bool = True,
    include_vision: bool = True,
) -> str:
    parts: list[str] = []
    if include_vision:
        parts.extend(
            [
                context.detected_product or "",
                *context.detected_components,
                *context.visible_objects,
                context.visual_summary,
            ]
        )
    if include_ocr:
        parts.extend([*context.detected_codes, *context.detected_numbers, context.ocr_text])
    return " ".join(dict.fromkeys(item.strip() for item in parts if item.strip()))[:1200]
