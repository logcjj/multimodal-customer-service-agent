from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from app.config.runtime import RuntimeSettings
from app.contracts.models import AgentRequest, AgentResult, Claim, Evidence, ModelKind
from app.models.llm_gateway import LLMGateway
from app.multimodal.visual_context import (
    empty_ocr,
    empty_vision,
    image_hashes,
    merge_visual_context,
    parse_ocr_output,
    parse_vlm_output,
)


class MultimodalAgent:
    id = "multimodal"

    def __init__(
        self,
        llm_gateway: LLMGateway | None = None,
        *,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self.llm_gateway = llm_gateway
        self.settings = settings or RuntimeSettings.from_env()

    def run(
        self,
        request: AgentRequest,
        event_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> AgentResult:
        started = perf_counter()
        image_count = len(request.images)
        ocr_output = None
        vision_output = None

        if self.settings.ocr_pipeline == "off":
            ocr = empty_ocr("disabled")
        elif not self.llm_gateway or not self.llm_gateway.available(ModelKind.OCR):
            ocr = empty_ocr("unavailable")
        else:
            self._emit(event_sink, "ocr.started", "running", "识别图片可见文字、错误码与铭牌字段")
            ocr_output = self.llm_gateway.generate(
                kind=ModelKind.OCR,
                system_prompt=(
                    "你是高精度 OCR 结构化提取器。只抄录图片中实际可见的内容，不猜测。"
                    "仅输出 JSON 对象，字段为 visible_text、codes、numbers、label_fields、confidence。"
                ),
                user_prompt=(
                    f"用户问题：{request.question}\n"
                    "逐字保留错误码、型号、序列号、数值和单位；看不清的字段不要输出。"
                ),
                images=request.images,
                temperature=0,
                max_tokens=800,
            )
            ocr = parse_ocr_output(ocr_output.text) if ocr_output else empty_ocr("provider_error")
            self._emit(
                event_sink,
                "ocr.completed",
                "completed" if ocr.status in {"ok", "empty"} else "failed",
                "OCR 结构化识别完成",
                payload={"provider_status": ocr.status, "confidence": ocr.confidence},
            )

        if not self.llm_gateway or not self.llm_gateway.available(ModelKind.VLM):
            vision = empty_vision("unavailable")
        else:
            self._emit(event_sink, "vlm.started", "running", "识别产品、部件和可见异常")
            vision_output = self.llm_gateway.generate(
                kind=ModelKind.VLM,
                system_prompt=(
                    "你是视觉检查智能体。只描述图片中可见的产品、部件、对象和异常，不推断不可见事实。"
                    "product 字段必须填写该物品所属主设备品类（例如洗碗机、烤箱、空调遥控器），"
                    "不要把颜色、形状或配件外观直接当作产品名；无法判断时填 null。"
                    "components 填写可见配件或部件的通用名称，不能确认的名称使用‘疑似’限定。"
                    "遇到显示屏图标时，同时记录图标外形；若能根据通用符号判断用途，可补充带‘疑似’的疑似功能名称，"
                    "但不得把推测写成确定事实。"
                    "遇到编号、箭头或流程图时，summary 必须逐项描述箭头的起点、终点和移动方向，"
                    "不要把箭头指向的对象误写成编号本身的含义，也不要把连接动作猜成拆卸动作。"
                    "仅输出 JSON 对象，字段为 product、components、visible_objects、summary、confidence。"
                ),
                user_prompt=f"用户问题：{request.question}\n请输出简洁的结构化可见事实。",
                images=request.images,
                temperature=0,
                max_tokens=600,
            )
            vision = parse_vlm_output(vision_output.text) if vision_output else empty_vision("provider_error")
            self._emit(
                event_sink,
                "vlm.completed",
                "completed" if vision.status in {"ok", "empty"} else "failed",
                "视觉结构化观察完成",
                payload={"provider_status": vision.status, "confidence": vision.confidence},
            )

        context = merge_visual_context(
            image_hashes=image_hashes(request.images),
            ocr=ocr,
            vision=vision,
        )
        evidence: list[Evidence] = []
        if ocr.status == "ok":
            ocr_text = ocr.visible_text or " ".join([*ocr.codes, *ocr.numbers, *ocr.label_fields.values()])
            evidence.append(
                Evidence(
                    evidence_id="ocr:user-image",
                    source_type="ocr",
                    title="用户图片 OCR",
                    text=ocr_text,
                    retrieval_stage="user_image_ocr",
                    evidence_confidence=ocr.confidence,
                )
            )
        if vision.status == "ok":
            visual_text = vision.summary or "、".join(
                item for item in [vision.product or "", *vision.components, *vision.visible_objects] if item
            )
            evidence.append(
                Evidence(
                    evidence_id="vision:user-image",
                    source_type="vision",
                    title="用户图片视觉观察",
                    text=visual_text,
                    product=vision.product,
                    retrieval_stage="user_image_vlm",
                    evidence_confidence=vision.confidence,
                )
            )
        model_names = [item.model for item in (ocr_output, vision_output) if item is not None]
        evidence_ids = [item.evidence_id for item in evidence]
        return AgentResult(
            task_id="vision-1",
            agent_id=self.id,
            status="completed",
            answer_fragment="",
            claims=(
                [Claim(text=f"已从本轮 {image_count} 张图片提取可见信息。", evidence_ids=evidence_ids)]
                if evidence_ids
                else []
            ),
            evidence=evidence,
            confidence=context.confidence,
            recommended_next_action="结合用户文字检索说明书",
            latency_ms=round((perf_counter() - started) * 1000),
            llm_generated=bool(model_names),
            model_used=" + ".join(model_names) or None,
            visual_context=context,
        )

    @staticmethod
    def _emit(
        event_sink: Callable[[dict[str, object]], None] | None,
        event_type: str,
        status: str,
        label: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> None:
        if event_sink is not None:
            event_sink(
                {
                    "type": event_type,
                    "agent_id": "multimodal",
                    "status": status,
                    "label": label,
                    "summary": "",
                    "payload": payload or {},
                }
            )
