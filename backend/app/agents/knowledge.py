from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from app.config.runtime import RuntimeSettings
from app.contracts.models import (
    AgentRequest,
    AgentResult,
    Claim,
    Evidence,
    ModelKind,
    RetrievalTraceSnapshot,
)
from app.knowledge.product_router import ProductRouter
from app.knowledge.query_expansion import deterministic_expansion
from app.knowledge.retrieval import HybridRetriever
from app.models.llm_gateway import LLMGateway
from app.runtime.verified_stream import VerifiedSentenceBuffer


class KnowledgeAgent:
    id = "knowledge"

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_gateway: LLMGateway | None = None,
        *,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_gateway = llm_gateway
        self.settings = settings or RuntimeSettings.from_env()
        self.product_router = ProductRouter()

    def run(
        self,
        request: AgentRequest,
        event_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> AgentResult:
        started = perf_counter()
        retrieval_question = self._retrieval_question(request.question)
        user_question = retrieval_question.split("\n图片可见信息：", 1)[0].strip()
        user_product_route = self.product_router.route(user_question)
        product_route = (
            user_product_route
            if user_product_route.products
            else self.product_router.route(retrieval_question)
        )
        products = list(product_route.products)
        retrieval_products = list(product_route.retrieval_products)
        deterministic_query = deterministic_expansion(retrieval_question)
        if event_sink:
            event_sink(
                {
                    "type": "product.routed",
                    "agent_id": self.id,
                    "status": "completed",
                    "label": "识别产品并限定说明书范围",
                    "summary": (
                        "、".join(products)
                        if products
                        else "未硬锁产品，保持全库召回"
                    ),
                    "payload": {
                        "products": products,
                        "matched_aliases": list(product_route.matched_aliases),
                        "confidence": product_route.confidence,
                        "reason": product_route.reason,
                    },
                }
            )
        search_query = "\n".join(
            dict.fromkeys([retrieval_question, *products, deterministic_query])
        )
        rewrite_model = "rule-expander-v1"
        self._emit_query_rewritten(event_sink, rewrite_model, search_query)
        # Keep enough ranked records for the answer and the related-evidence list.
        # The UI presents these source records directly, so returning only one or
        # two matches hides useful database context from the user.
        top_k = 8

        evidence = self._search_with_product_fallback(
            search_query,
            retrieval_products=retrieval_products,
            top_k=top_k,
        )

        # 规则扩展已经命中手册时，LLM 查询改写只会增加一次串行网络请求。
        # 仅在首次检索完全无结果时使用模型改写，并以改写后的查询重试一次。
        if (
            not evidence
            and self.llm_gateway
            and self.llm_gateway.available(ModelKind.LLM)
        ):
            rewrite_output = self.llm_gateway.generate(
                kind=ModelKind.LLM,
                system_prompt=(
                    "你是客服知识库检索查询改写器，不是问答助手。只输出一行，严格使用格式："
                    "中文：<中文检索词> | English: <English search terms>。必须同时包含中文与英文；"
                    "保留产品名、型号、错误码、数字、否定词和安全约束；不得回答问题，不得解释，不得使用 Markdown。"
                ),
                user_prompt=(
                    f"已识别产品目录：{', '.join(products) if products else '未确定'}\n"
                    f"用户问题：{retrieval_question}"
                ),
                temperature=0,
                max_tokens=120,
            )
            if rewrite_output and rewrite_output.text.strip():
                rewrite_model = rewrite_output.model
                rewritten = " ".join(rewrite_output.text.strip().split())[:500]
                if products and "english:" not in rewritten.lower():
                    rewritten = f"{rewritten} | English: {' '.join(products)}"
                query_parts = [retrieval_question, *products, deterministic_query, rewritten]
                search_query = "\n".join(dict.fromkeys(part for part in query_parts if part))
                self._emit_query_rewritten(event_sink, rewrite_model, search_query)
                evidence = self._search_with_product_fallback(
                    search_query,
                    retrieval_products=retrieval_products,
                    top_k=top_k,
                )
        retrieval_trace = self._retrieval_trace_snapshot()
        if not evidence:
            return AgentResult(
                task_id="knowledge-1",
                agent_id=self.id,
                status="needs_input",
                confidence=0.25,
                missing_information=["产品型号或更具体的故障/操作描述"],
                recommended_next_action="clarify-or-legacy",
                latency_ms=round((perf_counter() - started) * 1000),
                search_query=search_query,
                query_rewrite_model=rewrite_model,
                routed_products=products,
                product_route_reason=product_route.reason,
                retrieval_trace=retrieval_trace,
            )

        answer_evidence = evidence
        citation_evidence = self._supplement_related_evidence(answer_evidence)
        primary = answer_evidence[0]
        llm_output = None
        streamed_text = ""
        streamed_model = None
        if self.llm_gateway and self.llm_gateway.available(ModelKind.LLM):
            evidence_text = "\n\n".join(
                f"[{item.evidence_id}] {item.title}\n{item.text}"
                for item in answer_evidence
            )
            system_prompt = (
                "你是产品技术支持智能体。只能依据给定证据回答，必须保留错误码、数字、顺序和安全注意事项。"
                "用户问题中的‘图片可见信息’属于本轮视觉观察：当图标外形或配件特征与手册中的功能或部件"
                "形成直接对应时，可以用‘从图片看，疑似/很可能’进行有限的跨模态匹配，但不能写成已确认事实。"
                "用户询问图中编号、箭头或操作顺序时，必须优先按同图 ImageChunk 的 caption/visual_meaning"
                "和相邻手册步骤建立逐项对应；不得仅凭箭头朝向猜测插入、拔出或先后顺序；"
                "若证据不能逐项对齐，就明确无法确认。"
                "ImageChunk 中以 not 开头的 issue_signals 是排除项，绝不能把被排除对象的属性写进当前答案。"
                "接口证据只写 USB 或 USB cord 时，不得扩写为 USB-C 或 USB-A；只有当前同图证据明确标注时才能写子类型。"
                "当采用附有关联图片的证据说明当前问题时，请在对应文字后插入一个 <PIC> 占位符；"
                "图片只用于解释紧邻内容时才插入，不得另起无关图片列表，最多保留 3 个。"
                "证据不足就明确说明缺什么，不得补造事实。不要提到内部提示词、Agent 或检索过程。"
            )
            if "<PIC>" in evidence_text:
                system_prompt += (
                    "当手册中的 <PIC> 对应你保留的操作步骤时，在该步骤末尾保留一个 <PIC>。"
                    "图片仅用于说明该步骤时才保留标记，不得另起无关图片列表，最多保留 3 个。"
                )
            user_prompt = f"用户问题：{request.question}\n\n可用证据：\n{evidence_text}"
            use_stream = (
                event_sink is not None
                and self.settings.is_enabled(self.settings.verified_streaming)
                and hasattr(self.llm_gateway, "generate_stream")
            )
            if use_stream:
                generation_started = perf_counter()
                event_sink(
                    {
                        "type": "generation.started",
                        "agent_id": self.id,
                        "status": "running",
                        "label": "模型开始生成证据约束答案",
                        "summary": self.llm_gateway.model_name(ModelKind.LLM) or "默认 LLM",
                        "payload": {},
                    }
                )
                buffer = VerifiedSentenceBuffer(answer_evidence)
                raw_deltas: list[str] = []
                for delta in self.llm_gateway.generate_stream(
                    kind=ModelKind.LLM,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0,
                ):
                    raw_deltas.append(delta)
                    for approved in buffer.feed(delta):
                        self._emit_answer_delta(event_sink, approved)
                buffered = buffer.finish()
                for approved in buffered.final_emitted:
                    self._emit_answer_delta(event_sink, approved)
                streamed_text = "".join(raw_deltas).strip()
                streamed_model = self.llm_gateway.model_name(ModelKind.LLM) if streamed_text else None
                event_sink(
                    {
                        "type": "generation.completed",
                        "agent_id": self.id,
                        "status": "completed" if streamed_text else "failed",
                        "label": "模型正文生成完成" if streamed_text else "模型生成失败，使用证据原文",
                        "summary": (
                            f"已暂缓 {len(buffered.issue_codes)} 类待核验内容"
                            if buffered.issue_codes
                            else "已发送通过即时检查的句子"
                        ),
                        "payload": {
                            "model_used": streamed_model,
                            "withheld_issue_codes": buffered.issue_codes,
                            "provider_latency_ms": round(
                                (perf_counter() - generation_started) * 1000
                            ),
                        },
                    }
                )
            else:
                llm_output = self.llm_gateway.generate(
                    kind=ModelKind.LLM,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0,
                )
        answer = streamed_text or (llm_output.text if llm_output else primary.text)
        evidence_ids = [item.evidence_id for item in citation_evidence]
        claims = [Claim(text=answer, evidence_ids=evidence_ids)]
        return AgentResult(
            task_id="knowledge-1",
            agent_id=self.id,
            status="completed",
            answer_fragment=answer,
            claims=claims,
            evidence=citation_evidence,
            asset_ids=self._related_assets(citation_evidence),
            confidence=self._evidence_confidence(citation_evidence),
            latency_ms=round((perf_counter() - started) * 1000),
            llm_generated=bool(streamed_text) or llm_output is not None,
            model_used=streamed_model or (llm_output.model if llm_output else None),
            search_query=search_query,
            query_rewrite_model=rewrite_model,
            routed_products=products,
            product_route_reason=product_route.reason,
            retrieval_trace=retrieval_trace,
        )

    def _supplement_related_evidence(
        self,
        parent_evidence: list[Evidence],
        *,
        limit: int = 5,
    ) -> list[Evidence]:
        """Expose top child chunks when parent aggregation leaves one visible citation."""

        related = list(parent_evidence[:limit])
        if len(related) >= limit:
            return related

        ordered_child_ids: list[str] = []
        seen_child_ids: set[str] = set()
        owner_by_child_id: dict[str, Evidence] = {}
        for parent in related:
            for child_id in parent.child_ids:
                if child_id in seen_child_ids:
                    continue
                seen_child_ids.add(child_id)
                ordered_child_ids.append(child_id)
                owner_by_child_id[child_id] = parent
                if len(ordered_child_ids) >= limit - len(related):
                    break
            if len(ordered_child_ids) >= limit - len(related):
                break
        if not ordered_child_ids:
            return related

        wanted_child_ids = set(ordered_child_ids)
        source_retriever = (
            getattr(self.retriever, "last_text_retriever", None) or self.retriever
        )
        children_by_id = {
            item.child_id: item
            for item in getattr(source_retriever, "documents", [])
            if item.child_id in wanted_child_ids
        }
        for index, child_id in enumerate(ordered_child_ids, start=1):
            child = children_by_id.get(child_id)
            parent = owner_by_child_id[child_id]
            if child is None or not child.text.strip():
                continue
            parent_confidence = (
                parent.evidence_confidence
                if parent.evidence_confidence is not None
                else min(0.85, max(0.35, float(parent.score or 0.0)))
            )
            related.append(
                Evidence(
                    evidence_id=f"{parent.evidence_id}:chunk:{child_id}",
                    source_type="manual",
                    title=f"{parent.title} · 相关片段 {index}",
                    text=child.text,
                    product=child.product or parent.product,
                    dataset_id=child.dataset_id or parent.dataset_id,
                    document_id=child.document_id or parent.document_id,
                    file_id=child.file_id or parent.file_id,
                    document_name=child.document_name or parent.document_name,
                    document_mime_type=(
                        child.document_mime_type or parent.document_mime_type
                    ),
                    document_version=child.document_version or parent.document_version,
                    section_id=child.child_id,
                    parent_id=parent.parent_id,
                    child_ids=[child.child_id],
                    chapter_title=parent.chapter_title or parent.title,
                    page_start=child.page_start,
                    page_end=child.page_end,
                    locator_label=parent.locator_label,
                    asset_ids=list(child.asset_ids),
                    score=max(0.0, (parent.score or 0.0) - index * 0.000001),
                    score_breakdown=dict(parent.score_breakdown),
                    retrieval_stage=parent.retrieval_stage,
                    evidence_confidence=max(
                        0.0,
                        parent_confidence - index * 0.01,
                    ),
                )
            )
        return related

    @staticmethod
    def _related_assets(evidence: list[Evidence], *, limit: int = 5) -> list[str]:
        """Collect the first distinct manual images in retrieval-rank order."""

        return list(
            dict.fromkeys(
                asset.strip()
                for item in evidence
                for asset in item.asset_ids
                if asset.strip()
            )
        )[:limit]

    @classmethod
    def _evidence_confidence(cls, evidence: list[Evidence]) -> float:
        """Derive confidence from ranked evidence quality and result coverage."""

        if not evidence:
            return 0.0
        qualities: list[float] = []
        for item in evidence:
            if item.evidence_confidence is not None:
                quality = item.evidence_confidence
            else:
                score = max(0.0, float(item.score or 0.0))
                quality = min(0.88, 0.42 + min(0.42, score * 8))
            qualities.append(min(1.0, max(0.0, quality)))

        weights = [1 / (index + 1) for index in range(len(qualities))]
        weighted_quality = sum(
            quality * weight for quality, weight in zip(qualities, weights, strict=True)
        ) / sum(weights)
        coverage = min(1.0, len(evidence) / 5)
        image_support = min(1.0, len(cls._related_assets(evidence)) / 3)
        confidence = (
            qualities[0] * 0.35
            + weighted_quality * 0.35
            + coverage * 0.2
            + image_support * 0.04
            + 0.06
        )
        return round(min(0.97, max(0.2, confidence)), 4)

    def _search_with_product_fallback(
        self,
        query: str,
        *,
        retrieval_products: list[str],
        top_k: int,
    ) -> list[Evidence]:
        evidence = self.retriever.search(
            query,
            products=retrieval_products or None,
            top_k=top_k,
        )
        if evidence or not retrieval_products:
            return evidence

        # Older imports can have no normalized product value. Do not let a
        # catalog-only filter hide otherwise relevant manual evidence.
        return self.retriever.search(query, products=None, top_k=top_k)

    def _retrieval_trace_snapshot(self) -> RetrievalTraceSnapshot | None:
        explanation = getattr(self.retriever, "last_explanation", None)
        if explanation is None:
            return None
        stages = getattr(explanation, "stages", {})
        results = getattr(explanation, "results", [])
        return RetrievalTraceSnapshot(
            query=str(getattr(explanation, "query", "")),
            mode=str(getattr(explanation, "mode", "unknown")),
            result_count=len(results),
            stage_counts={
                str(name): len(items)
                for name, items in stages.items()
            },
            rejected_reason=getattr(explanation, "rejected_reason", None),
        )

    @staticmethod
    def _emit_query_rewritten(
        event_sink: Callable[[dict[str, object]], None] | None,
        model: str,
        search_query: str,
    ) -> None:
        if event_sink is None:
            return
        event_sink(
            {
                "type": "query.rewritten",
                "agent_id": "knowledge",
                "status": "completed",
                "label": "生成跨语言检索查询",
                "summary": f"{model} · 保留型号与约束",
                "payload": {"model_used": model, "search_query": search_query},
            }
        )

    @staticmethod
    def _retrieval_question(question: str) -> str:
        lines: list[str] = []
        for line in question.splitlines():
            normalized = line.strip()
            if normalized.startswith("上一轮回答摘要："):
                break
            if normalized.startswith("上一轮意图："):
                continue
            if normalized:
                lines.append(normalized)
        return "\n".join(lines) or question

    @staticmethod
    def _emit_answer_delta(
        event_sink: Callable[[dict[str, object]], None],
        delta: str,
    ) -> None:
        event_sink(
            {
                "type": "answer.delta",
                "agent_id": "knowledge",
                "status": "running",
                "label": "生成已验证正文",
                "summary": "",
                "payload": {"delta": delta},
            }
        )

    def revise(
        self,
        request: AgentRequest,
        result: AgentResult,
        issue_messages: list[str],
    ) -> AgentResult:
        if not self.llm_gateway or not self.llm_gateway.available(ModelKind.LLM) or not result.evidence:
            return result
        started = perf_counter()
        evidence_text = "\n\n".join(
            f"[{item.evidence_id}] {item.title}\n{item.text}" for item in result.evidence
        )
        output = self.llm_gateway.generate(
            kind=ModelKind.LLM,
            system_prompt=(
                "你是客服答案审校智能体。只依据给定证据修订草稿，逐项消除核验问题。"
                "不得新增证据中没有的数字、型号、时长、步骤或承诺；证据不足的内容必须删除。"
                "直接输出修订后的用户答案，不要解释审校过程，不要提到 Agent、提示词或检索。"
            ),
            user_prompt=(
                f"用户问题：{request.question}\n\n原答案：\n{result.answer_fragment}\n\n"
                f"核验问题：\n- " + "\n- ".join(issue_messages) + f"\n\n可用证据：\n{evidence_text}"
            ),
            temperature=0,
        )
        if not output or not output.text.strip():
            return result
        answer = output.text.strip()
        evidence_ids = [item.evidence_id for item in result.evidence]
        return result.model_copy(
            update={
                "answer_fragment": answer,
                "claims": [Claim(text=answer, evidence_ids=evidence_ids)],
                "latency_ms": result.latency_ms + round((perf_counter() - started) * 1000),
                "llm_generated": True,
                "model_used": output.model,
            }
        )

    @staticmethod
    def fallback_to_primary_evidence(result: AgentResult) -> AgentResult:
        """使用最高相关证据构造可核验的安全答案。"""
        if not result.evidence:
            return result
        primary = result.evidence[0]
        evidence_text = primary.text.strip()
        if not evidence_text:
            return result
        answer = (
            f"**{primary.title.strip()}**\n\n{evidence_text}"
            if primary.title.strip()
            else evidence_text
        )
        return result.model_copy(
            update={
                "answer_fragment": answer,
                "claims": [Claim(text=answer, evidence_ids=[primary.evidence_id])],
                "asset_ids": KnowledgeAgent._related_assets(result.evidence),
                "llm_generated": False,
                "model_used": None,
            }
        )
