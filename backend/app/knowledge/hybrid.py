from __future__ import annotations

import math
import re
from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field

import jieba

from app.contracts.models import Evidence
from app.runtime.error_codes import extract_normalized_error_codes, normalize_error_code


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    exact = re.findall(r"[a-z]+\d+[a-z0-9-]*|\d+(?:\.\d+)?(?:v|w|a|hz|mm|cm)?", lowered)
    words = [item.strip() for item in jieba.lcut(lowered, cut_all=False) if item.strip()]
    words = [item for item in words if re.search(r"[\w\u4e00-\u9fff]", item)]
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    bigrams = [run[index : index + 2] for run in cjk_runs for index in range(len(run) - 1)]
    return list(dict.fromkeys([*exact, *words, *bigrams]))


@dataclass(frozen=True)
class IndexedChild:
    child_id: str
    parent_id: str
    dataset_id: str
    document_id: str
    document_version: str
    title: str
    text: str
    page_start: int
    page_end: int
    file_id: str | None = None
    document_name: str | None = None
    document_mime_type: str | None = None
    parent_text: str | None = None
    product: str | None = None
    asset_ids: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass(frozen=True)
class ScoreBreakdown:
    lexical: float = 0.0
    dense: float = 0.0
    rrf: float = 0.0
    rerank: float = 0.0
    parent: float = 0.0


@dataclass(frozen=True)
class ParentRetrievalResult:
    parent_id: str
    dataset_id: str
    document_id: str
    document_version: str
    file_id: str | None
    document_name: str | None
    document_mime_type: str | None
    title: str
    text: str
    product: str | None
    page_start: int
    page_end: int
    asset_ids: list[str]
    matched_children: list[str]
    scores: ScoreBreakdown


@dataclass(frozen=True)
class RetrievalExplanation:
    query: str
    mode: str
    results: list[ParentRetrievalResult]
    stages: dict[str, list[dict[str, object]]]
    rejected_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


EmbedOverride = Callable[[list[str]], list[list[float]]]
RerankOverride = Callable[[str, list[str]], list[float]]


class PublishedHybridRetriever:
    def __init__(
        self,
        documents: list[IndexedChild],
        *,
        embed: EmbedOverride | None = None,
        rerank: RerankOverride | None = None,
        min_score: float = 0.012,
        rrf_k: int = 60,
        lexical_top_k: int = 20,
        dense_top_k: int = 20,
        rerank_top_k: int = 12,
    ) -> None:
        self.documents = documents
        self.embed = embed
        self.rerank = rerank
        self.min_score = min_score
        self.rrf_k = rrf_k
        self.lexical_top_k = lexical_top_k
        self.dense_top_k = dense_top_k
        self.rerank_top_k = rerank_top_k

        # Store each token string once. Document indexes use sorted uint32 IDs
        # so lexical membership remains fast without retaining millions of
        # duplicate Python strings.
        self._token_ids: dict[str, int] = {}
        self._tokens: dict[str, array[int]] = {
            item.child_id: self._encode_document_tokens(
                f"{item.document_name or ''} {item.product or ''} "
                f"{item.title} {item.text} {' '.join(item.keywords)}"
            )
            for item in documents
        }
        self._normalized_codes: dict[str, set[str]] = {}
        self._legacy_exact_codes: dict[str, set[str]] = {}
        # Every child of a parent carries the same parent text in the offline
        # bundle. Parse that text once and reuse its exact-code sets.
        parent_code_cache: dict[str, tuple[set[str], set[str]]] = {}
        for item in documents:
            child_identifier = (
                f"{item.document_name or ''} {item.product or ''} "
                f"{item.title} {item.text} {' '.join(item.keywords)}"
            )
            normalized_codes = set(extract_normalized_error_codes(child_identifier))
            legacy_codes = self._unrecognized_exact_codes(child_identifier)
            if item.parent_text:
                parent_codes = parent_code_cache.get(item.parent_id)
                if parent_codes is None:
                    parent_codes = (
                        set(extract_normalized_error_codes(item.parent_text)),
                        self._unrecognized_exact_codes(item.parent_text),
                    )
                    parent_code_cache[item.parent_id] = parent_codes
                normalized_codes.update(parent_codes[0])
                legacy_codes.update(parent_codes[1])
            self._normalized_codes[item.child_id] = normalized_codes
            self._legacy_exact_codes[item.child_id] = legacy_codes
        self._last_explanation: ContextVar[RetrievalExplanation | None] = (
            ContextVar(
                f"published_hybrid_last_explanation_{id(self)}",
                default=None,
            )
        )

    @property
    def last_explanation(self) -> RetrievalExplanation | None:
        return self._last_explanation.get()

    @last_explanation.setter
    def last_explanation(self, value: RetrievalExplanation | None) -> None:
        self._last_explanation.set(value)

    def search(self, query: str, products: list[str] | None = None, top_k: int = 5) -> list[Evidence]:
        explanation = self.explain(query, products=products, top_n=top_k)
        return [
            Evidence(
                evidence_id=f"evidence:{item.parent_id}",
                source_type="manual",
                title=item.title,
                text=item.text,
                product=item.product,
                dataset_id=item.dataset_id,
                document_id=item.document_id,
                file_id=item.file_id,
                document_name=item.document_name,
                document_mime_type=item.document_mime_type,
                document_version=item.document_version,
                section_id=item.parent_id,
                parent_id=item.parent_id,
                child_ids=item.matched_children,
                chapter_title=item.title,
                page_start=item.page_start,
                page_end=item.page_end,
                locator_label=(
                    f"导入章节 {item.page_start}"
                    if item.document_version.startswith("v6-import")
                    else (
                        f"第 {item.page_start} 页"
                        if item.page_start == item.page_end
                        else f"第 {item.page_start}-{item.page_end} 页"
                    )
                ),
                asset_ids=item.asset_ids,
                score=round(item.scores.parent, 6),
                score_breakdown={
                    "lexical": item.scores.lexical,
                    "dense": item.scores.dense,
                    "rrf": item.scores.rrf,
                    "rerank": item.scores.rerank,
                    "parent": item.scores.parent,
                },
                retrieval_stage=explanation.mode,
                evidence_confidence=round(
                    min(0.99, 0.5 + min(0.4, item.scores.parent * 8) + item.scores.rerank * 0.05),
                    4,
                ),
            )
            for item in explanation.results
        ]

    def explain(
        self,
        query: str,
        *,
        products: list[str] | None = None,
        dataset_ids: list[str] | None = None,
        top_n: int = 5,
        min_score: float | None = None,
        use_rerank: bool = True,
        query_vector: list[float] | None = None,
    ) -> RetrievalExplanation:
        warnings: list[str] = []
        allowed_products = set(products or [])
        allowed_datasets = set(dataset_ids or [])
        documents = [
            item
            for item in self.documents
            if (not allowed_products or item.product in allowed_products)
            and (not allowed_datasets or item.dataset_id in allowed_datasets)
        ]
        query_tokens = tokenize(query)
        normalized_codes = set(extract_normalized_error_codes(query))
        legacy_exact_codes = self._unrecognized_exact_codes(query)
        lexical_scores = self._bm25(
            query_tokens,
            documents,
            normalized_codes=normalized_codes,
            legacy_exact_codes=legacy_exact_codes,
        )
        lexical_ranked = sorted(lexical_scores, key=lexical_scores.get, reverse=True)[: self.lexical_top_k]

        dense_scores: dict[str, float] = {}
        embedded_documents = [item for item in documents if item.embedding]
        resolved_query_vector = query_vector
        if resolved_query_vector is None and self.embed and embedded_documents:
            try:
                query_vectors = self.embed([query])
            except Exception:
                query_vectors = []
            if len(query_vectors) == 1 and query_vectors[0]:
                resolved_query_vector = query_vectors[0]
            else:
                resolved_query_vector = []
        if embedded_documents and resolved_query_vector == []:
            warnings.append("查询向量生成失败，已降级为 BM25/RRF；红色查询点暂不可用。")
        if resolved_query_vector and embedded_documents:
            dense_scores = {
                item.child_id: max(0.0, self._cosine(resolved_query_vector, item.embedding or []))
                for item in embedded_documents
            }
        dense_ranked = sorted(
            (child_id for child_id, score in dense_scores.items() if score > 0),
            key=dense_scores.get,
            reverse=True,
        )[: self.dense_top_k]

        fused: dict[str, float] = defaultdict(float)
        for ranking in (lexical_ranked, dense_ranked):
            for rank, child_id in enumerate(ranking, start=1):
                fused[child_id] += 1 / (self.rrf_k + rank)
        fused_ranked = sorted(fused, key=fused.get, reverse=True)[: max(self.lexical_top_k, self.dense_top_k)]

        rerank_scores: dict[str, float] = {}
        doc_by_id = {item.child_id: item for item in documents}
        if use_rerank and self.rerank and fused_ranked:
            rerank_candidates = fused_ranked[: self.rerank_top_k]
            try:
                returned = self.rerank(query, [doc_by_id[item].text for item in rerank_candidates])
            except Exception:
                returned = []
            if len(returned) == len(rerank_candidates):
                rerank_scores = {
                    child_id: float(score)
                    for child_id, score in zip(rerank_candidates, returned, strict=True)
                }
                final_children = sorted(
                    rerank_candidates,
                    key=lambda item: rerank_scores.get(item, 0),
                    reverse=True,
                )
                mode = "hybrid-rerank"
            else:
                final_children = fused_ranked
                mode = "hybrid" if dense_ranked else "lexical-only"
                warnings.append("Rerank 调用失败，当前使用 RRF 排序。")
        else:
            final_children = fused_ranked
            mode = "hybrid" if dense_ranked else "lexical-only"

        grouped: dict[str, list[str]] = defaultdict(list)
        for child_id in final_children:
            grouped[doc_by_id[child_id].parent_id].append(child_id)

        parent_results: list[ParentRetrievalResult] = []
        for parent_id, child_ids in grouped.items():
            matched = [doc_by_id[item] for item in child_ids]
            best_id = child_ids[0]
            best = matched[0]
            best_lexical = max(lexical_scores.get(item, 0.0) for item in child_ids)
            best_dense = max(dense_scores.get(item, 0.0) for item in child_ids)
            best_rrf = max(fused.get(item, 0.0) for item in child_ids)
            best_rerank = max(rerank_scores.get(item, 0.0) for item in child_ids)
            coverage_bonus = min(0.006, max(0, len(child_ids) - 1) * 0.002)
            # 型号和错误码应优先于同产品下的泛化章节，避免多 Child 覆盖奖励把
            # 精确故障条目挤到后面。
            exact_match = any(
                normalized_codes & self._normalized_codes.get(child_id, set())
                or legacy_exact_codes & self._legacy_exact_codes.get(child_id, set())
                for child_id in child_ids
            )
            exact_bonus = 0.012 if exact_match else 0.0
            parent_score = best_rrf + coverage_bonus + exact_bonus + best_rerank * 0.004
            text = best.parent_text or "\n".join(dict.fromkeys(item.text for item in matched))
            parent_results.append(
                ParentRetrievalResult(
                    parent_id=parent_id,
                    dataset_id=best.dataset_id,
                    document_id=best.document_id,
                    document_version=best.document_version,
                    file_id=best.file_id,
                    document_name=best.document_name,
                    document_mime_type=best.document_mime_type,
                    title=best.title,
                    text=text,
                    product=best.product,
                    page_start=min(item.page_start for item in matched),
                    page_end=max(item.page_end for item in matched),
                    asset_ids=list(dict.fromkeys(asset for item in matched for asset in item.asset_ids)),
                    matched_children=child_ids,
                    scores=ScoreBreakdown(
                        lexical=round(best_lexical, 6),
                        dense=round(best_dense, 6),
                        rrf=round(best_rrf, 6),
                        rerank=round(best_rerank, 6),
                        parent=round(parent_score, 6),
                    ),
                )
            )
        threshold = self.min_score if min_score is None else min_score
        parent_results = [item for item in parent_results if item.scores.parent >= threshold]
        exact_codes = normalized_codes | legacy_exact_codes
        exact_code_rejected = False
        if exact_codes:
            before_exact_gate = len(parent_results)
            parent_results = [
                item
                for item in parent_results
                if normalized_codes.issubset(
                    set().union(
                        *(
                            self._normalized_codes.get(child_id, set())
                            for child_id in item.matched_children
                        )
                    )
                )
                and legacy_exact_codes.issubset(
                    set().union(
                        *(
                            self._legacy_exact_codes.get(child_id, set())
                            for child_id in item.matched_children
                        )
                    )
                )
            ]
            exact_code_rejected = before_exact_gate > 0 and not parent_results
            if exact_code_rejected:
                warnings.append("候选证据未包含查询中的精确型号或错误码，已拒绝语义近似结果。")
        parent_results.sort(key=lambda item: item.scores.parent, reverse=True)
        parent_results = parent_results[:top_n]
        stages = {
            "lexical": self._stage(lexical_ranked, lexical_scores, doc_by_id),
            "dense": self._stage(dense_ranked, dense_scores, doc_by_id),
            "rrf": self._stage(fused_ranked, fused, doc_by_id),
            "rerank": self._stage(sorted(rerank_scores, key=rerank_scores.get, reverse=True), rerank_scores, doc_by_id),
            "parent": [
                {"id": item.parent_id, "title": item.title, "score": item.scores.parent}
                for item in parent_results
            ],
        }
        explanation = RetrievalExplanation(
            query=query,
            mode=mode,
            results=parent_results,
            stages=stages,
            rejected_reason=(
                None
                if parent_results
                else (
                    "未找到包含指定型号或错误码的证据"
                    if exact_code_rejected
                    else "没有达到阈值的证据"
                )
            ),
            warnings=warnings,
        )
        self.last_explanation = explanation
        return explanation

    def _bm25(
        self,
        query_tokens: list[str],
        documents: list[IndexedChild],
        *,
        normalized_codes: set[str],
        legacy_exact_codes: set[str],
    ) -> dict[str, float]:
        if not query_tokens or not documents:
            return {}
        token_lists = [self._tokens[item.child_id] for item in documents]
        average_length = sum(len(tokens) for tokens in token_lists) / max(1, len(token_lists))
        query_token_ids = [
            token_id
            for token in query_tokens
            if (token_id := self._token_ids.get(token)) is not None
        ]
        matched_tokens: dict[str, list[int]] = {}
        frequencies: Counter[int] = Counter()
        for document, tokens in zip(documents, token_lists, strict=True):
            matched = [
                token_id
                for token_id in query_token_ids
                if self._contains_token(tokens, token_id)
            ]
            matched_tokens[document.child_id] = matched
            frequencies.update(matched)

        scores: dict[str, float] = {}
        for document, tokens in zip(documents, token_lists, strict=True):
            overlap = matched_tokens[document.child_id]
            exact = (
                normalized_codes & self._normalized_codes.get(document.child_id, set())
            ) | (
                legacy_exact_codes
                & self._legacy_exact_codes.get(document.child_id, set())
            )
            if len(overlap) < 2 and not exact:
                continue
            score = 0.0
            for token_id in overlap:
                document_frequency = frequencies[token_id]
                inverse = math.log(1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = 1 + 1.5 * (
                    1 - 0.75 + 0.75 * len(tokens) / max(1, average_length)
                )
                score += inverse * 2.5 / denominator
            score += len(exact) * 3.0
            if score > 0:
                scores[document.child_id] = score
        return scores

    def _encode_document_tokens(self, text: str) -> array[int]:
        encoded: set[int] = set()
        for token in tokenize(text):
            token_id = self._token_ids.get(token)
            if token_id is None:
                token_id = len(self._token_ids)
                self._token_ids[token] = token_id
            encoded.add(token_id)
        return array("I", sorted(encoded))

    @staticmethod
    def _contains_token(tokens: array[int], token_id: int) -> bool:
        position = bisect_left(tokens, token_id)
        return position < len(tokens) and tokens[position] == token_id

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return 0.0
        denominator = math.sqrt(sum(item * item for item in left)) * math.sqrt(sum(item * item for item in right))
        if denominator == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / denominator

    @staticmethod
    def _unrecognized_exact_codes(text: str) -> set[str]:
        values = set(re.findall(r"\b[a-z]+\d+[a-z0-9-]*\b", text.lower()))
        return {
            value
            for value in values
            if normalize_error_code(value) is None
        }

    @staticmethod
    def _stage(
        ranking: list[str],
        scores: dict[str, float],
        documents: dict[str, IndexedChild],
    ) -> list[dict[str, object]]:
        return [
            {
                "id": child_id,
                "title": documents[child_id].title,
                "score": round(float(scores.get(child_id, 0)), 6),
                "document_id": documents[child_id].document_id,
                "page_start": documents[child_id].page_start,
            }
            for child_id in ranking
            if child_id in documents
        ]
