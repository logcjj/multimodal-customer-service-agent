from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from app.contracts.models import Evidence
from app.knowledge.hybrid import tokenize


@dataclass(frozen=True)
class AssetMatchCandidate:
    asset_id: str
    document_id: str
    sha256: str | None
    source_name: str | None


@dataclass(frozen=True)
class AssetMatch:
    asset_id: str | None
    document_id: str | None
    strategy: Literal["sha256", "stem", "unmatched", "ambiguous"]


@dataclass(frozen=True)
class AssetImportContext:
    asset_id: str
    dataset_id: str
    document_id: str
    index_version: str
    page_number: int
    sha256: str | None
    source_name: str | None
    manual_name: str
    caption: str
    ocr_text: str
    related_parent_ids: list[str]
    related_child_ids: list[str]
    chapter_title: str = ""
    retrieval_text: str = ""


@dataclass(frozen=True)
class ImageInsightImportReport:
    total_insights: int
    matched_sha256: int
    matched_stem: int
    ambiguous_insights: int
    unmatched_insights: int
    malformed_insights: int
    fallback_assets: int
    imported_rows: int
    embedded_rows: int = 0


@dataclass(frozen=True)
class IndexedImageChunk:
    image_chunk_id: str
    dataset_id: str
    document_id: str
    document_version: str
    asset_id: str
    image_id: str
    manual_name: str
    chapter_title: str
    page_number: int
    caption: str
    ocr_text: str
    retrieval_text: str
    confidence: float
    related_parent_ids: list[str]
    related_child_ids: list[str]
    file_id: str | None = None
    document_name: str | None = None
    document_mime_type: str | None = None
    product: str | None = None
    embedding: list[float] | None = None


@dataclass(frozen=True)
class ImageRetrievalExplanation:
    query: str
    mode: str
    stages: dict[str, list[dict[str, object]]]
    warnings: list[str] = field(default_factory=list)


EmbedOverride = Callable[[list[str]], list[list[float]]]
RerankOverride = Callable[[str, list[str]], list[float]]


def _normalized_stem(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", Path(value).stem.lower())


def match_insight_to_asset(
    insight: dict[str, object],
    candidates: list[AssetMatchCandidate],
) -> AssetMatch:
    digest = str(insight.get("sha256") or "").strip().lower()
    if digest:
        matches = [item for item in candidates if (item.sha256 or "").lower() == digest]
        if len(matches) == 1:
            return AssetMatch(matches[0].asset_id, matches[0].document_id, "sha256")
        if len(matches) > 1:
            return AssetMatch(None, None, "ambiguous")

    stems = {
        value
        for value in (
            _normalized_stem(str(insight.get("file_name") or "")),
            _normalized_stem(str(insight.get("image_id") or "")),
        )
        if value
    }
    matches = [item for item in candidates if _normalized_stem(item.source_name) in stems]
    if len(matches) == 1:
        return AssetMatch(matches[0].asset_id, matches[0].document_id, "stem")
    if len(matches) > 1:
        return AssetMatch(None, None, "ambiguous")
    return AssetMatch(None, None, "unmatched")


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [str(item) for item in value if item is not None]
    else:
        values = []
    return list(dict.fromkeys(item.strip() for item in values if item.strip()))


def normalize_image_insight(
    insight: dict[str, object],
    *,
    dataset_id: str,
    document_id: str,
    index_version: str,
    asset_id: str,
    page_number: int,
) -> dict[str, object]:
    visible_text = _strings(insight.get("visible_text"))
    search_terms = _strings(insight.get("search_terms"))
    questions = _strings(insight.get("applicable_questions"))
    issue_signals = _strings(insight.get("issue_signals"))
    retrieval_text = str(insight.get("retrieval_text") or "").strip()
    caption = str(insight.get("visual_summary") or insight.get("chapter_hint") or "").strip()
    visual_meaning = str(insight.get("visual_meaning") or "").strip()
    ocr_text = "\n".join(visible_text)
    embedding_parts = [
        str(insight.get("manual_name") or ""),
        str(insight.get("chapter_hint") or ""),
        caption,
        visual_meaning,
        ocr_text,
        "；".join(search_terms),
        "；".join(questions),
        retrieval_text,
    ]
    embedding_text = "\n".join(part for part in embedding_parts if part).strip()[:12_000]
    raw_confidence = insight.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("image insight confidence must be numeric") from exc
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("image insight confidence must be between 0 and 1")
    image_id = str(insight.get("image_id") or Path(str(insight.get("file_name") or asset_id)).stem)
    content_hash = hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()
    stable_id = "image-" + hashlib.sha256(
        f"{dataset_id}|{document_id}|{asset_id}|{content_hash}".encode("utf-8")
    ).hexdigest()[:24]
    return {
        "id": stable_id,
        "dataset_id": dataset_id,
        "document_id": document_id,
        "index_version": index_version,
        "asset_id": asset_id,
        "image_id": image_id,
        "manual_name": str(insight.get("manual_name") or ""),
        "chapter_title": str(insight.get("chapter_hint") or ""),
        "page_number": max(1, int(page_number)),
        "caption": caption,
        "ocr_text": ocr_text,
        "visible_text": visible_text,
        "visual_summary": caption,
        "visual_meaning": visual_meaning,
        "retrieval_text": retrieval_text or embedding_text,
        "embedding_text": embedding_text,
        "search_terms": search_terms,
        "applicable_questions": questions,
        "issue_signals": issue_signals,
        "related_parent_ids": [],
        "related_child_ids": [],
        "confidence": confidence,
        "content_hash": content_hash,
    }


def build_image_chunk_rows(
    insights_path: str | Path,
    contexts: list[AssetImportContext],
) -> tuple[dict[str, list[dict[str, object]]], ImageInsightImportReport]:
    candidates = [
        AssetMatchCandidate(
            asset_id=item.asset_id,
            document_id=item.document_id,
            sha256=item.sha256,
            source_name=item.source_name,
        )
        for item in contexts
    ]
    context_by_asset = {item.asset_id: item for item in contexts}
    rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    matched_assets: set[str] = set()
    total = matched_sha256 = matched_stem = ambiguous = unmatched = malformed = 0

    with Path(insights_path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            total += 1
            try:
                insight = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(insight, dict):
                malformed += 1
                continue
            match = match_insight_to_asset(insight, candidates)
            if match.asset_id is None:
                if match.strategy == "ambiguous":
                    ambiguous += 1
                else:
                    unmatched += 1
                continue
            context = context_by_asset[match.asset_id]
            try:
                row = normalize_image_insight(
                    {
                        **insight,
                        "image_id": insight.get("image_id") or context.caption,
                        "manual_name": insight.get("manual_name") or context.manual_name,
                        "chapter_hint": insight.get("chapter_hint") or context.chapter_title,
                        "retrieval_text": insight.get("retrieval_text") or context.retrieval_text,
                    },
                    dataset_id=context.dataset_id,
                    document_id=context.document_id,
                    index_version=context.index_version,
                    asset_id=context.asset_id,
                    page_number=context.page_number,
                )
            except (TypeError, ValueError):
                malformed += 1
                continue
            row["related_parent_ids"] = list(context.related_parent_ids)
            row["related_child_ids"] = list(context.related_child_ids)
            rows[context.document_id].append(row)
            matched_assets.add(context.asset_id)
            if match.strategy == "sha256":
                matched_sha256 += 1
            else:
                matched_stem += 1

    fallback_assets = 0
    for context in contexts:
        if context.asset_id in matched_assets or not (
            context.caption.strip() or context.ocr_text.strip() or context.retrieval_text.strip()
        ):
            continue
        fallback = normalize_image_insight(
            {
                "image_id": context.caption or Path(context.source_name or context.asset_id).stem,
                "manual_name": context.manual_name,
                "chapter_hint": context.chapter_title,
                "visual_summary": context.caption or context.chapter_title,
                "visual_meaning": context.chapter_title or context.caption,
                "visible_text": context.ocr_text.splitlines(),
                "retrieval_text": "\n".join(
                    value
                    for value in (
                        context.chapter_title.strip(),
                        context.caption.strip(),
                        context.ocr_text.strip(),
                        context.retrieval_text.strip(),
                    )
                    if value
                ),
                "search_terms": [context.caption, context.chapter_title],
                "applicable_questions": [],
                "issue_signals": [],
                "confidence": 0.65,
            },
            dataset_id=context.dataset_id,
            document_id=context.document_id,
            index_version=context.index_version,
            asset_id=context.asset_id,
            page_number=context.page_number,
        )
        fallback["related_parent_ids"] = list(context.related_parent_ids)
        fallback["related_child_ids"] = list(context.related_child_ids)
        rows[context.document_id].append(fallback)
        fallback_assets += 1

    for document_rows in rows.values():
        document_rows.sort(key=lambda item: (int(item["page_number"]), str(item["id"])))
    report = ImageInsightImportReport(
        total_insights=total,
        matched_sha256=matched_sha256,
        matched_stem=matched_stem,
        ambiguous_insights=ambiguous,
        unmatched_insights=unmatched,
        malformed_insights=malformed,
        fallback_assets=fallback_assets,
        imported_rows=sum(len(items) for items in rows.values()),
    )
    return dict(rows), report


class PublishedImageRetriever:
    def __init__(
        self,
        documents: list[IndexedImageChunk],
        *,
        embed: EmbedOverride | None = None,
        rerank: RerankOverride | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.documents = documents
        self.embed = embed
        self.rerank = rerank
        self.rrf_k = rrf_k
        self._tokens = {
            item.image_chunk_id: tokenize(
                f"{item.manual_name} {item.chapter_title} {item.caption} "
                f"{item.ocr_text} {item.retrieval_text} {item.product or ''}"
            )
            for item in documents
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        query_vector: list[float] | None = None,
    ) -> tuple[list[Evidence], ImageRetrievalExplanation]:
        query_tokens = tokenize(query)
        lexical = self._bm25(query_tokens)
        lexical_ranked = sorted(lexical, key=lexical.get, reverse=True)[:20]
        embedded = [item for item in self.documents if item.embedding]
        resolved_vector = query_vector
        warnings: list[str] = []
        if resolved_vector is None and self.embed and embedded:
            try:
                returned = self.embed([query])
            except Exception:
                returned = []
            resolved_vector = returned[0] if len(returned) == 1 and returned[0] else []
        dense = {
            item.image_chunk_id: max(0.0, self._cosine(resolved_vector or [], item.embedding or []))
            for item in embedded
            if resolved_vector
        }
        dense_ranked = sorted(
            (item_id for item_id, score in dense.items() if score > 0),
            key=dense.get,
            reverse=True,
        )[:20]
        if embedded and resolved_vector == []:
            warnings.append("图片 Caption 查询向量生成失败，已使用图片 BM25/RRF。")

        fused: dict[str, float] = defaultdict(float)
        for ranking in (lexical_ranked, dense_ranked):
            for rank, item_id in enumerate(ranking, start=1):
                fused[item_id] += 1 / (self.rrf_k + rank)
        fused_ranked = sorted(fused, key=fused.get, reverse=True)[:20]
        by_id = {item.image_chunk_id: item for item in self.documents}
        rerank_scores: dict[str, float] = {}
        final = fused_ranked
        mode = "image-hybrid" if dense_ranked else "image-lexical-only"
        if self.rerank and fused_ranked:
            candidates = fused_ranked[:12]
            try:
                returned = self.rerank(query, [by_id[item].retrieval_text for item in candidates])
            except Exception:
                returned = []
            if len(returned) == len(candidates):
                rerank_scores = dict(zip(candidates, map(float, returned), strict=True))
                final = sorted(candidates, key=rerank_scores.get, reverse=True)
                mode = "image-hybrid-rerank"
            else:
                warnings.append("图片 Rerank 调用失败，当前使用图片 RRF 顺序。")

        evidence = [self._evidence(by_id[item], lexical, dense, fused, rerank_scores) for item in final[:top_k]]
        stages = {
            "image_lexical": self._stage(lexical_ranked, lexical, by_id),
            "image_dense": self._stage(dense_ranked, dense, by_id),
            "image_rrf": self._stage(fused_ranked, fused, by_id),
            "image_rerank": self._stage(
                sorted(rerank_scores, key=rerank_scores.get, reverse=True), rerank_scores, by_id
            ),
        }
        return evidence, ImageRetrievalExplanation(query=query, mode=mode, stages=stages, warnings=warnings)

    def _evidence(
        self,
        item: IndexedImageChunk,
        lexical: dict[str, float],
        dense: dict[str, float],
        fused: dict[str, float],
        rerank: dict[str, float],
    ) -> Evidence:
        score_breakdown = {
            "image_lexical": round(lexical.get(item.image_chunk_id, 0.0), 6),
            "image_dense": round(dense.get(item.image_chunk_id, 0.0), 6),
            "image_rrf": round(fused.get(item.image_chunk_id, 0.0), 6),
            "image_rerank": round(rerank.get(item.image_chunk_id, 0.0), 6),
        }
        score = score_breakdown["image_rrf"] + score_breakdown["image_rerank"] * 0.004
        return Evidence(
            evidence_id=f"image-evidence:{item.image_chunk_id}",
            source_type="image",
            title=item.caption or item.chapter_title or item.image_id,
            text=item.retrieval_text,
            product=item.product,
            dataset_id=item.dataset_id,
            document_id=item.document_id,
            file_id=item.file_id,
            document_name=item.document_name or item.manual_name,
            document_mime_type=item.document_mime_type,
            document_version=item.document_version,
            parent_id=item.related_parent_ids[0] if item.related_parent_ids else None,
            child_ids=item.related_child_ids,
            image_chunk_ids=[item.image_chunk_id],
            chapter_title=item.chapter_title,
            page_start=item.page_number,
            page_end=item.page_number,
            locator_label=f"第 {item.page_number} 页图片",
            asset_ids=[item.asset_id],
            score=max(0.0, round(score, 6)),
            score_breakdown=score_breakdown,
            retrieval_stage="image_rerank" if rerank else "image_rrf",
            evidence_confidence=item.confidence,
        )

    def _bm25(self, query_tokens: list[str]) -> dict[str, float]:
        if not query_tokens or not self.documents:
            return {}
        token_lists = [self._tokens[item.image_chunk_id] for item in self.documents]
        average_length = sum(map(len, token_lists)) / max(1, len(token_lists))
        frequencies = Counter(token for values in token_lists for token in set(values))
        scores: dict[str, float] = {}
        for item, tokens in zip(self.documents, token_lists, strict=True):
            counts = Counter(tokens)
            overlap = set(query_tokens) & set(tokens)
            if len(overlap) < 2:
                continue
            score = 0.0
            for token in query_tokens:
                frequency = counts[token]
                if not frequency:
                    continue
                document_frequency = frequencies[token]
                inverse = math.log(
                    1 + (len(self.documents) - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * len(tokens) / max(1, average_length)
                )
                score += inverse * (frequency * 2.5) / denominator
            if score > 0:
                scores[item.image_chunk_id] = score
        return scores

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return 0.0
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
            sum(value * value for value in right)
        )
        if denominator == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / denominator

    @staticmethod
    def _stage(
        ranking: list[str],
        scores: dict[str, float],
        documents: dict[str, IndexedImageChunk],
    ) -> list[dict[str, object]]:
        return [
            {
                "id": item_id,
                "title": documents[item_id].caption or documents[item_id].chapter_title,
                "score": round(scores.get(item_id, 0.0), 6),
                "document_id": documents[item_id].document_id,
                "page_start": documents[item_id].page_number,
                "asset_id": documents[item_id].asset_id,
            }
            for item_id in ranking
            if item_id in documents
        ]


_VISUAL_INTENT = re.compile(r"图片|截图|照片|图示|铭牌|标签|按钮|图标|部件|位置|外观|在哪|哪一页")


def merge_image_evidence(
    text_evidence: list[Evidence],
    image_evidence: list[Evidence],
    *,
    mode: Literal["off", "shadow", "on"],
    query: str,
) -> tuple[list[Evidence], list[Evidence]]:
    if mode == "off":
        return list(text_evidence), []
    if mode == "shadow":
        return list(text_evidence), list(image_evidence)
    if not _VISUAL_INTENT.search(query):
        return list(text_evidence), list(image_evidence)
    return [*text_evidence, *image_evidence], []
