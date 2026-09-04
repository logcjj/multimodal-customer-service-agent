from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

from app.contracts.models import Evidence


@dataclass(frozen=True)
class KnowledgeDocument:
    child_id: str
    parent_id: str
    title: str
    text: str
    product: str
    asset_ids: list[str] = field(default_factory=list)
    document_version: str = "v1"


@dataclass(frozen=True)
class ImageCandidate:
    asset_id: str
    product: str
    related_parent_ids: list[str]


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z]+\d+[a-z0-9-]*|[a-z]{2,}|\d+(?:\.\d+)?", lowered))
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    for run in cjk_runs:
        tokens.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
        if len(run) <= 4:
            tokens.add(run)
    return {token for token in tokens if token}


class HybridRetriever:
    """Small deterministic retrieval core with parent aggregation.

    The public interface is intentionally compatible with a future BM25 + dense +
    RRF implementation. The initial implementation keeps local development and
    fallback behavior deterministic while the legacy champion remains available.
    """

    def __init__(self, documents: list[KnowledgeDocument]) -> None:
        self.documents = documents
        self._document_tokens = {item.child_id: _tokens(f"{item.title} {item.text}") for item in documents}

    def search(self, query: str, products: list[str] | None = None, top_k: int = 5) -> list[Evidence]:
        query_tokens = _tokens(query)
        exact_codes = set(re.findall(r"\b[a-z]+\d+[a-z0-9-]*\b", query.lower()))
        allowed_products = set(products or [])
        scored_children: list[tuple[float, KnowledgeDocument]] = []

        for document in self.documents:
            if allowed_products and document.product not in allowed_products:
                continue
            document_tokens = self._document_tokens[document.child_id]
            overlap = query_tokens & document_tokens
            if not overlap:
                continue
            exact_hit = bool(exact_codes & document_tokens)
            if len(overlap) < 2 and not exact_hit:
                continue
            coverage = len(overlap) / max(1, len(query_tokens))
            specificity = sum(2.5 for token in exact_codes if token in document_tokens)
            title_overlap = len(query_tokens & _tokens(document.title)) * 0.3
            score = coverage * 4 + specificity + title_overlap + math.log1p(len(overlap))
            scored_children.append((score, document))

        grouped: dict[str, list[tuple[float, KnowledgeDocument]]] = defaultdict(list)
        for score, document in scored_children:
            grouped[document.parent_id].append((score, document))

        parent_results: list[Evidence] = []
        for parent_id, children in grouped.items():
            children.sort(key=lambda item: item[0], reverse=True)
            best = children[0][1]
            child_coverage_bonus = min(1.2, (len(children) - 1) * 0.4)
            parent_score = children[0][0] + child_coverage_bonus
            ordered_documents = sorted((item[1] for item in children), key=lambda item: item.child_id)
            text = "\n".join(dict.fromkeys(item.text for item in ordered_documents))
            assets = list(dict.fromkeys(asset for item in ordered_documents for asset in item.asset_ids))
            parent_results.append(
                Evidence(
                    evidence_id=f"evidence:{parent_id}",
                    source_type="manual",
                    title=best.title,
                    text=text,
                    product=best.product,
                    document_id=parent_id,
                    document_version=best.document_version,
                    section_id=parent_id,
                    parent_id=parent_id,
                    chapter_title=best.title,
                    asset_ids=assets,
                    score=round(parent_score, 4),
                    retrieval_stage="deterministic-parent",
                    evidence_confidence=round(min(0.95, 0.5 + parent_score / 20), 4),
                )
            )

        parent_results.sort(key=lambda item: item.score or 0, reverse=True)
        return parent_results[:top_k]

    @staticmethod
    def filter_images(evidence: Evidence, candidates: list[ImageCandidate]) -> list[str]:
        return [
            item.asset_id
            for item in candidates
            if item.product == evidence.product and evidence.document_id in item.related_parent_ids
        ]
