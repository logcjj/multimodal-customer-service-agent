from __future__ import annotations

import math
import unittest
from array import array
from collections import Counter

from app.knowledge.hybrid import IndexedChild, PublishedHybridRetriever, tokenize


def _child(child_id: str, text: str) -> IndexedChild:
    return IndexedChild(
        child_id=child_id,
        parent_id=f"parent-{child_id}",
        dataset_id="dataset",
        document_id=f"document-{child_id}",
        document_version="v1",
        title="Filter maintenance",
        text=text,
        page_start=1,
        page_end=1,
    )


def _reference_bm25(
    query: str,
    documents: list[IndexedChild],
) -> dict[str, float]:
    query_tokens = tokenize(query)
    token_lists = [
        tokenize(
            f"{item.document_name or ''} {item.product or ''} "
            f"{item.title} {item.text} {' '.join(item.keywords)}"
        )
        for item in documents
    ]
    average_length = sum(map(len, token_lists)) / len(token_lists)
    frequencies = Counter(token for tokens in token_lists for token in set(tokens))
    scores: dict[str, float] = {}
    for document, tokens in zip(documents, token_lists, strict=True):
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
                1
                + (len(documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(tokens) / average_length
            )
            score += inverse * (frequency * 2.5) / denominator
        scores[document.child_id] = score
    return scores


class CompactPublishedRetrieverTests(unittest.TestCase):
    def test_compact_tokens_preserve_bm25_scores(self) -> None:
        documents = [
            _child("one", "clean the filter every month"),
            _child("two", "replace the filter when damaged"),
            _child("three", "clean the outer housing safely"),
        ]
        retriever = PublishedHybridRetriever(documents)
        query = "clean filter maintenance"

        actual = retriever._bm25(
            tokenize(query),
            documents,
            normalized_codes=set(),
            legacy_exact_codes=set(),
        )

        self.assertEqual(set(actual), set(_reference_bm25(query, documents)))
        for child_id, expected in _reference_bm25(query, documents).items():
            self.assertAlmostEqual(actual[child_id], expected)
        self.assertTrue(
            all(isinstance(tokens, array) for tokens in retriever._tokens.values())
        )


if __name__ == "__main__":
    unittest.main()
