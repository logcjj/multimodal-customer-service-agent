from __future__ import annotations

import math
import re
from array import array
from collections import Counter
from collections.abc import Iterable
from types import ModuleType
from typing import Any


def install_compact_legacy_bm25(retrieval_module: ModuleType) -> None:
    """Replace the frozen BM25 index storage with a compact equivalent."""
    retriever_class = getattr(retrieval_module, "BM25Retriever", None)
    tokenize = getattr(retrieval_module, "tokenize", None)
    stop_terms = getattr(retrieval_module, "STOP_TERMS", set())
    if retriever_class is None or tokenize is None:
        raise RuntimeError("legacy BM25 runtime is incomplete")
    if getattr(retriever_class, "_aka_compact_index", False):
        return

    def compact_init(
        self: Any,
        chunks: Iterable[Any],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        # The frozen implementation retains every n-gram as a Python string
        # several times. Keep one string-to-id dictionary and store occurrences
        # as unsigned integers instead.
        self._token_ids: dict[str, int] = {}
        self.field_tokens: dict[str, list[array[int]]] = {
            "manual": [],
            "title": [],
            "body": [],
        }
        self.field_doc_freqs: dict[str, Counter[int]] = {
            field: Counter() for field in self.field_tokens
        }
        self.field_lens: dict[str, list[int]] = {
            field: [] for field in self.field_tokens
        }
        self.field_avgdl: dict[str, float] = {}
        self.field_term_freqs: dict[str, list[Counter[int]]] = {
            field: [] for field in self.field_tokens
        }
        self.doc_freq: Counter[int] = Counter()
        self.doc_lens: list[int] = []

        for chunk in self.chunks:
            combined_terms: set[int] = set()
            document_length = 0
            field_text = {
                "manual": chunk.manual,
                "title": chunk.title,
                "body": chunk.text,
            }
            for field, text in field_text.items():
                encoded = array("I")
                frequencies: Counter[int] = Counter()
                for token in tokenize(text):
                    token_id = self._token_ids.get(token)
                    if token_id is None:
                        token_id = len(self._token_ids)
                        self._token_ids[token] = token_id
                    encoded.append(token_id)
                    frequencies[token_id] += 1

                self.field_tokens[field].append(encoded)
                self.field_term_freqs[field].append(frequencies)
                self.field_lens[field].append(len(encoded))
                self.field_doc_freqs[field].update(frequencies.keys())
                combined_terms.update(frequencies.keys())
                document_length += len(encoded)

            self.doc_freq.update(combined_terms)
            self.doc_lens.append(document_length)

        for field, lengths in self.field_lens.items():
            self.field_avgdl[field] = sum(lengths) / max(1, len(lengths))
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))

        # No production search path reads the legacy combined term-frequency
        # copy. Keep the attribute for compatibility without duplicating the
        # complete index a second time.
        self.term_freqs: list[Counter[int]] = []
        self.product_terms_by_manual = {
            chunk.manual: self._product_terms(chunk.manual) for chunk in self.chunks
        }

    def compact_idf(self: Any, term: str) -> float:
        token_id = self._token_ids.get(term)
        document_frequency = self.doc_freq.get(token_id, 0) if token_id is not None else 0
        document_count = len(self.chunks)
        return math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def compact_field_idf(self: Any, field: str, term: str) -> float:
        token_id = self._token_ids.get(term)
        document_frequency = (
            self.field_doc_freqs[field].get(token_id, 0) if token_id is not None else 0
        )
        document_count = len(self.chunks)
        return math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def compact_field_bm25_score(
        self: Any,
        field: str,
        index: int,
        term: str,
        query_frequency: int,
    ) -> float:
        token_id = self._token_ids.get(term)
        if token_id is None:
            return 0.0
        frequency = self.field_term_freqs[field][index].get(token_id, 0)
        if not frequency:
            return 0.0
        document_length = self.field_lens[field][index] or 1
        average_length = self.field_avgdl[field] or 1.0
        denominator = frequency + self.k1 * (
            1 - self.b + self.b * document_length / average_length
        )
        return (
            compact_field_idf(self, field, term)
            * frequency
            * (self.k1 + 1)
            / denominator
            * min(query_frequency, 2)
        )

    def compact_proximity_bonus(
        self: Any,
        query_terms: list[str],
        index: int,
        product_terms: set[str],
    ) -> float:
        meaningful_terms = list(
            dict.fromkeys(
                term
                for term in query_terms
                if term not in stop_terms
                and term not in product_terms
                and not (len(term) == 1 and term.isascii())
                and (len(term) >= 3 or re.search(r"[\u4e00-\u9fff]", term))
            )
        )
        meaningful_ids = {
            token_id
            for term in meaningful_terms
            if (token_id := self._token_ids.get(term)) is not None
        }
        if len(meaningful_ids) < 2:
            return 0.0

        bonus = 0.0
        for field, field_weight, window_limit in (
            ("title", 10.0, 8),
            ("body", 2.6, 22),
        ):
            tokens = self.field_tokens[field][index]
            positions = [
                (position, token_id)
                for position, token_id in enumerate(tokens)
                if token_id in meaningful_ids
            ]
            if len(positions) < 2:
                continue

            best_hits = 0
            best_span = window_limit + 1
            left = 0
            counts: Counter[int] = Counter()
            for right_position, right_term in positions:
                counts[right_term] += 1
                while right_position - positions[left][0] > window_limit:
                    left_term = positions[left][1]
                    counts[left_term] -= 1
                    if counts[left_term] <= 0:
                        del counts[left_term]
                    left += 1
                hits = len(counts)
                span = right_position - positions[left][0]
                if hits > best_hits or (hits == best_hits and span < best_span):
                    best_hits = hits
                    best_span = span

            if best_hits < 2:
                continue
            closeness = max(
                0.0,
                (window_limit + 1 - best_span) / (window_limit + 1),
            )
            bonus += field_weight * (best_hits + 1.5 * closeness)
        return min(bonus, 45.0)

    retriever_class.__init__ = compact_init
    retriever_class.idf = compact_idf
    retriever_class.field_idf = compact_field_idf
    retriever_class.field_bm25_score = compact_field_bm25_score
    retriever_class._proximity_bonus = compact_proximity_bonus
    retriever_class._aka_compact_index = True
