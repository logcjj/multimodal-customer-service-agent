from __future__ import annotations

import math
import unittest
from array import array
from dataclasses import dataclass
from types import ModuleType

from app.compatibility.legacy_retrieval import install_compact_legacy_bm25


@dataclass(frozen=True)
class _Chunk:
    manual: str
    title: str
    text: str


def _legacy_module() -> ModuleType:
    class FrozenRetriever:
        @staticmethod
        def _product_terms(manual_name: str) -> set[str]:
            return {manual_name.lower()}

    module = ModuleType("fake_legacy_retrieval")
    module.BM25Retriever = FrozenRetriever
    module.STOP_TERMS = {"the"}
    module.tokenize = lambda text: text.lower().split()
    return module


class CompactLegacyBM25Tests(unittest.TestCase):
    def test_compact_index_preserves_field_bm25_score(self) -> None:
        module = _legacy_module()
        install_compact_legacy_bm25(module)
        retriever = module.BM25Retriever(
            [
                _Chunk("manual-a", "filter cleaning", "clean filter clean safely"),
                _Chunk("manual-b", "filter replacement", "replace filter safely"),
            ]
        )

        score = retriever.field_bm25_score("body", 0, "clean", 1)
        expected_idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
        expected_denominator = 2 + 1.5 * (1 - 0.75 + 0.75 * 4 / 3.5)
        expected = expected_idf * 2 * (1.5 + 1) / expected_denominator

        self.assertAlmostEqual(score, expected)
        self.assertIsInstance(retriever.field_tokens["body"][0], array)
        self.assertEqual(retriever.term_freqs, [])

    def test_compact_index_counts_document_frequency_once_per_document(self) -> None:
        module = _legacy_module()
        install_compact_legacy_bm25(module)
        retriever = module.BM25Retriever(
            [
                _Chunk("same", "same", "same"),
                _Chunk("other", "title", "body"),
            ]
        )

        expected = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))

        self.assertAlmostEqual(retriever.idf("same"), expected)

    def test_compact_index_preserves_proximity_bonus_and_is_idempotent(self) -> None:
        module = _legacy_module()
        install_compact_legacy_bm25(module)
        compact_init = module.BM25Retriever.__init__
        install_compact_legacy_bm25(module)
        retriever = module.BM25Retriever(
            [_Chunk("manual", "filter clean", "filter can be clean safely")]
        )

        self.assertIs(module.BM25Retriever.__init__, compact_init)
        self.assertGreater(
            retriever._proximity_bonus(["filter", "clean"], 0, set()),
            0,
        )


if __name__ == "__main__":
    unittest.main()
