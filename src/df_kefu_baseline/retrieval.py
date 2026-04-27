from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .manuals import ManualChunk


CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-/\.]*")
STOP_TERMS = {
    "如何", "怎么", "什么", "哪些", "内容", "正确", "需要", "告诉", "根据", "手册", "说明",
    "请问", "一下", "这个", "那个", "如果", "可以", "应该", "以及", "进行", "使用",
    "the", "and", "what", "how", "are", "for", "with", "proper", "correct", "steps",
    "step", "procedure", "procedures", "if", "this", "that", "is", "to", "of", "on",
    "in", "before", "after", "when", "while", "using", "use", "do", "does", "can",
    "you", "i", "my", "it", "should", "want", "need",
}
HIGH_INTENT_TERMS = {
    "充电", "扣紧", "拆卸", "安装", "设置", "清洁", "启动", "停机", "更换", "检查",
    "调整", "连接", "配对", "排空", "打开", "关闭", "取出", "装入", "保修", "维修",
    "故障", "troubleshooting", "cleaning", "mounting", "charge", "charging", "connect",
    "start", "stop", "adjust", "install", "remove",
}
MANUAL_ALIASES = {
    "vr头显": ("vr", "headset"),
    "健身追踪器": ("fitness tracker", "tracker"),
    "蓝牙激光鼠标": ("bluetooth mouse", "mouse"),
    "电钻": ("drill", "dcb107", "dcb112"),
    "相机": ("camera",),
    "空调": ("air conditioner",),
    "洗碗机": ("dishwasher",),
    "烤箱": ("oven",),
    "发电机": ("generator",),
    "水泵": ("pump",),
    "摩托艇": ("watercraft", "jet ski"),
    "汇总英文": (
        "airfryer", "air fryer", "pressure cooker", "coffee machine", "earphones",
        "ereader", "e-reader", "fax", "grill", "jetski", "landline", "lawn mower",
        "microwave", "motherboard", "vacuum", "snowmobile", "television", "toothbrush",
    ),
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = WORD_RE.findall(text)
    for block in CJK_RE.findall(text):
        if len(block) <= 6:
            tokens.append(block)
        tokens.extend(block[i : i + 2] for i in range(max(0, len(block) - 1)))
        tokens.extend(block[i : i + 3] for i in range(max(0, len(block) - 2)))
    return tokens


@dataclass(frozen=True)
class SearchResult:
    chunk: ManualChunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: Iterable[ManualChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.manual + " " + chunk.title + " " + chunk.text) for chunk in self.chunks]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.product_terms_by_manual = {
            chunk.manual: self._product_terms(chunk.manual) for chunk in self.chunks
        }

    def idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 6) -> list[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        query_counts = Counter(query_terms)
        scored: list[SearchResult] = []
        for idx, chunk in enumerate(self.chunks):
            tf = self.term_freqs[idx]
            doc_len = self.doc_lens[idx] or 1
            product_terms = self.product_terms_by_manual.get(chunk.manual, set())
            score = 0.0
            for term, qtf in query_counts.items():
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                query_weight = 0.25 if term in product_terms else 1.0
                score += self.idf(term) * freq * (self.k1 + 1) / denom * min(qtf, 2) * query_weight
            score += self._manual_name_bonus(query, chunk)
            score += self._focused_term_bonus(query, chunk)
            if score > 0:
                scored.append(SearchResult(chunk=chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def search_many(
        self,
        queries: Iterable[str],
        top_k: int = 10,
        per_query_k: int = 12,
    ) -> list[SearchResult]:
        fused_scores: dict[str, float] = {}
        best_result: dict[str, SearchResult] = {}
        rrf_k = 60.0

        for query_idx, query in enumerate(queries):
            query_weight = 1.0 if query_idx == 0 else 0.72
            for rank, result in enumerate(self.search(query, top_k=per_query_k), start=1):
                chunk_id = result.chunk.id
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + query_weight / (rrf_k + rank)
                if chunk_id not in best_result or result.score > best_result[chunk_id].score:
                    best_result[chunk_id] = result

        fused = [
            SearchResult(chunk=result.chunk, score=result.score + fused_scores[result.chunk.id] * 120.0)
            for result in best_result.values()
        ]
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[:top_k]

    @staticmethod
    def _manual_name_bonus(query: str, chunk: ManualChunk) -> float:
        manual = chunk.manual.replace("手册", "").lower()
        query_l = query.lower()
        if manual and manual in query_l:
            return 2.5
        return 1.5 if any(alias in query_l for alias in MANUAL_ALIASES.get(manual, ())) else 0.0

    @staticmethod
    def _product_terms(manual_name: str) -> set[str]:
        manual = manual_name.replace("手册", "").lower()
        if manual == "汇总英文":
            return set()
        terms = set(tokenize(manual))
        for alias in MANUAL_ALIASES.get(manual, ()):
            terms.update(tokenize(alias))
        return terms

    @staticmethod
    def _focused_term_bonus(query: str, chunk: ManualChunk) -> float:
        manual = chunk.manual.replace("手册", "").lower()
        product_terms = BM25Retriever._product_terms(chunk.manual)

        title = chunk.title.lower()
        head_text = chunk.text[:900].lower()
        bonus = 0.0
        for term in set(tokenize(query)):
            if len(term) < 2 or term in STOP_TERMS or term in product_terms:
                continue
            if any(ch.isdigit() for ch in term) and len(term) == 2:
                continue
            if term in title:
                if term in HIGH_INTENT_TERMS:
                    bonus += 38.0
                else:
                    bonus += min(18.0, 4.0 + len(term) * 1.6)
            elif term in head_text:
                if term in HIGH_INTENT_TERMS:
                    bonus += 12.0
                else:
                    bonus += min(8.0, 1.5 + len(term) * 0.8)
        return min(bonus, 65.0)
