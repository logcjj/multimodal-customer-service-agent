from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import uuid4

from app.knowledge.parsers import NormalizedBlock, NormalizedDocument


@dataclass(frozen=True)
class ParentDraft:
    local_id: str
    title: str
    heading_path: list[str]
    text: str
    page_start: int
    page_end: int
    token_count: int
    product: str | None = None


@dataclass(frozen=True)
class ChildDraft:
    local_id: str
    parent_local_id: str
    title: str
    text: str
    normalized_text: str
    page_start: int
    page_end: int
    token_count: int
    keywords: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)
    product: str | None = None


@dataclass(frozen=True)
class ChunkingResult:
    parents: list[ParentDraft]
    children: list[ChildDraft]


class ChunkingService:
    def __init__(self, child_chars: int = 520, overlap_chars: int = 70) -> None:
        self.child_chars = child_chars
        self.overlap_chars = overlap_chars

    def chunk(self, document: NormalizedDocument, profile: str) -> ChunkingResult:
        if profile not in {"general", "manual", "qa", "table", "picture"}:
            raise ValueError(f"unsupported parser profile: {profile}")
        sections = self._sections(document)
        parents: list[ParentDraft] = []
        children: list[ChildDraft] = []
        for section_index, section in enumerate(sections, start=1):
            parent_local_id = f"parent-{section_index}-{uuid4().hex[:8]}"
            title = section["title"] or document.title
            blocks: list[tuple[int, NormalizedBlock]] = section["blocks"]
            content_parts = [block.text for _, block in blocks if block.text]
            text = "\n".join(content_parts).strip() or title
            page_start = min((page for page, _ in blocks), default=1)
            page_end = max((page for page, _ in blocks), default=page_start)
            product = self._detect_product(f"{title} {text}")
            parents.append(
                ParentDraft(
                    local_id=parent_local_id,
                    title=title,
                    heading_path=[title],
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    token_count=len(text),
                    product=product,
                )
            )
            asset_ids = [block.asset_local_id for _, block in blocks if block.asset_local_id]
            pieces = self._profile_pieces(text, profile)
            for piece_index, piece in enumerate(pieces, start=1):
                keywords = self._keywords(f"{title} {piece}", profile)
                questions = [title] if profile == "qa" else []
                tags = [profile]
                children.append(
                    ChildDraft(
                        local_id=f"child-{section_index}-{piece_index}-{uuid4().hex[:8]}",
                        parent_local_id=parent_local_id,
                        title=title,
                        text=piece,
                        normalized_text=" ".join(piece.lower().split()),
                        page_start=page_start,
                        page_end=page_end,
                        token_count=len(piece),
                        keywords=keywords,
                        questions=questions,
                        tags=tags,
                        asset_ids=[item for item in asset_ids if item],
                        product=product,
                    )
                )
        return ChunkingResult(parents=parents, children=children)

    @staticmethod
    def _sections(document: NormalizedDocument) -> list[dict[str, object]]:
        sections: list[dict[str, object]] = []
        current: dict[str, object] = {"title": document.title, "blocks": []}
        for page in document.pages:
            for block in page.blocks:
                if block.kind == "heading" and current["blocks"]:
                    sections.append(current)
                    current = {"title": block.text, "blocks": []}
                elif block.kind == "heading":
                    current["title"] = block.text
                else:
                    current["blocks"].append((page.page_number, block))
        if current["blocks"] or not sections:
            sections.append(current)
        return sections

    def _profile_pieces(self, text: str, profile: str) -> list[str]:
        if profile == "table":
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) > 1:
                header = lines[0]
                return [f"{header}\n{line}" for line in lines[1:]]
        if profile == "qa":
            pairs = re.split(r"(?=\b(?:Q|问)[:：])", text, flags=re.IGNORECASE)
            pairs = [item.strip() for item in pairs if item.strip()]
            if pairs:
                return pairs
        if len(text) <= self.child_chars:
            return [text]
        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.child_chars)
            boundary = max(text.rfind("。", start, end), text.rfind("\n", start, end))
            if boundary > start + self.child_chars // 2:
                end = boundary + 1
            pieces.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(start + 1, end - self.overlap_chars)
        return [item for item in pieces if item]

    @staticmethod
    def _keywords(text: str, profile: str) -> list[str]:
        values = re.findall(r"\b[A-Za-z]+\d+[A-Za-z0-9-]*\b|\b\d+(?:\.\d+)?(?:V|W|A|Hz|mm|cm)?\b", text)
        if profile == "manual":
            for word in ("警告", "注意", "断电", "过滤器", "排水", "安装", "清洁"):
                if word in text:
                    values.append(word)
        return list(dict.fromkeys(values))[:20]

    @staticmethod
    def _detect_product(text: str) -> str | None:
        lowered = text.lower()
        mapping = {
            "washing-machine": ("洗衣机", "e03", "排水"),
            "air-purifier": ("空气净化器", "滤网", "净化器"),
            "fitness-tracker": ("手环", "健身追踪器", "表带"),
        }
        for product, terms in mapping.items():
            if any(term in lowered for term in terms):
                return product
        return None

