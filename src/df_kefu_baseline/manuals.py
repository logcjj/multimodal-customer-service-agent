from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import ILLUSTRATION_DIR, MANUAL_DIR
from .products import english_manual_name, infer_english_product_key


PIC_TOKEN_RE = re.compile(r"<PIC>")
PIC_REF_RE = re.compile(r"\[PIC:([^\]]+)\]")
HEADING_RE = re.compile(r"(?m)(?=^#\s+)")
SPACED_INLINE_HEADING_RE = re.compile(r"(?<!^)(?<![A-Za-z0-9#])\s*#\s+(?=[\u4e00-\u9fffA-Za-z])")
GLUED_UPPER_HEADING_RE = re.compile(
    r"(?<=[\u4e00-\u9fffA-Za-z0-9）)\]])#\s+(?=[A-Z][A-Z0-9/&(),. -]{2,})"
)
TROUBLESHOOTING_HEADING_BODY_RE = re.compile(r"(?im)^(# trouble\s*shooting)\s+(?=[A-Za-z])")


@dataclass(frozen=True)
class ManualDocument:
    name: str
    path: Path
    text: str
    image_ids: tuple[str, ...]


@dataclass(frozen=True)
class ManualChunk:
    id: str
    manual: str
    title: str
    text: str
    image_ids: tuple[str, ...]


def read_manual(path: Path) -> ManualDocument:
    return read_manual_documents(path)[0]


def read_manual_documents(path: Path) -> list[ManualDocument]:
    raw = path.read_text(encoding="utf-8-sig")
    texts: list[str] = []
    image_ids: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = None

    if isinstance(parsed, list) and parsed:
        texts.append(str(parsed[0]))
        if len(parsed) > 1 and isinstance(parsed[1], list):
            image_ids.extend(str(item) for item in parsed[1])
        text = "\n\n".join(texts)
        return [ManualDocument(name=path.stem, path=path, text=text, image_ids=tuple(image_ids))]
    else:
        # Some bundled manuals are concatenations of one list literal per line:
        # ["manual text", ["image_id", ...]]
        # Treat each line as one source document instead of indexing the raw
        # Python literal syntax, otherwise retrieval is dominated by artifacts.
        documents: list[ManualDocument] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                try:
                    line_parsed = ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    line_parsed = None
            if isinstance(line_parsed, list) and line_parsed:
                text = str(line_parsed[0])
                line_image_ids: list[str] = []
                if len(line_parsed) > 1 and isinstance(line_parsed[1], list):
                    line_image_ids.extend(str(item) for item in line_parsed[1])
                if path.stem == "汇总英文手册":
                    product_key = infer_english_product_key(text, fallback_index=len(documents) + 1)
                    name = english_manual_name(product_key)
                else:
                    name = f"{path.stem}::{len(documents) + 1:02d}"
                documents.append(
                    ManualDocument(
                        name=name,
                        path=path,
                        text=text,
                        image_ids=tuple(line_image_ids),
                    )
                )
        if documents:
            return documents
        return [ManualDocument(name=path.stem, path=path, text=raw, image_ids=())]


def load_manuals(manual_dir: Path = MANUAL_DIR) -> list[ManualDocument]:
    documents: list[ManualDocument] = []
    for path in sorted(manual_dir.glob("*.txt")):
        documents.extend(read_manual_documents(path))
    return documents


def image_path_for_id(image_id: str, illustration_dir: Path = ILLUSTRATION_DIR) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = illustration_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def attach_image_refs(text: str, image_ids: Iterable[str]) -> str:
    images = iter(image_ids)

    def repl(_: re.Match[str]) -> str:
        image_id = next(images, "")
        if image_id and image_path_for_id(image_id) is not None:
            return f"[PIC:{image_id}]"
        return ""

    return PIC_TOKEN_RE.sub(repl, text)


def preprocess_manual_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Correct a recurring OCR heading typo while preserving the manual's meaning.
    text = re.sub(r"\bTROUBLEH?SOOTING(?=[A-Za-z])", "TROUBLESHOOTING ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTROUBLEH?SOOTING\b", "TROUBLESHOOTING", text, flags=re.IGNORECASE)
    # OCR sometimes glues markdown headings to the previous sentence:
    # "...module# TROUBLESHOOTING..." should become a new retrievable section.
    text = SPACED_INLINE_HEADING_RE.sub("\n# ", text)
    text = GLUED_UPPER_HEADING_RE.sub("\n# ", text)
    text = re.sub(r"(?m)^#(?=\S)", "# ", text)
    text = re.sub(r"(?m)^#\s+", "# ", text)
    text = TROUBLESHOOTING_HEADING_BODY_RE.sub(r"\1\n", text)
    text = re.sub(r"\b(TROUBLESHOOTING)(?=[a-z])", r"\1 ", text, flags=re.IGNORECASE)
    return text


def split_sections(text: str) -> list[str]:
    text = preprocess_manual_text(text)
    text = re.sub(r"[ \t]+", " ", text)
    parts = [part.strip() for part in HEADING_RE.split(text) if part.strip()]
    return parts or [text.strip()]


def section_title(section: str, default: str) -> str:
    first_line = section.strip().splitlines()[0] if section.strip() else ""
    if first_line.startswith("#"):
        title = first_line.lstrip("#").strip()
        title = re.sub(r"\s+#\s*$", "", title).strip()
        return title[:120] or default
    return default


def split_long_section(section: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    if len(section) <= max_chars:
        return [section]

    chunks: list[str] = []
    start = 0
    while start < len(section):
        end = min(len(section), start + max_chars)
        if end < len(section):
            pivot = max(
                section.rfind("\n", start, end),
                section.rfind("。", start, end),
                section.rfind(". ", start, end),
            )
            if pivot > start + max_chars * 0.55:
                end = pivot + 1
        chunks.append(section[start:end].strip())
        if end >= len(section):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def build_chunks(manuals: Iterable[ManualDocument]) -> list[ManualChunk]:
    chunks: list[ManualChunk] = []
    for manual in manuals:
        wired = attach_image_refs(manual.text, manual.image_ids)
        for section_idx, section in enumerate(split_sections(wired)):
            title = section_title(section, manual.name)
            for piece_idx, piece in enumerate(split_long_section(section)):
                image_ids = tuple(dict.fromkeys(PIC_REF_RE.findall(piece)))
                chunk_id = f"{manual.name}:{section_idx}:{piece_idx}"
                chunks.append(
                    ManualChunk(
                        id=chunk_id,
                        manual=manual.name,
                        title=title,
                        text=piece,
                        image_ids=image_ids,
                    )
                )
    return chunks
