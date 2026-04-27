from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import ILLUSTRATION_DIR, MANUAL_DIR


PIC_TOKEN_RE = re.compile(r"<PIC>")
PIC_REF_RE = re.compile(r"\[PIC:([^\]]+)\]")
HEADING_RE = re.compile(r"(?m)(?=^#\s+)")


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
    else:
        # Some bundled manuals are concatenations of one list literal per line:
        # ["manual text", ["image_id", ...]]
        # Treat each line as one source document instead of indexing the raw
        # Python literal syntax, otherwise retrieval is dominated by artifacts.
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
                texts.append(str(line_parsed[0]))
                if len(line_parsed) > 1 and isinstance(line_parsed[1], list):
                    image_ids.extend(str(item) for item in line_parsed[1])
        if not texts:
            texts.append(raw)

    text = "\n\n".join(texts)
    return ManualDocument(name=path.stem, path=path, text=text, image_ids=tuple(image_ids))


def load_manuals(manual_dir: Path = MANUAL_DIR) -> list[ManualDocument]:
    return [read_manual(path) for path in sorted(manual_dir.glob("*.txt"))]


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


def split_sections(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+#\s+", "\n# ", text)
    text = re.sub(r"[ \t]+", " ", text)
    parts = [part.strip() for part in HEADING_RE.split(text) if part.strip()]
    return parts or [text.strip()]


def section_title(section: str, default: str) -> str:
    first_line = section.strip().splitlines()[0] if section.strip() else ""
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()[:120] or default
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
