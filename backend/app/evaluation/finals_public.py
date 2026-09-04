from __future__ import annotations

import base64
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field


_URL_RE = re.compile(r"https?://[^\s\"'，。！？；]+", re.IGNORECASE)
_PIC_RE = re.compile(r"<PIC>")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LITERAL_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_COMMON_REPLACEMENTS = {
    "\u00a0": " ",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "…": "，",
    "·": "-",
    "•": "-",
    "≤": "<=",
    "≥": ">=",
    "℃": "C",
    "℉": "F",
}
_ALLOWED_IMAGE_HOSTS = {"pic.imgdb.cn", "files.superbed.cc", "xhslink.com"}
_MAX_IMAGE_BYTES = 8 * 1024 * 1024


class FinalsQuestion(BaseModel):
    id: str
    question: str
    image_urls: list[str] = Field(default_factory=list)


class DownloadedPublicImage(BaseModel):
    source_url: str
    resolved_url: str
    mime_type: str
    sha256: str
    size_bytes: int
    data_url: str


def load_finals_questions(path: str | Path) -> list[FinalsQuestion]:
    resolved = Path(path).expanduser().resolve()
    text = _read_csv_text(resolved)
    rows = csv.DictReader(text.splitlines())
    questions: list[FinalsQuestion] = []
    seen: set[str] = set()
    for row in rows:
        question_id = str(row.get("id") or "").strip()
        raw_question = str(row.get("question") or "")
        if not question_id or not raw_question.strip():
            raise ValueError("finals public rows require non-empty id and question")
        if question_id in seen:
            raise ValueError(f"duplicate finals public question id: {question_id}")
        seen.add(question_id)
        image_urls = list(dict.fromkeys(_URL_RE.findall(raw_question)))
        clean_question = _normalize_question(_URL_RE.sub("", raw_question))
        if not clean_question:
            clean_question = "请结合图片说明相关情况。"
        questions.append(
            FinalsQuestion(
                id=question_id,
                question=clean_question,
                image_urls=image_urls,
            )
        )
    return questions


def format_competition_ret(
    answer: str,
    image_ids: list[str] | None = None,
    *,
    max_images: int = 3,
) -> str:
    clean_answer = sanitize_submission_text(answer)
    selected = list(
        dict.fromkeys(
            sanitize_submission_text(item)
            for item in (image_ids or [])
            if sanitize_submission_text(item)
        )
    )[:max_images]
    existing_pics = len(_PIC_RE.findall(clean_answer))
    if selected:
        if existing_pics > len(selected):
            seen = 0

            def keep_supported(match: re.Match[str]) -> str:
                nonlocal seen
                seen += 1
                return match.group(0) if seen <= len(selected) else ""

            clean_answer = _PIC_RE.sub(keep_supported, clean_answer)
            existing_pics = len(selected)
        if existing_pics < len(selected):
            clean_answer = clean_answer.rstrip() + "\n" + "\n".join(
                "<PIC>" for _ in range(len(selected) - existing_pics)
            )
        return (
            json.dumps(clean_answer, ensure_ascii=False)
            + ","
            + json.dumps(selected, ensure_ascii=False)
        )
    return _PIC_RE.sub("", clean_answer).strip()


def download_public_image(
    url: str,
    *,
    client: httpx.Client | None = None,
    max_bytes: int = _MAX_IMAGE_BYTES,
) -> DownloadedPublicImage:
    host = (urlparse(url).hostname or "").lower()
    if host not in _ALLOWED_IMAGE_HOSTS:
        raise ValueError(f"image host is not allowed: {host or 'missing'}")
    owns_client = client is None
    resolved_client = client or httpx.Client(
        follow_redirects=True,
        timeout=30,
        trust_env=False,
    )
    try:
        response = resolved_client.get(url)
        response.raise_for_status()
    finally:
        if owns_client:
            resolved_client.close()
    resolved_host = (response.url.host or "").lower()
    if resolved_host not in _ALLOWED_IMAGE_HOSTS:
        raise ValueError(f"redirected image host is not allowed: {resolved_host or 'missing'}")
    payload = response.content
    if not payload or len(payload) > max_bytes:
        raise ValueError("image payload is empty or exceeds the size limit")
    declared = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    detected = _detect_image_mime(payload)
    if detected is None or (declared and not declared.startswith("image/")):
        raise ValueError("downloaded resource is not an image")
    encoded = base64.b64encode(payload).decode("ascii")
    return DownloadedPublicImage(
        source_url=url,
        resolved_url=str(response.url),
        mime_type=detected,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        data_url=f"data:{detected};base64,{encoded}",
    )


def sanitize_submission_text(value: object) -> str:
    clean = str(value or "")
    for old, new in _COMMON_REPLACEMENTS.items():
        clean = clean.replace(old, new)
    clean = _ZERO_WIDTH_RE.sub("", clean)
    clean = _CONTROL_RE.sub("", clean)
    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def _read_csv_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("gb18030")


def _normalize_question(value: str) -> str:
    value = _LITERAL_UNICODE_ESCAPE_RE.sub(
        lambda match: chr(int(match.group(1), 16)),
        value,
    )
    value = re.sub(r'"\s*,\s*\??\s*"', "\n", value)
    lines: list[str] = []
    for line in value.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        clean = line.strip().strip('"“”')
        clean = re.sub(r"[，,]\s*\?$", "", clean)
        clean = clean.strip(" ，,").strip('"“”')
        if clean:
            lines.append(clean)
    return "\n".join(lines)


def _detect_image_mime(payload: bytes) -> str | None:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    return None
