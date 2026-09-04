from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException


MAX_FILE_BYTES = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


@dataclass(frozen=True)
class StoredFile:
    content_hash: str
    mime_type: str
    size_bytes: int
    relative_path: str
    absolute_path: Path


class ContentAddressedStorage:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "objects"
        self.root.mkdir(parents=True, exist_ok=True)

    def store(self, original_name: str, declared_mime: str | None, content: bytes) -> StoredFile:
        if not content:
            raise HTTPException(status_code=400, detail="文件内容为空")
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="文件超过 50MB 限制")
        extension = Path(original_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="不支持的文件类型")
        detected = self._detect_mime(extension, content)
        expected = ALLOWED_EXTENSIONS[extension]
        normalized_declared = (declared_mime or mimetypes.guess_type(original_name)[0] or expected).lower()
        if not self._mime_compatible(expected, normalized_declared) or not self._mime_compatible(expected, detected):
            raise HTTPException(status_code=415, detail="文件类型与内容不一致")

        digest = hashlib.sha256(content).hexdigest()
        relative = Path("objects") / digest[:2] / f"{digest}{extension}"
        absolute = self.root.parent / relative
        absolute.parent.mkdir(parents=True, exist_ok=True)
        if not absolute.exists():
            temporary = absolute.with_suffix(f"{absolute.suffix}.tmp")
            temporary.write_bytes(content)
            temporary.replace(absolute)
        return StoredFile(
            content_hash=digest,
            mime_type=expected,
            size_bytes=len(content),
            relative_path=relative.as_posix(),
            absolute_path=absolute,
        )

    def resolve(self, relative_path: str) -> Path:
        path = (self.root.parent / relative_path).resolve()
        root = self.root.resolve()
        if root not in path.parents:
            raise HTTPException(status_code=400, detail="非法文件路径")
        if not path.exists():
            raise HTTPException(status_code=404, detail="原始文件不存在")
        return path

    @staticmethod
    def _detect_mime(extension: str, content: bytes) -> str:
        if content.startswith(b"%PDF-"):
            return "application/pdf"
        if content.startswith(b"PK\x03\x04") and extension == ".docx":
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if extension in {".txt", ".md"}:
            try:
                content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(status_code=415, detail="文本文件必须使用 UTF-8 编码") from exc
            return "text/markdown" if extension == ".md" else "text/plain"
        return "application/octet-stream"

    @staticmethod
    def _mime_compatible(expected: str, actual: str) -> bool:
        if expected == actual:
            return True
        return expected.startswith("text/") and actual in {"text/plain", "text/markdown", "application/octet-stream"}

