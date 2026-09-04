from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


SCHEMA_VERSION = "aka-index-bundle-v2"
_JSONL_ARTIFACTS = ("text_chunks.jsonl", "image_chunks.jsonl", "assets.jsonl")
_VECTOR_ARTIFACTS = ("text_vectors.npz", "image_caption_vectors.npz")
_REQUIRED_FILES = (*_JSONL_ARTIFACTS, *_VECTOR_ARTIFACTS, "manifest.json", "checksums.sha256")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SourceManifest(BaseModel):
    document_id: str
    file_id: str
    source_name: str
    source_sha256: str
    mime_type: str
    size_bytes: int = Field(ge=0)
    parser_fingerprint: str
    document_version: str | None = None


class ArtifactManifest(BaseModel):
    file_name: str
    sha256: str
    size_bytes: int = Field(ge=0)
    row_count: int = Field(ge=0)


class IncrementalStats(BaseModel):
    reused: int = 0
    added: int = 0
    updated: int = 0
    deleted: int = 0


class IndexManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    dataset_id: str
    index_version: str
    parent_index_version: str | None = None
    built_at: datetime
    parser_version: str
    embedding_model: str | None = None
    vector_dimension: int = Field(ge=0)
    sources: list[SourceManifest]
    artifacts: dict[str, ArtifactManifest]
    counts: dict[str, int]
    incremental: IncrementalStats = Field(default_factory=IncrementalStats)
    validation_status: str = "valid"
    evaluation_status: str = "not_run"
    approval_status: str = "awaiting_approval"


class BundleIssue(BaseModel):
    code: str
    message: str
    file_name: str | None = None


class BundleValidationReport(BaseModel):
    valid: bool
    errors: list[BundleIssue] = Field(default_factory=list)


class IncrementalPlan(BaseModel):
    reuse: list[str] = Field(default_factory=list)
    add: list[str] = Field(default_factory=list)
    update: list[str] = Field(default_factory=list)
    delete: list[str] = Field(default_factory=list)


def plan_incremental_build(
    previous: list[SourceManifest],
    current: list[SourceManifest],
) -> IncrementalPlan:
    old = {item.document_id: item for item in previous}
    new = {item.document_id: item for item in current}
    reuse: list[str] = []
    add: list[str] = []
    update: list[str] = []
    for document_id, source in new.items():
        earlier = old.get(document_id)
        if earlier is None:
            add.append(document_id)
        elif (
            earlier.source_sha256 == source.source_sha256
            and earlier.parser_fingerprint == source.parser_fingerprint
        ):
            reuse.append(document_id)
        else:
            update.append(document_id)
    return IncrementalPlan(
        reuse=sorted(reuse),
        add=sorted(add),
        update=sorted(update),
        delete=sorted(set(old) - set(new)),
    )


class IndexBundle:
    def __init__(self, root: Path, manifest: IndexManifest) -> None:
        self.root = root
        self.manifest = manifest

    @classmethod
    def load(cls, root: str | Path) -> "IndexBundle":
        resolved = Path(root).resolve()
        manifest_path = resolved / "manifest.json"
        manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        return cls(resolved, manifest)

    def validate(self) -> BundleValidationReport:
        errors: list[BundleIssue] = []
        for file_name in _REQUIRED_FILES:
            if not (self.root / file_name).is_file():
                errors.append(
                    BundleIssue(
                        code="index_schema_invalid",
                        message=f"索引包缺少必需文件：{file_name}",
                        file_name=file_name,
                    )
                )
        if errors:
            return BundleValidationReport(valid=False, errors=errors)

        expected_checksums: dict[str, str] = {}
        for line in (self.root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                errors.append(
                    BundleIssue(code="index_schema_invalid", message="checksums.sha256 格式错误")
                )
                continue
            expected_checksums[parts[1]] = parts[0]
        for file_name, expected in expected_checksums.items():
            path = self.root / file_name
            if not path.is_file() or _sha256(path) != expected:
                errors.append(
                    BundleIssue(
                        code="index_checksum_mismatch",
                        message=f"索引产物校验失败：{file_name}",
                        file_name=file_name,
                    )
                )

        for file_name, artifact in self.manifest.artifacts.items():
            path = self.root / file_name
            if not path.is_file() or _sha256(path) != artifact.sha256:
                errors.append(
                    BundleIssue(
                        code="index_checksum_mismatch",
                        message=f"Manifest 产物校验失败：{file_name}",
                        file_name=file_name,
                    )
                )
                continue
            if path.stat().st_size != artifact.size_bytes:
                errors.append(
                    BundleIssue(
                        code="index_schema_invalid",
                        message=f"Manifest 文件大小不一致：{file_name}",
                        file_name=file_name,
                    )
                )

        for file_name in _JSONL_ARTIFACTS:
            artifact = self.manifest.artifacts.get(file_name)
            if artifact is None:
                continue
            try:
                rows = [
                    json.loads(line)
                    for line in (self.root / file_name).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (UnicodeDecodeError, json.JSONDecodeError):
                errors.append(
                    BundleIssue(
                        code="index_schema_invalid",
                        message=f"JSONL 无法解析：{file_name}",
                        file_name=file_name,
                    )
                )
                continue
            if len(rows) != artifact.row_count:
                errors.append(
                    BundleIssue(
                        code="index_schema_invalid",
                        message=f"JSONL 行数不一致：{file_name}",
                        file_name=file_name,
                    )
                )

        for file_name in _VECTOR_ARTIFACTS:
            artifact = self.manifest.artifacts.get(file_name)
            if artifact is None:
                continue
            try:
                with np.load(self.root / file_name, allow_pickle=False) as payload:
                    ids = payload["ids"]
                    vectors = payload["vectors"]
            except (OSError, ValueError, KeyError):
                errors.append(
                    BundleIssue(
                        code="index_schema_invalid",
                        message=f"向量文件无法解析：{file_name}",
                        file_name=file_name,
                    )
                )
                continue
            if vectors.ndim != 2 or vectors.shape[0] != len(ids):
                errors.append(
                    BundleIssue(
                        code="embedding_incomplete",
                        message=f"向量 ID 与行数不一致：{file_name}",
                        file_name=file_name,
                    )
                )
            if vectors.shape[0] and vectors.shape[1] != self.manifest.vector_dimension:
                errors.append(
                    BundleIssue(
                        code="embedding_incomplete",
                        message=f"向量维度不一致：{file_name}",
                        file_name=file_name,
                    )
                )
            if not np.isfinite(vectors).all() or len(set(ids.tolist())) != len(ids):
                errors.append(
                    BundleIssue(
                        code="embedding_incomplete",
                        message=f"向量包含无效值或重复 ID：{file_name}",
                        file_name=file_name,
                    )
                )
            if len(ids) != artifact.row_count:
                errors.append(
                    BundleIssue(
                        code="index_schema_invalid",
                        message=f"向量行数与 Manifest 不一致：{file_name}",
                        file_name=file_name,
                    )
                )
        return BundleValidationReport(valid=not errors, errors=errors)


class IndexBundleWriter:
    def __init__(self, bundle_root: str | Path) -> None:
        self.bundle_root = Path(bundle_root).resolve()
        self.bundle_root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        dataset_id: str,
        index_version: str,
        parser_version: str,
        embedding_model: str | None,
        vector_dimension: int,
        sources: list[SourceManifest],
        text_chunks: list[dict[str, Any]],
        image_chunks: list[dict[str, Any]],
        assets: list[dict[str, Any]],
        text_vectors: dict[str, list[float]],
        image_caption_vectors: dict[str, list[float]],
        parent_index_version: str | None = None,
        incremental: IncrementalStats | None = None,
        evaluation_status: str = "not_run",
        approval_status: str = "awaiting_approval",
    ) -> IndexBundle:
        target = self.bundle_root / dataset_id / index_version
        if target.is_dir():
            existing = IndexBundle.load(target)
            report = existing.validate()
            if report.valid:
                return existing
            raise ValueError(f"existing index bundle is invalid: {target}")

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{index_version}-", dir=target.parent))
        try:
            rows_by_file = {
                "text_chunks.jsonl": text_chunks,
                "image_chunks.jsonl": image_chunks,
                "assets.jsonl": assets,
            }
            for file_name, rows in rows_by_file.items():
                self._write_jsonl(staging / file_name, rows)
            self._write_vectors(
                staging / "text_vectors.npz",
                text_vectors,
                vector_dimension,
            )
            self._write_vectors(
                staging / "image_caption_vectors.npz",
                image_caption_vectors,
                vector_dimension,
            )

            row_counts = {
                **{name: len(rows) for name, rows in rows_by_file.items()},
                "text_vectors.npz": len(text_vectors),
                "image_caption_vectors.npz": len(image_caption_vectors),
            }
            artifacts = {
                file_name: ArtifactManifest(
                    file_name=file_name,
                    sha256=_sha256(staging / file_name),
                    size_bytes=(staging / file_name).stat().st_size,
                    row_count=row_counts[file_name],
                )
                for file_name in (*_JSONL_ARTIFACTS, *_VECTOR_ARTIFACTS)
            }
            manifest = IndexManifest(
                dataset_id=dataset_id,
                index_version=index_version,
                parent_index_version=parent_index_version,
                built_at=datetime.now(UTC),
                parser_version=parser_version,
                embedding_model=embedding_model,
                vector_dimension=vector_dimension,
                sources=sorted(sources, key=lambda item: item.document_id),
                artifacts=artifacts,
                counts={
                    "sources": len(sources),
                    "text_chunks": len(text_chunks),
                    "parent_chunks": sum(
                        1 for item in text_chunks if item.get("chunk_type") == "parent"
                    ),
                    "child_chunks": sum(
                        1 for item in text_chunks if item.get("chunk_type") == "child"
                    ),
                    "image_chunks": len(image_chunks),
                    "assets": len(assets),
                    "text_vectors": len(text_vectors),
                    "image_caption_vectors": len(image_caption_vectors),
                },
                incremental=incremental or IncrementalStats(),
                evaluation_status=evaluation_status,
                approval_status=approval_status,
            )
            (staging / "manifest.json").write_text(
                json.dumps(
                    manifest.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            checksum_names = [*_JSONL_ARTIFACTS, *_VECTOR_ARTIFACTS, "manifest.json"]
            (staging / "checksums.sha256").write_text(
                "".join(f"{_sha256(staging / name)}  {name}\n" for name in checksum_names),
                encoding="utf-8",
            )
            report = IndexBundle(staging, manifest).validate()
            if not report.valid:
                messages = "; ".join(item.message for item in report.errors)
                raise ValueError(f"index bundle validation failed: {messages}")
            os.replace(staging, target)
            return IndexBundle.load(target)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )

    @staticmethod
    def _write_vectors(path: Path, values: dict[str, list[float]], dimension: int) -> None:
        if dimension < 0:
            raise ValueError("vector dimension must be non-negative")
        ids = sorted(values)
        matrix: list[list[float]] = []
        for item_id in ids:
            vector = values[item_id]
            if len(vector) != dimension or any(not math.isfinite(float(item)) for item in vector):
                raise ValueError(f"vector has invalid dimension or value: {item_id}")
            matrix.append([float(item) for item in vector])
        array = np.asarray(matrix, dtype=np.float32)
        if not matrix:
            array = np.empty((0, dimension), dtype=np.float32)
        np.savez_compressed(path, ids=np.asarray(ids, dtype=np.str_), vectors=array)
