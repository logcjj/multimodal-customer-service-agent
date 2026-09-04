from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


SCHEMA_VERSION = "aka-champion-public-baseline-v1"


class Champion(Protocol):
    def answer(self, question: str, images: list[str]) -> str: ...


class PublicQuestion(BaseModel):
    id: str
    question: str


class ChampionSnapshotManifest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    generator_id: str
    question_source_sha256: str
    snapshot_sha256: str
    answer_digest: str
    case_count: int = Field(ge=0)
    built_at: datetime


class ChampionVerificationReport(BaseModel):
    valid: bool
    case_count: int
    exact_matches: int
    mismatches: list[str] = Field(default_factory=list)
    empty_answers: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ChampionModeVerificationReport(BaseModel):
    valid: bool
    rollout_mode: str
    case_count: int
    exact_matches: int
    legacy_used: int
    mismatches: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def load_public_questions(path: str | Path) -> list[PublicQuestion]:
    resolved = Path(path).resolve()
    with resolved.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    questions: list[PublicQuestion] = []
    seen: set[str] = set()
    for row in rows:
        question_id = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        if not question_id or not question:
            raise ValueError("public question rows require non-empty id and question")
        if question_id in seen:
            raise ValueError(f"duplicate public question id: {question_id}")
        seen.add(question_id)
        questions.append(PublicQuestion(id=question_id, question=question))
    return questions


def build_champion_snapshot(
    *,
    champion: Champion,
    question_path: str | Path,
    snapshot_path: str | Path,
    manifest_path: str | Path,
    generator_id: str,
) -> ChampionSnapshotManifest:
    questions = load_public_questions(question_path)
    rows: list[dict[str, str]] = []
    for item in questions:
        answer = _normalize_answer(champion.answer(item.question, []))
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "question_sha256": _text_sha256(item.question),
                "answer": answer,
                "answer_sha256": _text_sha256(answer),
            }
        )

    resolved_snapshot = Path(snapshot_path).resolve()
    resolved_manifest = Path(manifest_path).resolve()
    _write_jsonl_atomic(resolved_snapshot, rows)
    manifest = ChampionSnapshotManifest(
        generator_id=generator_id,
        question_source_sha256=_file_sha256(Path(question_path).resolve()),
        snapshot_sha256=_file_sha256(resolved_snapshot),
        answer_digest=_answer_digest(rows),
        case_count=len(rows),
        built_at=datetime.now(UTC),
    )
    _write_text_atomic(
        resolved_manifest,
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    return manifest


def verify_champion_snapshot(
    *,
    champion: Champion,
    snapshot_path: str | Path,
    manifest_path: str | Path,
) -> ChampionVerificationReport:
    resolved_snapshot = Path(snapshot_path).resolve()
    resolved_manifest = Path(manifest_path).resolve()
    manifest = ChampionSnapshotManifest.model_validate_json(
        resolved_manifest.read_text(encoding="utf-8")
    )
    errors: list[str] = []
    if manifest.schema_version != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if _file_sha256(resolved_snapshot) != manifest.snapshot_sha256:
        errors.append("snapshot_checksum_mismatch")
    rows = _read_jsonl(resolved_snapshot)
    if len(rows) != manifest.case_count:
        errors.append("case_count_mismatch")
    if _answer_digest(rows) != manifest.answer_digest:
        errors.append("answer_digest_mismatch")

    mismatches: list[str] = []
    empty_answers: list[str] = []
    for row in rows:
        question_id = str(row.get("id") or "")
        question = str(row.get("question") or "")
        expected_hash = str(row.get("answer_sha256") or "")
        answer = _normalize_answer(champion.answer(question, []))
        if not answer:
            empty_answers.append(question_id)
        if _text_sha256(answer) != expected_hash:
            mismatches.append(question_id)
    exact_matches = len(rows) - len(mismatches)
    return ChampionVerificationReport(
        valid=not errors and not mismatches and not empty_answers,
        case_count=len(rows),
        exact_matches=exact_matches,
        mismatches=mismatches,
        empty_answers=empty_answers,
        errors=errors,
    )


def verify_answer_runner(
    *,
    snapshot_path: str | Path,
    manifest_path: str | Path | None = None,
    answer_runner: Callable[[str], tuple[str, bool]],
    rollout_mode: str,
) -> ChampionModeVerificationReport:
    resolved_snapshot = Path(snapshot_path).resolve()
    rows = _read_jsonl(resolved_snapshot)
    errors: list[str] = []
    if manifest_path is not None:
        manifest = ChampionSnapshotManifest.model_validate_json(
            Path(manifest_path).resolve().read_text(encoding="utf-8")
        )
        if manifest.schema_version != SCHEMA_VERSION:
            errors.append("schema_version_mismatch")
        if _file_sha256(resolved_snapshot) != manifest.snapshot_sha256:
            errors.append("snapshot_checksum_mismatch")
        if len(rows) != manifest.case_count:
            errors.append("case_count_mismatch")
        if _answer_digest(rows) != manifest.answer_digest:
            errors.append("answer_digest_mismatch")
    if errors:
        return ChampionModeVerificationReport(
            valid=False,
            rollout_mode=rollout_mode,
            case_count=len(rows),
            exact_matches=0,
            legacy_used=0,
            errors=errors,
        )
    mismatches: list[str] = []
    legacy_used = 0
    for row in rows:
        question_id = str(row.get("id") or "")
        question = str(row.get("question") or "")
        expected_hash = str(row.get("answer_sha256") or "")
        answer, used_legacy = answer_runner(question)
        legacy_used += int(used_legacy)
        if _text_sha256(_normalize_answer(answer)) != expected_hash:
            mismatches.append(question_id)
    return ChampionModeVerificationReport(
        valid=not mismatches,
        rollout_mode=rollout_mode,
        case_count=len(rows),
        exact_matches=len(rows) - len(mismatches),
        legacy_used=legacy_used,
        mismatches=mismatches,
        errors=errors,
    )


def _normalize_answer(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("champion snapshot rows must be JSON objects")
        rows.append(value)
    return rows


def _answer_digest(rows: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row.get("id") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("answer_sha256") or "").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    _write_text_atomic(path, content)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
