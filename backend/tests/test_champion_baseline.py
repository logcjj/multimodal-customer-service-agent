from __future__ import annotations

import csv
import json

from app.evaluation.champion_baseline import (
    build_champion_snapshot,
    load_public_questions,
    verify_answer_runner,
    verify_champion_snapshot,
)


class FakeChampion:
    def __init__(self, suffix: str = "") -> None:
        self.suffix = suffix

    def answer(self, question: str, images: list[str]) -> str:
        assert images == []
        return f"冻结答案：{question}{self.suffix}"


def _questions(path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["id", "question"])
        writer.writeheader()
        writer.writerow({"id": "1", "question": "第一问\n第二行"})
        writer.writerow({"id": "2", "question": "第二问"})


def test_champion_snapshot_is_deterministic_and_verifiable(tmp_path) -> None:
    question_path = tmp_path / "question_public.csv"
    snapshot_path = tmp_path / "champion.jsonl"
    manifest_path = tmp_path / "champion.manifest.json"
    _questions(question_path)

    manifest = build_champion_snapshot(
        champion=FakeChampion(),
        question_path=question_path,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        generator_id="fake-champion-v1",
    )
    report = verify_champion_snapshot(
        champion=FakeChampion(),
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
    )

    assert [item.question for item in load_public_questions(question_path)] == [
        "第一问\n第二行",
        "第二问",
    ]
    assert manifest.case_count == 2
    assert report.valid is True
    assert report.exact_matches == 2
    assert report.mismatches == []
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["generator_id"] == (
        "fake-champion-v1"
    )


def test_champion_snapshot_reports_answer_drift(tmp_path) -> None:
    question_path = tmp_path / "question_public.csv"
    snapshot_path = tmp_path / "champion.jsonl"
    manifest_path = tmp_path / "champion.manifest.json"
    _questions(question_path)
    build_champion_snapshot(
        champion=FakeChampion(),
        question_path=question_path,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        generator_id="fake-champion-v1",
    )

    report = verify_champion_snapshot(
        champion=FakeChampion("-changed"),
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
    )

    assert report.valid is False
    assert report.exact_matches == 0
    assert report.mismatches == ["1", "2"]


def test_answer_runner_can_be_compared_with_frozen_snapshot(tmp_path) -> None:
    question_path = tmp_path / "question_public.csv"
    snapshot_path = tmp_path / "champion.jsonl"
    manifest_path = tmp_path / "champion.manifest.json"
    _questions(question_path)
    build_champion_snapshot(
        champion=FakeChampion(),
        question_path=question_path,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        generator_id="fake-champion-v1",
    )

    report = verify_answer_runner(
        snapshot_path=snapshot_path,
        answer_runner=lambda question: (f"冻结答案：{question}", True),
        rollout_mode="legacy_only",
    )

    assert report.valid is True
    assert report.exact_matches == 2
    assert report.legacy_used == 2
    assert report.mismatches == []


def test_answer_runner_rejects_tampered_snapshot_before_running_answers(tmp_path) -> None:
    question_path = tmp_path / "question_public.csv"
    snapshot_path = tmp_path / "champion.jsonl"
    manifest_path = tmp_path / "champion.manifest.json"
    _questions(question_path)
    build_champion_snapshot(
        champion=FakeChampion(),
        question_path=question_path,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        generator_id="fake-champion-v1",
    )
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8").replace("第一问", "被篡改的问题", 1),
        encoding="utf-8",
    )

    def must_not_run(question: str) -> tuple[str, bool]:
        raise AssertionError(f"tampered baseline must not execute: {question}")

    report = verify_answer_runner(
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        answer_runner=must_not_run,
        rollout_mode="legacy_only",
    )

    assert report.valid is False
    assert report.exact_matches == 0
    assert report.legacy_used == 0
    assert "snapshot_checksum_mismatch" in report.errors
