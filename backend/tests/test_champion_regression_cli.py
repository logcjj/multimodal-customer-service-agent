from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from types import SimpleNamespace

from app.evaluation.champion_baseline import build_champion_snapshot
from scripts import champion_regression
from scripts.champion_regression import parse_args, verify_orchestrator_snapshot


class FakeChampion:
    def answer(self, question: str, images: list[str]) -> str:
        assert images == []
        return f"冻结答案：{question}"


class FakeOrchestrator:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def run(self, request):
        self.questions.append(request.question)
        return SimpleNamespace(answer=f"冻结答案：{request.question}", used_legacy=True)


def test_cli_accepts_real_orchestrator_regression_options(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    data_dir = tmp_path / "runtime-data"

    args = parse_args(
        [
            "verify-orchestrator",
            "--data-dir",
            str(data_dir),
            "--report-path",
            str(report_path),
            "--rollout-mode",
            "legacy_only",
            "--required-dataset-id",
            "v6-manuals",
        ]
    )

    assert args.mode == "verify-orchestrator"
    assert args.data_dir == data_dir
    assert args.report_path == report_path
    assert args.rollout_mode == "legacy_only"
    assert args.required_dataset_id == "v6-manuals"


def test_real_orchestrator_verification_writes_machine_readable_report(tmp_path) -> None:
    question_path = tmp_path / "question_public.csv"
    snapshot_path = tmp_path / "champion.jsonl"
    manifest_path = tmp_path / "champion.manifest.json"
    report_path = tmp_path / "reports" / "legacy-only.json"
    with question_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["id", "question"])
        writer.writeheader()
        writer.writerow({"id": "1", "question": "第一问"})
        writer.writerow({"id": "2", "question": "第二问"})
    build_champion_snapshot(
        champion=FakeChampion(),
        question_path=question_path,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        generator_id="fake-v1",
    )
    orchestrator = FakeOrchestrator()

    report = verify_orchestrator_snapshot(
        orchestrator=orchestrator,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        report_path=report_path,
        rollout_mode="legacy_only",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.valid is True
    assert orchestrator.questions == ["第一问", "第二问"]
    assert payload["rollout_mode"] == "legacy_only"
    assert payload["case_count"] == 2
    assert payload["exact_matches"] == 2
    assert payload["legacy_used"] == 2


def test_cli_runs_orchestrator_mode_through_offline_runtime(tmp_path, monkeypatch) -> None:
    question_path = tmp_path / "question_public.csv"
    snapshot_path = tmp_path / "champion.jsonl"
    manifest_path = tmp_path / "champion.manifest.json"
    report_path = tmp_path / "report.json"
    data_dir = tmp_path / "runtime-data"
    data_dir.mkdir()
    with question_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["id", "question"])
        writer.writeheader()
        writer.writerow({"id": "1", "question": "第一问"})
    build_champion_snapshot(
        champion=FakeChampion(),
        question_path=question_path,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        generator_id="fake-v1",
    )
    opened: list[tuple[object, ...]] = []

    @contextmanager
    def fake_runtime(*, runtime_data_dir, rollout_mode, required_dataset_id):
        opened.append((runtime_data_dir, rollout_mode, required_dataset_id))
        yield FakeOrchestrator()

    monkeypatch.setattr(
        champion_regression,
        "open_offline_orchestrator",
        fake_runtime,
        raising=False,
    )

    exit_code = champion_regression.main(
        [
            "verify-orchestrator",
            "--snapshot-path",
            str(snapshot_path),
            "--manifest-path",
            str(manifest_path),
            "--data-dir",
            str(data_dir),
            "--report-path",
            str(report_path),
            "--required-dataset-id",
            "v6-manuals",
        ]
    )

    assert exit_code == 0
    assert opened == [(data_dir.resolve(), "legacy_only", "v6-manuals")]
    assert json.loads(report_path.read_text(encoding="utf-8"))["valid"] is True
