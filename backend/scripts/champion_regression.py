from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from app.compatibility.legacy_champion import LegacyChampionAdapter
from app.config.runtime import RuntimeSettings
from app.contracts.models import AgentRequest, AgentResponse
from app.evaluation.champion_baseline import (
    ChampionModeVerificationReport,
    build_champion_snapshot,
    verify_answer_runner,
    verify_champion_snapshot,
)
from app.knowledge.service import KnowledgeService, LiveKnowledgeRetriever
from app.observability.traces import TraceStore
from app.runtime.orchestrator import Orchestrator
from app.storage.database import Database


class AnswerOrchestrator(Protocol):
    def run(self, request: AgentRequest) -> AgentResponse: ...


@contextmanager
def open_offline_orchestrator(
    *,
    runtime_data_dir: str | Path,
    rollout_mode: str,
    required_dataset_id: str,
):
    data_dir = Path(runtime_data_dir).expanduser().resolve()
    settings = replace(
        RuntimeSettings.from_env(),
        database_url=None,
        offline_index_mode="on",
        caption_embedding="off",
        ocr_pipeline="off",
        session_memory="off",
    )
    runtime_database = Database(data_dir, database_url=None)
    knowledge = KnowledgeService(runtime_database, settings=settings)
    knowledge.embed_override = None
    knowledge.rerank_override = None
    runtime_status = knowledge.preload_active_bundles()
    required_status = runtime_status.get(required_dataset_id, {})
    if required_status.get("status") != "ready":
        knowledge.shutdown()
        runtime_database.engine.dispose()
        raise RuntimeError(
            f"required offline index is not ready: {required_dataset_id} "
            f"({required_status.get('status', 'missing')})"
        )

    with tempfile.TemporaryDirectory(prefix="aka-champion-traces-") as trace_root:
        trace_database = Database(Path(trace_root), database_url=None)
        orchestrator = Orchestrator(
            retriever=LiveKnowledgeRetriever(knowledge),
            trace_store=TraceStore(trace_database),
            legacy=LegacyChampionAdapter(),
            rollout_mode=rollout_mode,
            llm_gateway=None,
            settings=settings,
            session_memory=None,
        )
        if not orchestrator.legacy.available:
            knowledge.shutdown()
            runtime_database.engine.dispose()
            trace_database.engine.dispose()
            raise RuntimeError("legacy champion is unavailable")
        try:
            yield orchestrator
        finally:
            knowledge.shutdown()
            trace_database.engine.dispose()
            runtime_database.engine.dispose()


def verify_orchestrator_snapshot(
    *,
    orchestrator: AnswerOrchestrator,
    snapshot_path: str | Path,
    manifest_path: str | Path,
    report_path: str | Path,
    rollout_mode: str,
) -> ChampionModeVerificationReport:
    def answer_runner(question: str) -> tuple[str, bool]:
        response = orchestrator.run(AgentRequest(question=question, deadline_ms=120_000))
        return response.answer, response.used_legacy

    report = verify_answer_runner(
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        answer_runner=answer_runner,
        rollout_mode=rollout_mode,
    )
    resolved_report = Path(report_path).expanduser().resolve()
    resolved_report.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved_report.name}.",
        dir=resolved_report.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            target.write(
                json.dumps(
                    report.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, resolved_report)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    backend_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="冻结并验证 aka Champion 的公开 400 题本地基线")
    parser.add_argument("mode", choices=("build", "verify", "verify-orchestrator"))
    parser.add_argument("--question-path", type=Path)
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=backend_root / "data/evaluation-baselines/champion-public-400.jsonl",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=backend_root / "data/evaluation-baselines/champion-public-400.manifest.json",
    )
    parser.add_argument(
        "--generator-id",
        default="aka-legacy-answer-engine-no-llm-v1",
    )
    parser.add_argument("--data-dir", type=Path, default=backend_root / "data")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=backend_root
        / "data/evaluation-reports/champion-legacy-only-public-400.json",
    )
    parser.add_argument("--rollout-mode", choices=("legacy_only",), default="legacy_only")
    parser.add_argument("--required-dataset-id", default="v6-manuals")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "verify-orchestrator":
        try:
            with open_offline_orchestrator(
                runtime_data_dir=args.data_dir.expanduser().resolve(),
                rollout_mode=args.rollout_mode,
                required_dataset_id=args.required_dataset_id,
            ) as orchestrator:
                report = verify_orchestrator_snapshot(
                    orchestrator=orchestrator,
                    snapshot_path=args.snapshot_path.expanduser().resolve(),
                    manifest_path=args.manifest_path.expanduser().resolve(),
                    report_path=args.report_path.expanduser().resolve(),
                    rollout_mode=args.rollout_mode,
                )
        except Exception as exc:
            print(f"status=failed error={type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(
            f"status={'passed' if report.valid else 'failed'} "
            f"mode={report.rollout_mode} cases={report.case_count} "
            f"exact_matches={report.exact_matches} mismatches={len(report.mismatches)} "
            f"legacy_used={report.legacy_used} report={args.report_path.expanduser().resolve()}"
        )
        return 0 if report.valid else 1

    champion = LegacyChampionAdapter()
    if not champion.available or champion.source_root is None:
        print("status=failed error=legacy_champion_unavailable", file=sys.stderr)
        return 1
    question_path = (
        args.question_path.expanduser().resolve()
        if args.question_path
        else champion.source_root.parent / "question_public.csv"
    )
    snapshot_path = args.snapshot_path.expanduser().resolve()
    manifest_path = args.manifest_path.expanduser().resolve()
    try:
        if args.mode == "build":
            manifest = build_champion_snapshot(
                champion=champion,
                question_path=question_path,
                snapshot_path=snapshot_path,
                manifest_path=manifest_path,
                generator_id=args.generator_id,
            )
            print(
                "status=ready "
                f"cases={manifest.case_count} "
                f"answer_digest={manifest.answer_digest} "
                f"snapshot={snapshot_path}"
            )
            return 0
        report = verify_champion_snapshot(
            champion=champion,
            snapshot_path=snapshot_path,
            manifest_path=manifest_path,
        )
    except Exception as exc:
        print(f"status=failed error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        f"status={'passed' if report.valid else 'failed'} "
        f"cases={report.case_count} exact_matches={report.exact_matches} "
        f"mismatches={len(report.mismatches)} empty_answers={len(report.empty_answers)} "
        f"errors={','.join(report.errors) or 'none'}"
    )
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
