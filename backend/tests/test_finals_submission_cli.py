from __future__ import annotations

from app.evaluation.finals_public import FinalsQuestion
from scripts.finals_submission import _run_case, parse_args


def test_finals_submission_cli_accepts_public_set_and_output_paths(tmp_path) -> None:
    questions = tmp_path / "evaluation_public.csv"
    output = tmp_path / "submission.csv"
    report = tmp_path / "report.json"

    args = parse_args(
        [
            "--question-path",
            str(questions),
            "--output",
            str(output),
            "--report-path",
            str(report),
            "--api-url",
            "http://127.0.0.1:8002/api/chat",
            "--workers",
            "2",
        ]
    )

    assert args.question_path == questions
    assert args.output == output
    assert args.report_path == report
    assert args.api_url == "http://127.0.0.1:8002/api/chat"
    assert args.workers == 2


def test_finals_case_uses_run_scoped_session_to_avoid_previous_evaluation_memory(
    monkeypatch,
) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "answer": "本轮独立回答",
                "assets": [],
                "verification": {"passed": True, "action": "accept", "issues": []},
                "citations": [],
                "trace": {"selected_agents": [], "total_latency_ms": 1},
            }

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def post(self, url, json):
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr("scripts.finals_submission.httpx.Client", FakeClient)

    _run_case(
        FinalsQuestion(id="1", question="测试问题"),
        api_url="http://127.0.0.1:8002/api/chat",
        timeout=10,
        asset_to_image={},
        max_manual_images=3,
        run_id="20260725T010203Z",
    )

    assert captured["payload"]["session_id"] == "finals-public-20260725T010203Z-1"
