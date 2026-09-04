from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlmodel import select

from app.evaluation.finals_public import (
    FinalsQuestion,
    download_public_image,
    format_competition_ret,
    load_finals_questions,
)
from app.knowledge.models import ImageChunkRecord
from app.storage.database import Database


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    backend_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="用真实 V3.3 API 运行决赛公开题并生成官方 id,ret CSV",
    )
    parser.add_argument("--question-path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=backend_root / "data/evaluation-reports/finals-public-50-agent-first.csv",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=backend_root / "data/evaluation-reports/finals-public-50-agent-first.json",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8002/api/chat")
    parser.add_argument("--data-dir", type=Path, default=backend_root / "data")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-manual-images", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    question_path = args.question_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report_path.expanduser().resolve()
    questions = load_finals_questions(question_path)
    if args.limit > 0:
        questions = questions[: args.limit]
    if not questions:
        print("status=failed error=no_questions")
        return 1
    built_at = datetime.now(UTC)
    run_id = built_at.strftime("%Y%m%dT%H%M%S%fZ")
    asset_to_image = _asset_to_image_id(args.data_dir.expanduser().resolve())
    indexed_questions = list(enumerate(questions, start=1))
    completed: dict[int, tuple[str, str, dict[str, object]]] = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(args.workers, 4)),
        thread_name_prefix="finals-public",
    ) as executor:
        futures = {
            executor.submit(
                _run_case,
                item,
                api_url=args.api_url,
                timeout=args.timeout,
                asset_to_image=asset_to_image,
                max_manual_images=max(0, args.max_manual_images),
                run_id=run_id,
            ): (position, item)
            for position, item in indexed_questions
        }
        for future in as_completed(futures):
            position, item = futures[future]
            try:
                qid, ret, detail = future.result()
            except Exception as exc:
                qid = item.id
                ret = ""
                detail = {
                    "id": item.id,
                    "question": item.question,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
            completed[position] = (qid, ret, detail)
            print(
                f"[{len(completed)}/{len(questions)}] id={qid} "
                f"status={detail.get('status')} verified={detail.get('verified', False)}",
                flush=True,
            )

    ordered = [completed[position] for position, _ in indexed_questions]
    _write_submission_atomic(output_path, [(qid, ret) for qid, ret, _ in ordered])
    details = [detail for _, _, detail in ordered]
    failed = [item for item in details if item.get("status") != "completed"]
    report = {
        "schema_version": "aka-finals-public-evaluation-v1",
        "built_at": built_at.isoformat(),
        "run_id": run_id,
        "question_path": str(question_path),
        "question_source_sha256": _file_sha256(question_path),
        "case_count": len(questions),
        "api_url": args.api_url,
        "official_score": None,
        "official_score_status": "requires_platform_submission",
        "metrics": {
            "completed": len(details) - len(failed),
            "failed": len(failed),
            "verified": sum(bool(item.get("verified")) for item in details),
            "used_legacy": sum(bool(item.get("used_legacy")) for item in details),
            "with_citations": sum(int(item.get("citation_count", 0)) > 0 for item in details),
            "image_questions": sum(bool(item.image_urls) for item in questions),
            "images_downloaded": sum(int(item.get("images_downloaded", 0)) for item in details),
            "images_failed": sum(int(item.get("images_failed", 0)) for item in details),
        },
        "details": details,
    }
    _write_json_atomic(report_path, report)
    print(
        f"status={'ready' if not failed else 'failed'} cases={len(questions)} "
        f"failed={len(failed)} submission={output_path} report={report_path}"
    )
    return 0 if not failed else 1


def _run_case(
    question: FinalsQuestion,
    *,
    api_url: str,
    timeout: float,
    asset_to_image: dict[str, str],
    max_manual_images: int,
    run_id: str,
) -> tuple[str, str, dict[str, object]]:
    data_urls: list[str] = []
    image_details: list[dict[str, object]] = []
    for url in question.image_urls:
        try:
            downloaded = download_public_image(url)
        except Exception as exc:
            image_details.append(
                {
                    "source_url": url,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            )
            continue
        data_urls.append(downloaded.data_url)
        image_details.append(
            {
                "source_url": url,
                "resolved_url": downloaded.resolved_url,
                "status": "downloaded",
                "mime_type": downloaded.mime_type,
                "sha256": downloaded.sha256,
                "size_bytes": downloaded.size_bytes,
            }
        )

    payload = {
        "question": question.question,
        "images": data_urls[:3],
        "session_id": f"finals-public-{run_id}-{question.id}",
        "deadline_ms": 120_000,
    }
    body: dict[str, object] | None = None
    last_error: Exception | None = None
    for _ in range(2):
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                response = client.post(api_url, json=payload)
                response.raise_for_status()
                value = response.json()
            if not isinstance(value, dict):
                raise ValueError("chat API response must be a JSON object")
            body = value
            break
        except Exception as exc:
            last_error = exc
    if body is None:
        assert last_error is not None
        raise last_error

    answer = str(body.get("answer") or "").strip()
    if not answer:
        raise ValueError("chat API returned an empty answer")
    response_assets = [str(item) for item in body.get("assets", []) if str(item)]
    manual_image_ids = list(
        dict.fromkeys(
            asset_to_image[item]
            for item in response_assets
            if item in asset_to_image
        )
    )[:max_manual_images]
    ret = format_competition_ret(
        answer,
        manual_image_ids,
        max_images=max_manual_images,
    )
    verification = body.get("verification") if isinstance(body.get("verification"), dict) else {}
    citations = body.get("citations") if isinstance(body.get("citations"), list) else []
    trace = body.get("trace") if isinstance(body.get("trace"), dict) else {}
    detail = {
        "id": question.id,
        "question": question.question,
        "status": "completed",
        "route": body.get("route"),
        "used_legacy": bool(body.get("used_legacy")),
        "verified": bool(verification.get("passed")),
        "verification_action": verification.get("action"),
        "verification_issue_codes": [
            str(item.get("code"))
            for item in verification.get("issues", [])
            if isinstance(item, dict) and item.get("code")
        ],
        "citation_count": len(citations),
        "citation_sources": [
            {
                "source_type": item.get("source_type"),
                "document_name": item.get("document_name"),
                "chapter_title": item.get("chapter_title"),
                "page_start": item.get("page_start"),
                "score": item.get("score"),
            }
            for item in citations
            if isinstance(item, dict)
        ],
        "selected_agents": trace.get("selected_agents", []),
        "total_latency_ms": trace.get("total_latency_ms"),
        "answer_length": len(answer),
        "manual_image_ids": manual_image_ids,
        "images_downloaded": len(data_urls),
        "images_failed": len(question.image_urls) - len(data_urls),
        "input_images": image_details,
    }
    return question.id, ret, detail


def _asset_to_image_id(data_dir: Path) -> dict[str, str]:
    database = Database(data_dir, database_url=None)
    try:
        with database.session() as session:
            records = session.exec(
                select(ImageChunkRecord).where(ImageChunkRecord.enabled == True)  # noqa: E712
            ).all()
            return {
                record.asset_id: record.image_id
                for record in records
                if record.asset_id and record.image_id
            }
    finally:
        database.engine.dispose()


def _write_submission_atomic(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=["id", "ret"], lineterminator="\n")
            writer.writeheader()
            for question_id, ret in rows:
                writer.writerow({"id": question_id, "ret": ret})
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
