from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Any

from audit_submission import audit_submission, parse_ret
from generate_submission import format_competition_ret
from llm_judge_submission import evidence_for_question, judge_row

from df_kefu_baseline.answer import AnswerEngine
from df_kefu_baseline.config import PROJECT_ROOT, QUESTION_PATH
from df_kefu_baseline.llm_client import OpenAICompatibleClient
from df_kefu_baseline.manuals import image_path_for_id
from df_kefu_baseline.submission_format import sanitize_submission_text


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)
PIC_RE = re.compile(r"<PIC>")


def read_questions(path: Path = QUESTION_PATH) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["id"]: row["question"] for row in csv.DictReader(f)}


def read_submission(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["id"]: row["ret"] for row in csv.DictReader(f)}


def write_submission(rows_by_id: dict[str, str], questions: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"], lineterminator="\n")
        writer.writeheader()
        for qid in questions:
            writer.writerow({"id": qid, "ret": sanitize_submission_text(rows_by_id[qid])})


def parse_json_object(raw: str) -> dict[str, Any]:
    match = JSON_OBJECT_RE.search(raw)
    if not match:
        raise ValueError(f"No JSON object found: {raw[:200]}")
    return json.loads(match.group(0))


def build_rewrite_messages(
    question: str,
    answer: str,
    images: list[str],
    evidence: str,
    judge_result: dict[str, Any],
) -> list[dict[str, str]]:
    system = (
        "你是 DataFountain 多模态客服智能体的答案优化器。"
        "你必须只依据给定证据改写答案，禁止编造证据中没有的参数、承诺、型号或步骤。"
        "目标是提高离线 judge 和线上评测可能得分：答题直接、产品正确、步骤完整、语言匹配、格式合规。"
        "如果证据中有必要图片，可在答案中使用 <PIC>，并在 image_ids 中列出对应图片 ID；"
        "<PIC> 数量必须和 image_ids 数量一致。输出必须是 JSON 对象，不要 Markdown。"
    )
    user = {
        "question": question,
        "current_answer": answer,
        "current_images": images,
        "retrieved_evidence": evidence,
        "judge_result": judge_result,
        "required_json_schema": {
            "answer": "rewritten final answer string",
            "image_ids": ["image ids used by <PIC>, may be empty"],
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def normalize_rewrite(answer: str, image_ids: list[str]) -> str:
    clean_ids: list[str] = []
    for image_id in image_ids:
        image_id = str(image_id).strip()
        if image_id and image_id not in clean_ids and image_path_for_id(image_id) is not None:
            clean_ids.append(image_id)

    pic_count = len(PIC_RE.findall(answer))
    clean_ids = clean_ids[:pic_count]
    if len(clean_ids) < pic_count:
        keep = len(clean_ids)
        seen = 0

        def keep_available_pic(match: re.Match[str]) -> str:
            nonlocal seen
            seen += 1
            return match.group(0) if seen <= keep else ""

        answer = PIC_RE.sub(keep_available_pic, answer)

    answer_with_footer = answer.strip()
    if clean_ids:
        answer_with_footer += "\nRelated images: " + ", ".join(clean_ids)
    return format_competition_ret(answer_with_footer)


def rewrite_answer(
    client: OpenAICompatibleClient,
    question: str,
    ret: str,
    evidence: str,
    judge_result: dict[str, Any],
    retries: int = 2,
) -> str:
    answer, images, _ = parse_ret(ret)
    messages = build_rewrite_messages(question, answer, images, evidence, judge_result)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            raw = client.chat(messages, temperature=0.1)
            data = parse_json_object(raw)
            new_answer = str(data.get("answer", "")).strip()
            new_images = data.get("image_ids", [])
            if not new_answer:
                raise ValueError("empty rewritten answer")
            if not isinstance(new_images, list):
                new_images = []
            return normalize_rewrite(new_answer, [str(item) for item in new_images])
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"rewrite failed: {last_error}")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def selected_items(questions: dict[str, str], args: argparse.Namespace) -> list[tuple[str, str]]:
    items = list(questions.items())
    if args.start_id:
        items = [item for item in items if int(item[0]) >= args.start_id]
    if args.end_id:
        items = [item for item in items if int(item[0]) <= args.end_id]
    if args.limit:
        items = items[: args.limit]
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM judge -> rewrite -> judge optimization loops for a submission CSV.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "submissions" / "llm_loop")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "llm_loop")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=82.0)
    parser.add_argument("--max-fixes", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--evidence-limit", type=int, default=5)
    parser.add_argument("--min-improvement", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = resolve_path(args.submission)
    work_dir = resolve_path(args.work_dir)
    report_dir = resolve_path(args.report_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    questions = read_questions()
    items = selected_items(questions, args)
    scoped_questions = {qid: question for qid, question in items}
    current_rows = read_submission(source_path)
    missing = [qid for qid in scoped_questions if qid not in current_rows]
    if missing:
        raise RuntimeError(f"submission is missing ids: {missing[:20]}")

    client = OpenAICompatibleClient()
    engine = AnswerEngine(use_llm=False)

    current_path = work_dir / "iter0.csv"
    if set(scoped_questions) == set(questions):
        shutil.copyfile(source_path, current_path)
    else:
        write_submission(current_rows, scoped_questions, current_path)

    best_path = current_path
    best_avg = -1.0
    summary_rows: list[dict[str, Any]] = []

    for iteration in range(1, args.iterations + 1):
        print(f"=== iteration {iteration}: judging {current_path} ===")
        judge_rows: list[dict[str, Any]] = []
        for idx, (qid, question) in enumerate(items, start=1):
            row = judge_row(
                client,
                engine,
                qid,
                question,
                current_rows[qid],
                evidence_limit=args.evidence_limit,
            )
            judge_rows.append(row)
            print(f"[judge {idx}/{len(items)}] id={qid} score={row['score']} risk={row['risk']}")
            if args.delay > 0:
                time.sleep(args.delay)

        judge_path = report_dir / f"judge_iter{iteration}.csv"
        with judge_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(judge_rows[0].keys()))
            writer.writeheader()
            writer.writerows(judge_rows)

        scores = [float(row["score"]) for row in judge_rows]
        avg_score = mean(scores) if scores else 0.0
        risk_bad = sum(row["risk"] == "bad" for row in judge_rows)
        risk_review = sum(row["risk"] == "review" for row in judge_rows)
        if avg_score > best_avg + args.min_improvement:
            best_avg = avg_score
            best_path = current_path

        candidates = sorted(
            [
                row
                for row in judge_rows
                if float(row["score"]) < args.threshold or row["risk"] in {"bad", "review"}
            ],
            key=lambda row: float(row["score"]),
        )[: args.max_fixes]
        print(
            f"iteration={iteration} avg={avg_score:.2f} bad={risk_bad} review={risk_review} "
            f"fix_candidates={len(candidates)}"
        )

        summary_rows.append(
            {
                "iteration": iteration,
                "submission": str(current_path),
                "judge_report": str(judge_path),
                "avg_score": f"{avg_score:.2f}",
                "bad": risk_bad,
                "review": risk_review,
                "fix_candidates": len(candidates),
            }
        )
        if not candidates:
            print("No low-score candidates; stopping.")
            break

        next_rows = dict(current_rows)
        changed = 0
        for idx, row in enumerate(candidates, start=1):
            qid = row["id"]
            question = scoped_questions[qid]
            evidence = evidence_for_question(engine, question, limit=args.evidence_limit)
            try:
                next_rows[qid] = rewrite_answer(client, question, current_rows[qid], evidence, row)
                changed += 1
                print(f"[rewrite {idx}/{len(candidates)}] id={qid} updated")
            except Exception as exc:
                print(f"[rewrite {idx}/{len(candidates)}] id={qid} skipped: {exc}")
            if args.delay > 0:
                time.sleep(args.delay)

        if changed == 0:
            print("No answers were rewritten; stopping.")
            break

        next_path = work_dir / f"iter{iteration}.csv"
        write_submission(next_rows, scoped_questions, next_path)
        audit_path = report_dir / f"audit_iter{iteration}.csv"
        counts, _ = audit_submission(next_path, audit_path)
        print(f"audit normal={counts['姝ｅ父']} medium={counts['涓瓑']} severe={counts['涓ラ噸']} saved={audit_path}")
        current_rows = next_rows
        current_path = next_path

    summary_path = report_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["iteration", "submission", "judge_report", "avg_score", "bad", "review", "fix_candidates"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"best={best_path} best_avg={best_avg:.2f}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
