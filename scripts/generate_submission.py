from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from df_kefu_baseline.answer import AnswerEngine
from df_kefu_baseline.config import SUBMISSION_DIR
from df_kefu_baseline.data import read_questions, write_submission


IMAGE_FOOTER_RE = re.compile(r"^(?:相关插图|Related images):\s*(.+?)\s*$", re.MULTILINE)
PIC_RE = re.compile(r"<PIC>")


def format_competition_ret(answer: str) -> str:
    """Format ret as required by the competition: "answer", ["image_id"]."""
    image_ids: list[str] = []

    def collect_images(match: re.Match[str]) -> str:
        for image_id in re.split(r"\s*,\s*", match.group(1).strip()):
            image_id = image_id.strip()
            if image_id and image_id not in image_ids:
                image_ids.append(image_id)
        return ""

    clean_answer = IMAGE_FOOTER_RE.sub(collect_images, answer)
    clean_answer = re.sub(r"\n{3,}", "\n\n", clean_answer).strip()
    pic_count = len(PIC_RE.findall(clean_answer))
    image_ids = image_ids[:pic_count]
    if len(image_ids) < pic_count:
        keep = len(image_ids)
        seen = 0

        def keep_available_pic(match: re.Match[str]) -> str:
            nonlocal seen
            seen += 1
            return match.group(0) if seen <= keep else ""

        clean_answer = PIC_RE.sub(keep_available_pic, clean_answer)

    formatted = json.dumps(clean_answer, ensure_ascii=False)
    if image_ids and "<PIC>" in clean_answer:
        formatted += ", " + json.dumps(image_ids, ensure_ascii=False)
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DataFountain 1165 submission CSV.")
    parser.add_argument("--output", type=Path, default=SUBMISSION_DIR / "submission_baseline.csv")
    parser.add_argument("--use-llm", action="store_true", help="Use OpenAI-compatible API from environment variables.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N questions.")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between LLM calls in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = read_questions()
    if args.start_id:
        questions = [item for item in questions if int(item.id) >= args.start_id]
    if args.end_id:
        questions = [item for item in questions if int(item.id) <= args.end_id]
    if args.limit:
        questions = questions[: args.limit]

    engine = AnswerEngine(use_llm=args.use_llm)
    rows: list[tuple[str, str]] = []
    for idx, item in enumerate(questions, start=1):
        answer = engine.answer(item.question, qid=item.id)
        rows.append((item.id, format_competition_ret(answer)))
        print(f"[{idx}/{len(questions)}] id={item.id} done")
        if args.use_llm and args.delay > 0:
            time.sleep(args.delay)

    write_submission(rows, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
