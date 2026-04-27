from __future__ import annotations

import argparse

from df_kefu_baseline.answer import AnswerEngine, readable_chunk_text
from df_kefu_baseline.data import read_questions
from df_kefu_baseline.query_planner import build_query_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect query planning and retrieval evidence.")
    parser.add_argument("--id", default="", help="Question id from question_public.csv.")
    parser.add_argument("--question", default="", help="Ad-hoc question text.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.id and not args.question:
        raise SystemExit("Please pass --id or --question.")

    engine = AnswerEngine(use_llm=False)
    question = args.question
    if args.id:
        rows = {item.id: item.question for item in read_questions()}
        if args.id not in rows:
            raise SystemExit(f"Question id not found: {args.id}")
        question = rows[args.id]

    plan = build_query_plan(question, engine.manual_names)
    print(f"Question: {plan.normalized}")
    print(f"Language: {plan.language}")
    print(f"Target manuals: {', '.join(sorted(plan.target_manuals)) or '-'}")
    print("Variants:")
    for variant in plan.variants:
        print(f"  - {variant}")

    print("\nEvidence:")
    for idx, result in enumerate(engine.retrieve(plan)[: args.top_k], start=1):
        chunk = result.chunk
        images = ", ".join(chunk.image_ids) if chunk.image_ids else "-"
        preview = readable_chunk_text(chunk).replace("\n", " ")[:420]
        print(f"{idx}. score={result.score:.2f} manual={chunk.manual} title={chunk.title}")
        print(f"   images={images}")
        print(f"   {preview}")

    print("\nAnswer:")
    print(engine.answer(question, qid=args.id or None))


if __name__ == "__main__":
    main()

