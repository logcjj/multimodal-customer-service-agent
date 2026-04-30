from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

from df_kefu_baseline.answer import AnswerEngine, readable_chunk_text
from df_kefu_baseline.config import PROJECT_ROOT, QUESTION_PATH
from df_kefu_baseline.llm_client import OpenAICompatibleClient
from df_kefu_baseline.policy import answer_policy_question, looks_like_manual_question
from df_kefu_baseline.query_planner import build_query_plan

from audit_submission import parse_ret


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def read_questions(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["id"]: row["question"] for row in csv.DictReader(f)}


def read_submission(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["id"]: row["ret"] for row in csv.DictReader(f)}


def policy_evidence_for_question(question: str) -> str:
    policy_answer = answer_policy_question(question)
    if not policy_answer and not looks_like_manual_question(question):
        policy_answer = (
            "通用售后政策题应优先按平台客服规则处理：围绕订单号、凭证、退换货、退款、发票、物流、投诉、维修、"
            "质保和人工客服升级给出可执行答复；不要强行召回商品说明书。"
        )
    if not policy_answer:
        return ""
    return "[policy] source=local_customer_service_policy\n" + policy_answer


def evidence_for_question(engine: AnswerEngine, question: str, limit: int = 5) -> str:
    plan = build_query_plan(question, engine.manual_names)
    results = engine.retrieve(plan)
    blocks: list[str] = []
    policy_block = policy_evidence_for_question(question)
    if policy_block:
        blocks.append(policy_block)
    for idx, result in enumerate(results[:limit], start=1):
        chunk = result.chunk
        text = readable_chunk_text(chunk)
        images = ", ".join(chunk.image_ids) if chunk.image_ids else "none"
        blocks.append(
            f"[{idx}] manual={chunk.manual}\n"
            f"title={chunk.title}\n"
            f"score={result.score:.2f}\n"
            f"images={images}\n"
            f"text={text[:1200]}"
        )
    return "\n\n".join(blocks)


def build_judge_messages(question: str, answer: str, images: list[str], evidence: str) -> list[dict[str, str]]:
    system = (
        "你是 DataFountain 多模态客服问答比赛的离线评测 judge。"
        "请只根据题目、候选答案和给定证据打分，不要引入外部知识。"
        "评分要严格，重点惩罚答非所问、产品召回错误、手册依据不足、遗漏关键步骤、图片占位与图片数组不一致、语言不匹配和编造内容。"
        "输出必须是一个 JSON 对象，不要 Markdown。"
    )
    user = {
        "question": question,
        "candidate_answer": answer,
        "candidate_images": images,
        "retrieved_evidence": evidence,
        "rubric": {
            "relevance": "0-25，是否直接回答题目且命中正确产品/场景",
            "groundedness": "0-25，是否能由证据支持，是否避免编造",
            "completeness": "0-20，关键步骤、注意事项、条件是否完整",
            "clarity": "0-15，表达是否清晰、客服语气是否自然、长度是否合适",
            "format": "0-15，语言、<PIC>、图片引用和提交格式是否合理",
        },
        "required_json_schema": {
            "score": "0-100 number",
            "relevance": "0-25 number",
            "groundedness": "0-25 number",
            "completeness": "0-20 number",
            "clarity": "0-15 number",
            "format": "0-15 number",
            "risk": "ok|review|bad",
            "issues": ["short issue labels"],
            "fix_suggestion": "one concise suggestion in Chinese",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def parse_judge_json(raw: str) -> dict[str, Any]:
    match = JSON_OBJECT_RE.search(raw)
    if not match:
        raise ValueError(f"No JSON object found: {raw[:200]}")
    data = json.loads(match.group(0))
    score = float(data.get("score", 0))
    data["score"] = max(0.0, min(100.0, score))
    return data


def judge_row(
    client: OpenAICompatibleClient,
    engine: AnswerEngine,
    qid: str,
    question: str,
    ret: str,
    evidence_limit: int = 5,
    retries: int = 2,
) -> dict[str, Any]:
    answer, images, parse_error = parse_ret(ret)
    evidence = evidence_for_question(engine, question, limit=evidence_limit)
    messages = build_judge_messages(question, answer, images, evidence)
    last_error = ""
    for attempt in range(retries + 1):
        try:
            raw = client.chat(messages, temperature=0.0)
            judged = parse_judge_json(raw)
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt >= retries:
                judged = {
                    "score": 0,
                    "relevance": 0,
                    "groundedness": 0,
                    "completeness": 0,
                    "clarity": 0,
                    "format": 0,
                    "risk": "bad",
                    "issues": [f"judge_error: {last_error[:120]}"],
                    "fix_suggestion": "重新运行 judge 或检查 LLM API 配置。",
                }
            else:
                time.sleep(1.5 * (attempt + 1))
    if parse_error:
        judged["score"] = min(float(judged.get("score", 0)), 45.0)
        judged["risk"] = "bad"
        judged["issues"] = [*judged.get("issues", []), parse_error]
    return {
        "id": qid,
        "score": f"{float(judged.get('score', 0)):.2f}",
        "risk": judged.get("risk", "review"),
        "relevance": judged.get("relevance", ""),
        "groundedness": judged.get("groundedness", ""),
        "completeness": judged.get("completeness", ""),
        "clarity": judged.get("clarity", ""),
        "format": judged.get("format", ""),
        "issues": "; ".join(str(item) for item in judged.get("issues", [])),
        "fix_suggestion": str(judged.get("fix_suggestion", "")),
        "question": question.replace("\n", " / "),
        "answer_start": answer[:400].replace("\n", " / "),
    }


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use an OpenAI-compatible LLM as an offline judge for a submission CSV.")
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "llm_judge.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--evidence-limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission_path = resolve_path(args.submission)
    output_path = resolve_path(args.output)
    questions = read_questions(QUESTION_PATH)
    submissions = read_submission(submission_path)
    items = [(qid, question) for qid, question in questions.items() if qid in submissions]
    if args.start_id:
        items = [item for item in items if int(item[0]) >= args.start_id]
    if args.end_id:
        items = [item for item in items if int(item[0]) <= args.end_id]
    if args.limit:
        items = items[: args.limit]

    client = OpenAICompatibleClient()
    engine = AnswerEngine(use_llm=False)
    rows: list[dict[str, Any]] = []
    for idx, (qid, question) in enumerate(items, start=1):
        row = judge_row(client, engine, qid, question, submissions[qid], evidence_limit=args.evidence_limit)
        rows.append(row)
        print(f"[{idx}/{len(items)}] id={qid} score={row['score']} risk={row['risk']}")
        if args.delay > 0:
            time.sleep(args.delay)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "score",
        "risk",
        "relevance",
        "groundedness",
        "completeness",
        "clarity",
        "format",
        "issues",
        "fix_suggestion",
        "question",
        "answer_start",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    scores = [float(row["score"]) for row in rows]
    risk_counts = {risk: sum(row["risk"] == risk for row in rows) for risk in ("ok", "review", "bad")}
    print(f"saved={output_path}")
    if scores:
        print(f"avg={mean(scores):.2f} min={min(scores):.2f} max={max(scores):.2f} risks={risk_counts}")


if __name__ == "__main__":
    main()
