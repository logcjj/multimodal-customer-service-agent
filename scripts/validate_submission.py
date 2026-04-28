from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from audit_submission import parse_ret

from df_kefu_baseline.config import PROJECT_ROOT, QUESTION_PATH
from df_kefu_baseline.data import read_questions
from df_kefu_baseline.submission_format import find_disallowed_submission_chars


PIC_RE = re.compile(r"<PIC>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate submission CSV format before DataFountain upload.")
    parser.add_argument("submission", type=Path)
    return parser.parse_args()


def count_top_level_strings(ret: str) -> int:
    decoder = __import__("json").JSONDecoder()
    idx = 0
    count = 0
    while idx < len(ret):
        while idx < len(ret) and ret[idx].isspace():
            idx += 1
        if idx >= len(ret):
            break
        value, idx = decoder.raw_decode(ret, idx)
        if isinstance(value, str):
            count += 1
        while idx < len(ret) and ret[idx].isspace():
            idx += 1
        if idx < len(ret) and ret[idx] == ",":
            idx += 1
    return count


def validate_submission(path: Path) -> list[str]:
    issues: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "ret"]:
            issues.append(f"列名应为 ['id', 'ret']，实际为 {reader.fieldnames}")
        rows = list(reader)

    expected_ids = [item.id for item in read_questions(QUESTION_PATH)]
    expected_set = set(expected_ids)
    seen: set[str] = set()
    duplicates: set[str] = set()
    row_by_id: dict[str, str] = {}
    for row in rows:
        qid = str(row.get("id", "")).strip()
        if qid in seen:
            duplicates.add(qid)
        seen.add(qid)
        row_by_id[qid] = row.get("ret", "")

    missing = [qid for qid in expected_ids if qid not in seen]
    extra = sorted(seen - expected_set, key=lambda item: int(item) if item.isdigit() else item)
    if missing:
        issues.append(f"缺少id: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if extra:
        issues.append(f"多余id: {extra[:20]}{'...' if len(extra) > 20 else ''}")
    if duplicates:
        issues.append(f"重复id: {sorted(duplicates)}")

    for qid in expected_ids:
        ret = row_by_id.get(qid, "")
        if not ret:
            issues.append(f"id={qid}: ret为空")
            continue
        answer, images, parse_error = parse_ret(ret)
        if parse_error:
            issues.append(f"id={qid}: {parse_error}")
        bad_chars = find_disallowed_submission_chars(ret)
        if bad_chars:
            detail = ", ".join(
                f"{item['char']}({item['codepoint']})x{item['count']}" for item in bad_chars[:5]
            )
            issues.append(f"id={qid}: ret包含异常字符 {detail}")
        if len(PIC_RE.findall(answer)) != len(images):
            issues.append(f"id={qid}: <PIC>数量({len(PIC_RE.findall(answer))}) != 图片数量({len(images)})")

    return issues


def main() -> None:
    args = parse_args()
    path = args.submission
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    issues = validate_submission(path)
    if issues:
        print(f"FAILED: {len(issues)} issue(s)")
        for issue in issues[:80]:
            print("-", issue)
        sys.exit(1)
    print(f"OK: {path}")


if __name__ == "__main__":
    main()
