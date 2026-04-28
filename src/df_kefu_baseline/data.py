from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .config import QUESTION_PATH
from .submission_format import sanitize_submission_text


@dataclass(frozen=True)
class Question:
    id: str
    question: str


def normalize_question(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = []
    for line in text.split("\n"):
        clean = line.strip()
        clean = re.sub(r'^[\s"“”]+', "", clean)
        clean = re.sub(r'[\s"“”]+[,，]?\s*$', "", clean)
        if clean:
            lines.append(clean)
    return "\n".join(lines)


def split_question_parts(question: str) -> list[str]:
    parts = [part.strip() for part in normalize_question(question).splitlines() if part.strip()]
    return parts or [normalize_question(question)]


def read_questions(path: Path = QUESTION_PATH) -> list[Question]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        return [
            Question(id=str(row["id"]).strip(), question=normalize_question(row["question"]))
            for row in rows
        ]


def write_submission(rows: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"], lineterminator="\n")
        writer.writeheader()
        for qid, ret in rows:
            writer.writerow({"id": qid, "ret": sanitize_submission_text(ret)})
