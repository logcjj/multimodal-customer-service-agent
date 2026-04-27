from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .config import QUESTION_PATH


@dataclass(frozen=True)
class Question:
    id: str
    question: str


def normalize_question(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = []
    for line in text.split("\n"):
        clean = line.strip().strip('"').strip()
        if clean:
            lines.append(clean)
    return "\n".join(lines)


def read_questions(path: Path = QUESTION_PATH) -> list[Question]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        return [
            Question(id=str(row["id"]).strip(), question=normalize_question(row["question"]))
            for row in rows
        ]


def write_submission(rows: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        for qid, ret in rows:
            writer.writerow({"id": qid, "ret": ret})

