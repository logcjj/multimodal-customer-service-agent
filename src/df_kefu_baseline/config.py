from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANUAL_DIR = PROJECT_ROOT / "手册"
ILLUSTRATION_DIR = MANUAL_DIR / "插图"
QUESTION_PATH = PROJECT_ROOT / "question_public.csv"
SUBMISSION_EXAMPLE_PATH = PROJECT_ROOT / "submission_example.csv"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

