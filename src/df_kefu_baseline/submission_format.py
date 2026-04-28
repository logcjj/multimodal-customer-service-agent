from __future__ import annotations

import re
from collections import Counter


# Official examples encode each answer as a JSON string inside the ret field:
# "answer1","answer2" or "answer<PIC>",["image_id"].
RET_ITEM_SEPARATOR = ","
RET_IMAGE_SEPARATOR = ","

DISALLOWED_SUBMISSION_CHAR_RE = re.compile(
    r"[^\x20-\x7E\n\r\t\u4e00-\u9fa5\u3000-\u303F\uFF00-\uFFEF]+"
)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")

COMMON_REPLACEMENTS = {
    "\u00a0": " ",
    "\u1680": " ",
    "\u2000": " ",
    "\u2001": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2004": " ",
    "\u2005": " ",
    "\u2006": " ",
    "\u2007": " ",
    "\u2008": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u2028": "\n",
    "\u2029": "\n",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "−": "-",
    "…": "...",
    "·": "-",
    "•": "-",
    "●": "-",
    "○": "o",
    "→": "->",
    "←": "<-",
    "×": "x",
    "≤": "<=",
    "≥": ">=",
    "℃": "C",
    "℉": "F",
}


def sanitize_submission_text(text: str) -> str:
    clean = str(text or "")
    for old, new in COMMON_REPLACEMENTS.items():
        clean = clean.replace(old, new)
    clean = ZERO_WIDTH_RE.sub("", clean)
    clean = CONTROL_CHAR_RE.sub("", clean)
    clean = DISALLOWED_SUBMISSION_CHAR_RE.sub("", clean)
    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def find_disallowed_submission_chars(text: str) -> list[dict[str, str]]:
    counter: Counter[str] = Counter()
    for match in DISALLOWED_SUBMISSION_CHAR_RE.finditer(str(text or "")):
        counter.update(match.group(0))
    return [
        {"char": char, "codepoint": f"U+{ord(char):04X}", "count": str(count)}
        for char, count in counter.most_common()
    ]
