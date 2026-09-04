from __future__ import annotations

import re


_MANUAL_DISPLAY_ERROR_CODE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:ERR[\s_-]*(?:\d{1,3}|CF)|NO[\s_-]*CF|FULL[\s_-]*CF)"
    r"(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)
_STANDARD_ERROR_CODE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Za-z]{1,3}[_-]?\d{2,3}"
    r"(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)


def normalize_error_code(value: str) -> str | None:
    normalized = value.strip().upper()
    manual = re.fullmatch(
        r"(?:ERR[\s_-]*(?:\d{1,3}|CF)|NO[\s_-]*CF|FULL[\s_-]*CF)",
        normalized,
        flags=re.IGNORECASE,
    )
    if manual:
        compact = re.sub(r"[\s_-]+", "", normalized)
        if compact.startswith("ERR"):
            suffix = compact[3:]
            if suffix.isdigit() and len(suffix) < 2:
                suffix = suffix.zfill(2)
            return f"ERR {suffix}"
        if compact.startswith("NO"):
            return "NO CF"
        return "FULL CF"
    if re.fullmatch(r"[A-Z]{1,3}[_-]?\d{2,3}", normalized):
        return re.sub(r"[_-]+", "", normalized)
    return None


def extract_manual_display_error_codes(text: str) -> list[str]:
    """Extract spaced camera/manual display codes in a stable form."""

    values: list[str] = []
    for match in _MANUAL_DISPLAY_ERROR_CODE.finditer(text):
        normalized = normalize_error_code(match.group(0))
        if normalized is None:
            continue
        if normalized not in values:
            values.append(normalized)
    return values


def extract_normalized_error_codes(text: str) -> list[str]:
    matches = [
        *(
            (match.start(), match.group(0))
            for match in _MANUAL_DISPLAY_ERROR_CODE.finditer(text)
        ),
        *(
            (match.start(), match.group(0))
            for match in _STANDARD_ERROR_CODE.finditer(text)
        ),
    ]
    values: list[str] = []
    for _, raw in sorted(matches):
        normalized = normalize_error_code(raw)
        if normalized is not None and normalized not in values:
            values.append(normalized)
    return values


def error_code_search_forms(text: str) -> list[str]:
    """Return separator variants for codes that manuals commonly spell differently."""

    forms: list[str] = []
    for normalized in extract_normalized_error_codes(text):
        parts = normalized.split()
        if len(parts) == 1:
            continue
        variants = [normalized, "".join(parts), "_".join(parts)]
        for variant in variants:
            if variant not in forms:
                forms.append(variant)
    return forms


def has_manual_display_error_code(text: str) -> bool:
    return _MANUAL_DISPLAY_ERROR_CODE.search(text) is not None
