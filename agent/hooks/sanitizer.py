from __future__ import annotations

import json
import re
from typing import Any


MAX_SUMMARY_CHARS = 320

SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("id_card", re.compile(r"\b\d{17}[\dXx]\b")),
    ("phone", re.compile(r"\b1[3-9]\d{9}\b")),
    ("bank_card", re.compile(r"\b\d{16,19}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("api_key", re.compile(r"(?i)\b(?:sk|ak|api[_-]?key|token|secret)[-_A-Za-z0-9]{8,}\b")),
]

SENSITIVE_KEYWORDS = (
    "身份证",
    "身份证号",
    "手机号",
    "电话",
    "银行卡",
    "住址",
    "家庭住址",
    "密码",
    "密钥",
    "token",
    "api key",
    "secret",
)


def to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def detect_sensitive_types(value: Any) -> list[str]:
    text = to_text(value)
    detected: list[str] = []
    lowered = text.lower()

    for name, pattern in SENSITIVE_PATTERNS:
        if pattern.search(text):
            detected.append(name)

    for keyword in SENSITIVE_KEYWORDS:
        if keyword.lower() in lowered and keyword not in detected:
            detected.append(keyword)

    return detected


def redact_text(text: str) -> str:
    redacted = text
    for name, pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)

    for keyword in SENSITIVE_KEYWORDS:
        redacted = re.sub(re.escape(keyword), "[REDACTED_FIELD]", redacted, flags=re.I)

    return redacted


def summarize_value(value: Any, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    text = redact_text(to_text(value))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
