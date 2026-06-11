from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from utils.path_tool import get_abs_path


HOOK_LOG_ROOT = get_abs_path("logs/hooks")


def get_hook_log_path(date_value: str | None = None) -> str:
    os.makedirs(HOOK_LOG_ROOT, exist_ok=True)
    date_text = date_value or datetime.now().strftime("%Y%m%d")
    return os.path.join(HOOK_LOG_ROOT, f"tool_calls_{date_text}.jsonl")


def write_hook_event(event: dict[str, Any]) -> None:
    event = {
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        **event,
    }
    with open(get_hook_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
