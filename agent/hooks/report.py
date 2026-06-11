from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from agent.hooks.recorder import get_hook_log_path


def load_events(date_value: str) -> list[dict]:
    path = Path(get_hook_log_path(date_value))
    if not path.exists():
        return []

    events: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def summarize_events(events: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        grouped[event.get("tool_name", "unknown")].append(event)

    rows: list[dict] = []
    for tool_name, items in sorted(grouped.items()):
        after_events = [item for item in items if item.get("stage") == "after_tool_call"]
        error_events = [item for item in items if item.get("stage") == "on_tool_error"]
        blocked_events = [item for item in items if item.get("status") == "blocked"]
        latencies = [
            float(item["latency_ms"])
            for item in after_events + error_events
            if item.get("latency_ms") is not None
        ]
        rows.append(
            {
                "tool_name": tool_name,
                "calls": len(after_events) + len(error_events),
                "success": len(after_events),
                "errors": len(error_events),
                "blocked": len(blocked_events),
                "avg_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tool hook JSONL logs.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"), help="日志日期，格式 YYYYMMDD")
    args = parser.parse_args()

    events = load_events(args.date)
    rows = summarize_events(events)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
