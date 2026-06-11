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
        trace_events = [item for item in items if item.get("stage") == "websearch_trace"]
        blocked_events = [item for item in items if item.get("status") == "blocked"]
        latencies = [
            float(item["latency_ms"])
            for item in after_events + error_events
            if item.get("latency_ms") is not None
        ]
        row = {
            "tool_name": tool_name,
            "calls": len(after_events) + len(error_events),
            "success": len(after_events),
            "errors": len(error_events),
            "blocked": len(blocked_events),
            "avg_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
        }
        if trace_events:
            row.update(summarize_websearch_trace(trace_events))
        rows.append(row)
    return rows


def summarize_websearch_trace(trace_events: list[dict]) -> dict:
    cache_hits = [event for event in trace_events if event.get("cache_hit")]
    source_counts = [
        int(event["source_count"])
        for event in trace_events
        if event.get("source_count") is not None
    ]
    urls_after_filter = [
        int(event["urls_after_filter"])
        for event in trace_events
        if event.get("urls_after_filter") is not None
    ]
    fetch_success_counts = []
    fetch_error_counts = []
    for event in trace_events:
        fetch_stats = event.get("fetch_stats") or {}
        if fetch_stats.get("success_count") is not None:
            fetch_success_counts.append(int(fetch_stats["success_count"]))
        if fetch_stats.get("error_count") is not None:
            fetch_error_counts.append(int(fetch_stats["error_count"]))

    return {
        "websearch_trace_events": len(trace_events),
        "cache_hits": len(cache_hits),
        "cache_hit_rate": round(len(cache_hits) / len(trace_events), 4) if trace_events else 0.0,
        "avg_source_count": round(mean(source_counts), 3) if source_counts else 0.0,
        "avg_urls_after_filter": round(mean(urls_after_filter), 3) if urls_after_filter else 0.0,
        "avg_fetch_success": round(mean(fetch_success_counts), 3) if fetch_success_counts else 0.0,
        "avg_fetch_errors": round(mean(fetch_error_counts), 3) if fetch_error_counts else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tool hook JSONL logs.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"), help="日志日期，格式 YYYYMMDD")
    args = parser.parse_args()

    events = load_events(args.date)
    rows = summarize_events(events)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
