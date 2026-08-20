from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from agent.route_evaluation import (
    DEFAULT_OUTPUT_DIR,
    RouteEvalExample,
    load_route_datasets,
)
from utils.path_tool import get_abs_path


_VALID_TOOLS = {
    "rag_summarize",
    "web_search",
    "get_weather",
    "fetch_external_data",
    "fill_context_for_report",
}


@dataclass(frozen=True)
class BaselineToolResult:
    query: str
    expected_tool: str
    predicted_tool: str
    tool_correct: bool
    invalid_tool: bool
    unnecessary_tool: bool
    latency_ms: float


def extract_predicted_tool(response: Any) -> str:
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls is None:
        additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
        tool_calls = additional_kwargs.get("tool_calls", [])
    if not tool_calls:
        return ""
    first_call = tool_calls[0]
    if isinstance(first_call, dict):
        name = first_call.get("name")
        if name:
            return str(name)
        function = first_call.get("function") or {}
        return str(function.get("name") or "")
    return str(getattr(first_call, "name", "") or "")


def evaluate_unrouted_agent(
    model: Any,
    examples: list[RouteEvalExample],
    *,
    system_prompt: str,
) -> list[BaselineToolResult]:
    """Evaluate first tool selection with every legacy tool exposed."""
    model_with_tools = model.bind_tools(_load_all_tools(), tool_choice="auto")
    results: list[BaselineToolResult] = []
    for example in examples:
        started_at = time.perf_counter()
        response = model_with_tools.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example.query},
            ]
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        predicted_tool = extract_predicted_tool(response)
        results.append(
            BaselineToolResult(
                query=example.query,
                expected_tool=example.expected_tool,
                predicted_tool=predicted_tool,
                tool_correct=predicted_tool == example.expected_tool,
                invalid_tool=bool(predicted_tool) and predicted_tool not in _VALID_TOOLS,
                unnecessary_tool=not example.expected_tool and bool(predicted_tool),
                latency_ms=latency_ms,
            )
        )
    return results


def aggregate_baseline_results(
    results: list[BaselineToolResult],
) -> dict[str, float]:
    if not results:
        return {
            "case_count": 0.0,
            "tool_selection_accuracy": 0.0,
            "invalid_tool_rate": 0.0,
            "unnecessary_tool_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    tool_cases = [result for result in results if result.expected_tool]
    latencies = sorted(result.latency_ms for result in results)
    p95_index = min(len(latencies) - 1, max(0, math.ceil(len(latencies) * 0.95) - 1))
    return {
        "case_count": float(len(results)),
        "tool_selection_accuracy": (
            mean(result.tool_correct for result in tool_cases) if tool_cases else 0.0
        ),
        "invalid_tool_rate": mean(result.invalid_tool for result in results),
        "unnecessary_tool_rate": mean(
            result.unnecessary_tool for result in results
        ),
        "avg_latency_ms": mean(result.latency_ms for result in results),
        "p95_latency_ms": latencies[p95_index],
    }


def save_baseline_report(
    *,
    results: list[BaselineToolResult],
    summary: dict[str, float],
    output_path: Path,
    dataset_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 无路由全工具 Agent Baseline 报告",
        "",
        f"- dataset: `{dataset_label}`",
        f"- total_cases: `{len(results)}`",
        "",
        "## 指标汇总",
        "",
        f"- tool_selection_accuracy: `{summary['tool_selection_accuracy']:.4f}`",
        f"- invalid_tool_rate: `{summary['invalid_tool_rate']:.4f}`",
        f"- unnecessary_tool_rate: `{summary['unnecessary_tool_rate']:.4f}`",
        f"- avg_latency_ms: `{summary['avg_latency_ms']:.4f}`",
        f"- p95_latency_ms: `{summary['p95_latency_ms']:.4f}`",
        "",
        "## 单题结果",
        "",
        "| query | expected_tool | predicted_tool | tool_correct | unnecessary_tool | latency_ms |",
        "|---|---|---|---:|---:|---:|",
    ]
    for result in results:
        query = result.query.replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {query} | {result.expected_tool} | {result.predicted_tool} | "
            f"{result.tool_correct} | {result.unnecessary_tool} | {result.latency_ms:.4f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the legacy full-tool Agent baseline.")
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-examples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_paths = args.dataset or ["data/agent_route_eval_dataset.csv"]
    dataset_paths = [Path(path) for path in raw_paths]
    dataset_paths = [
        path if path.is_absolute() else Path(get_abs_path(str(path)))
        for path in dataset_paths
    ]
    examples = load_route_datasets(dataset_paths)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError(f"baseline 评测集为空: {dataset_paths}")

    from model.factory import chat_model
    from utils.prompt_loader import load_system_prompts

    results = evaluate_unrouted_agent(
        chat_model,
        examples,
        system_prompt=load_system_prompts(),
    )
    summary = aggregate_baseline_results(results)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(get_abs_path(args.output_dir))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"baseline_tool_eval_{run_id}.md"
    save_baseline_report(
        results=results,
        summary=summary,
        output_path=report_path,
        dataset_label="; ".join(str(path) for path in dataset_paths),
    )
    print(f"无路由全工具 baseline 评测完成: {len(results)} cases")
    print(f"Markdown: {report_path}")
    for key in (
        "tool_selection_accuracy",
        "invalid_tool_rate",
        "unnecessary_tool_rate",
        "p95_latency_ms",
    ):
        print(f"- {key}: {summary[key]:.4f}")


def _load_all_tools() -> list[Any]:
    from agent.tools.agent_tools import (
        fetch_external_data,
        fill_context_for_report,
        get_weather,
        rag_summarize,
        web_search,
    )

    return [
        rag_summarize,
        get_weather,
        fetch_external_data,
        fill_context_for_report,
        web_search,
    ]


if __name__ == "__main__":
    main()
