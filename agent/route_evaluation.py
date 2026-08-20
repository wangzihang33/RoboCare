from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

from agent.routing import HybridRouter, RouteDecision, RouteName, RouteStatus
from utils.path_tool import get_abs_path


DEFAULT_DATASET = "data/agent_route_eval_dataset.csv"
DEFAULT_OUTPUT_DIR = "outputs/evaluations"


def build_evaluation_router(*, with_llm: bool) -> HybridRouter:
    if not with_llm:
        return HybridRouter()

    from agent.route_classifier import SmallLLMRouteClassifier
    from model.factory import build_chat_model
    from utils.config_handler import agent_conf

    model_name = str(agent_conf.get("router_model_name") or "").strip()
    if not model_name:
        raise ValueError("启用联合小模型评测时必须配置 ROUTER_MODEL_NAME")
    return HybridRouter(
        llm_classifier=SmallLLMRouteClassifier(
            build_chat_model(
                model_name,
                provider=str(agent_conf.get("router_provider", "deepseek")),
                api_key_env=str(
                    agent_conf.get("router_api_key_env", "DEEPSEEK_API_KEY")
                ),
                base_url=str(agent_conf.get("router_base_url") or "") or None,
            )
        )
    )


@dataclass(frozen=True)
class RouteEvalExample:
    query: str
    expected_status: RouteStatus
    expected_route: RouteName | None
    expected_tool: str
    expected_missing_slots: tuple[str, ...]


@dataclass(frozen=True)
class RouteEvalResult:
    query: str
    expected_status: RouteStatus
    predicted_status: RouteStatus
    expected_route: RouteName | None
    predicted_route: RouteName | None
    expected_tool: str
    predicted_tool: str
    expected_missing_slots: tuple[str, ...]
    predicted_missing_slots: tuple[str, ...]
    source: str
    reason_code: str
    status_correct: bool
    route_correct: bool
    tool_correct: bool
    missing_slots_correct: bool
    clarification_correct: bool
    latency_ms: float


def load_route_dataset(dataset_path: Path) -> list[RouteEvalExample]:
    examples: list[RouteEvalExample] = []
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            query = (row.get("query") or "").strip()
            if not query:
                continue
            raw_route = (row.get("expected_route") or "").strip()
            examples.append(
                RouteEvalExample(
                    query=query,
                    expected_status=RouteStatus(
                        (row.get("expected_status") or "").strip()
                    ),
                    expected_route=RouteName(raw_route) if raw_route else None,
                    expected_tool=(row.get("expected_tool") or "").strip(),
                    expected_missing_slots=_parse_slots(
                        row.get("expected_missing_slots") or ""
                    ),
                )
            )
    return examples


def load_route_datasets(dataset_paths: list[Path]) -> list[RouteEvalExample]:
    examples: list[RouteEvalExample] = []
    for dataset_path in dataset_paths:
        examples.extend(load_route_dataset(dataset_path))
    return examples


def evaluate_router(
    router: HybridRouter,
    examples: list[RouteEvalExample],
) -> list[RouteEvalResult]:
    results: list[RouteEvalResult] = []
    for example in examples:
        started_at = time.perf_counter()
        decision = router.route(example.query)
        latency_ms = (time.perf_counter() - started_at) * 1000
        predicted_tool = decision.tool_candidates[0] if decision.tool_candidates else ""
        results.append(_build_result(example, decision, predicted_tool, latency_ms))
    return results


def _build_result(
    example: RouteEvalExample,
    decision: RouteDecision,
    predicted_tool: str,
    latency_ms: float,
) -> RouteEvalResult:
    expected_clarification = example.expected_status is not RouteStatus.DECISIVE
    return RouteEvalResult(
        query=example.query,
        expected_status=example.expected_status,
        predicted_status=decision.status,
        expected_route=example.expected_route,
        predicted_route=decision.route,
        expected_tool=example.expected_tool,
        predicted_tool=predicted_tool,
        expected_missing_slots=example.expected_missing_slots,
        predicted_missing_slots=decision.missing_slots,
        source=decision.source,
        reason_code=decision.reason_code,
        status_correct=decision.status is example.expected_status,
        route_correct=decision.route is example.expected_route,
        tool_correct=predicted_tool == example.expected_tool,
        missing_slots_correct=(
            set(decision.missing_slots) == set(example.expected_missing_slots)
        ),
        clarification_correct=(
            decision.requires_clarification == expected_clarification
        ),
        latency_ms=latency_ms,
    )


def aggregate_results(results: list[RouteEvalResult]) -> dict[str, float]:
    metric_names = (
        "route_count",
        "status_accuracy",
        "route_accuracy",
        "tool_selection_accuracy",
        "missing_slot_accuracy",
        "clarification_accuracy",
        "macro_f1",
        "invalid_tool_rate",
        "validation_fallback_rate",
        "avg_latency_ms",
        "p95_latency_ms",
    )
    if not results:
        return {name: 0.0 for name in metric_names}

    tool_results = [result for result in results if result.expected_tool]
    slot_results = [
        result
        for result in results
        if result.expected_status is RouteStatus.NEEDS_CLARIFICATION
    ]
    invalid_tool_count = sum(_has_invalid_predicted_tool(result) for result in results)
    validation_fallback_count = sum(
        result.source == "rule_fallback" for result in results
    )
    latencies = sorted(result.latency_ms for result in results)
    p95_index = min(len(latencies) - 1, max(0, math.ceil(len(latencies) * 0.95) - 1))
    return {
        "route_count": float(len(results)),
        "status_accuracy": mean(result.status_correct for result in results),
        "route_accuracy": mean(result.route_correct for result in results),
        "tool_selection_accuracy": (
            mean(result.tool_correct for result in tool_results) if tool_results else 0.0
        ),
        "missing_slot_accuracy": (
            mean(result.missing_slots_correct for result in slot_results)
            if slot_results
            else 0.0
        ),
        "clarification_accuracy": mean(
            result.clarification_correct for result in results
        ),
        "macro_f1": _macro_f1(results),
        "invalid_tool_rate": invalid_tool_count / len(results),
        "validation_fallback_rate": validation_fallback_count / len(results),
        "avg_latency_ms": mean(result.latency_ms for result in results),
        "p95_latency_ms": latencies[p95_index],
    }


def _macro_f1(results: list[RouteEvalResult]) -> float:
    scores: list[float] = []
    for route in RouteName:
        true_positive = sum(
            result.expected_route is route and result.predicted_route is route
            for result in results
        )
        false_positive = sum(
            result.expected_route is not route and result.predicted_route is route
            for result in results
        )
        false_negative = sum(
            result.expected_route is route and result.predicted_route is not route
            for result in results
        )
        if true_positive + false_positive == 0 and true_positive + false_negative == 0:
            continue
        precision = true_positive / (true_positive + false_positive) if true_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return mean(scores) if scores else 0.0


def _has_invalid_predicted_tool(result: RouteEvalResult) -> bool:
    if not result.predicted_tool:
        return False
    allowed_tools = {
        RouteName.DIRECT: set(),
        RouteName.LOCAL_RAG: {"rag_summarize"},
        RouteName.WEB_RAG: {"web_search"},
        RouteName.BUSINESS_QUERY: {"get_weather", "fetch_external_data"},
        RouteName.TROUBLESHOOTING: {"rag_summarize"},
    }
    if result.predicted_route is None:
        return True
    return result.predicted_tool not in allowed_tools[result.predicted_route]


def save_route_report(
    *,
    results: list[RouteEvalResult],
    summary: dict[str, float],
    output_path: Path,
    dataset_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Agent 路由评测报告",
        "",
        f"- dataset: `{dataset_path}`",
        f"- total_cases: `{len(results)}`",
        "",
        "## 指标汇总",
        "",
        f"- status_accuracy: `{summary['status_accuracy']:.4f}`",
        f"- route_accuracy: `{summary['route_accuracy']:.4f}`",
        f"- macro_f1: `{summary['macro_f1']:.4f}`",
        f"- tool_selection_accuracy: `{summary['tool_selection_accuracy']:.4f}`",
        f"- missing_slot_accuracy: `{summary['missing_slot_accuracy']:.4f}`",
        f"- clarification_accuracy: `{summary['clarification_accuracy']:.4f}`",
        f"- invalid_tool_rate: `{summary['invalid_tool_rate']:.4f}`",
        f"- validation_fallback_rate: `{summary['validation_fallback_rate']:.4f}`",
        f"- avg_latency_ms: `{summary['avg_latency_ms']:.4f}`",
        f"- p95_latency_ms: `{summary['p95_latency_ms']:.4f}`",
        "",
        "## 单题结果",
        "",
        "| query | expected_status | predicted_status | expected_route | predicted_route | expected_tool | predicted_tool | expected_missing_slots | predicted_missing_slots | latency_ms | source | reason_code |",
        "|---|---|---|---|---|---|---|---|---|---:|---|---|",
    ]
    for result in results:
        query = result.query.replace("|", "/").replace("\n", " ")
        expected_route = result.expected_route.value if result.expected_route else ""
        predicted_route = result.predicted_route.value if result.predicted_route else ""
        lines.append(
            f"| {query} | {result.expected_status.value} | {result.predicted_status.value} | "
            f"{expected_route} | {predicted_route} | {result.expected_tool} | "
            f"{result.predicted_tool} | {';'.join(result.expected_missing_slots)} | "
            f"{';'.join(result.predicted_missing_slots)} | {result.latency_ms:.4f} | "
            f"{result.source} | {result.reason_code} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate customer-service route decisions.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="可重复传入多个 CSV，结果按联合数据集统计",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="启用已配置的小模型，仅评测 CONFLICT/NO_MATCH 样本",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = [Path(dataset) for dataset in (args.dataset or [DEFAULT_DATASET])]
    dataset_paths = [
        path if path.is_absolute() else Path(get_abs_path(str(path)))
        for path in dataset_paths
    ]
    examples = load_route_datasets(dataset_paths)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError(f"路由评测集为空: {dataset_paths}")

    results = evaluate_router(
        build_evaluation_router(with_llm=args.with_llm),
        examples,
    )
    summary = aggregate_results(results)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(get_abs_path(args.output_dir))
    report_path = output_dir / f"agent_route_eval_{run_id}.md"
    save_route_report(
        results=results,
        summary=summary,
        output_path=report_path,
        dataset_path="; ".join(str(path) for path in dataset_paths),
    )
    print(f"Agent 路由评测完成: {len(results)} cases")
    print(f"Markdown: {report_path}")
    for key in (
        "status_accuracy",
        "route_accuracy",
        "macro_f1",
        "tool_selection_accuracy",
        "missing_slot_accuracy",
        "clarification_accuracy",
        "invalid_tool_rate",
        "validation_fallback_rate",
        "p95_latency_ms",
    ):
        print(f"- {key}: {summary[key]:.4f}")


def _parse_slots(raw_value: str) -> tuple[str, ...]:
    return tuple(slot.strip() for slot in raw_value.split(";") if slot.strip())


if __name__ == "__main__":
    main()
