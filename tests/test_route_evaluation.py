from pathlib import Path
import sys

from agent.route_evaluation import (
    RouteEvalExample,
    aggregate_results,
    build_evaluation_router,
    evaluate_router,
    load_route_dataset,
    load_route_datasets,
    parse_args,
    save_route_report,
)
from agent.routing import HybridRouter, RouteDecision, RouteName, RouteStatus


def test_evaluation_reports_status_route_tool_and_slot_accuracy():
    examples = [
        RouteEvalExample(
            query="扫地机器人滤网多久清洗一次？",
            expected_status=RouteStatus.DECISIVE,
            expected_route=RouteName.LOCAL_RAG,
            expected_tool="rag_summarize",
            expected_missing_slots=(),
        ),
        RouteEvalExample(
            query="今天适合拖地吗？",
            expected_status=RouteStatus.NEEDS_CLARIFICATION,
            expected_route=RouteName.BUSINESS_QUERY,
            expected_tool="get_weather",
            expected_missing_slots=("city",),
        ),
    ]

    results = evaluate_router(HybridRouter(), examples)
    summary = aggregate_results(results)

    assert summary["status_accuracy"] == 1.0
    assert summary["route_accuracy"] == 1.0
    assert summary["tool_selection_accuracy"] == 1.0
    assert summary["missing_slot_accuracy"] == 1.0
    assert summary["clarification_accuracy"] == 1.0
    assert summary["invalid_tool_rate"] == 0.0
    assert "avg_confidence" not in summary


def test_evaluation_handles_no_route_as_an_explicit_no_match_state():
    examples = [
        RouteEvalExample(
            query="这个怎么样？",
            expected_status=RouteStatus.NO_MATCH,
            expected_route=None,
            expected_tool="",
            expected_missing_slots=(),
        )
    ]

    result = evaluate_router(HybridRouter(), examples)[0]

    assert result.status_correct is True
    assert result.route_correct is True
    assert result.predicted_route is None


def test_missing_slot_accuracy_ignores_conflict_and_no_match_cases():
    examples = [
        RouteEvalExample(
            query="这个怎么样？",
            expected_status=RouteStatus.NO_MATCH,
            expected_route=None,
            expected_tool="",
            expected_missing_slots=(),
        ),
        RouteEvalExample(
            query="明天雨大吗",
            expected_status=RouteStatus.NEEDS_CLARIFICATION,
            expected_route=RouteName.BUSINESS_QUERY,
            expected_tool="get_weather",
            expected_missing_slots=("city",),
        ),
    ]

    summary = aggregate_results(evaluate_router(HybridRouter(), examples))

    assert summary["missing_slot_accuracy"] == 0.0


def test_evaluation_exposes_per_route_macro_f1():
    examples = [
        RouteEvalExample(
            query="你好",
            expected_status=RouteStatus.DECISIVE,
            expected_route=RouteName.DIRECT,
            expected_tool="",
            expected_missing_slots=(),
        ),
        RouteEvalExample(
            query="这个怎么样？",
            expected_status=RouteStatus.DECISIVE,
            expected_route=RouteName.LOCAL_RAG,
            expected_tool="rag_summarize",
            expected_missing_slots=(),
        ),
    ]

    summary = aggregate_results(evaluate_router(HybridRouter(), examples))

    assert 0.0 <= summary["macro_f1"] <= 1.0
    assert summary["route_count"] == 2.0


def test_loads_status_and_semicolon_separated_missing_slots(tmp_path: Path):
    dataset = tmp_path / "routes.csv"
    dataset.write_text(
        "query,expected_status,expected_route,expected_tool,expected_missing_slots\n"
        "生成使用报告,needs_clarification,business_query,fetch_external_data,user_id;month\n",
        encoding="utf-8",
    )

    example = load_route_dataset(dataset)[0]

    assert example.expected_status is RouteStatus.NEEDS_CLARIFICATION
    assert example.expected_missing_slots == ("user_id", "month")


def test_loads_union_from_multiple_route_dataset_files(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    header = "query,expected_status,expected_route,expected_tool,expected_missing_slots\n"
    first.write_text(header + "你好,decisive,direct,,\n", encoding="utf-8")
    second.write_text(header + "这个怎么样？,no_match,,,\n", encoding="utf-8")

    examples = load_route_datasets([first, second])

    assert [example.query for example in examples] == ["你好", "这个怎么样？"]


def test_report_contains_status_and_slots_but_not_confidence(tmp_path: Path):
    example = RouteEvalExample(
        query="今天适合拖地吗？",
        expected_status=RouteStatus.NEEDS_CLARIFICATION,
        expected_route=RouteName.BUSINESS_QUERY,
        expected_tool="get_weather",
        expected_missing_slots=("city",),
    )
    results = evaluate_router(HybridRouter(), [example])
    output = tmp_path / "report.md"

    save_route_report(
        results=results,
        summary=aggregate_results(results),
        output_path=output,
        dataset_path=tmp_path / "dataset.csv",
    )

    report = output.read_text(encoding="utf-8")
    assert "status_accuracy" in report
    assert "missing_slot_accuracy" in report
    assert "confidence" not in report


def test_evaluation_router_is_rule_only_without_llm_flag():
    router = build_evaluation_router(with_llm=False)

    assert router.llm_classifier is None


def test_invalid_tool_rate_checks_predicted_route_compatibility():
    class FakeRouter:
        def route(self, _: str):
            return RouteDecision(
                status=RouteStatus.DECISIVE,
                route=RouteName.BUSINESS_QUERY,
                reason_code="model_selected_weather",
                tool_candidates=("get_weather",),
                evidence_spans=("天气",),
                source="llm",
            )

    example = RouteEvalExample(
        query="主刷报错，顺便查天气",
        expected_status=RouteStatus.CONFLICT,
        expected_route=None,
        expected_tool="",
        expected_missing_slots=(),
    )

    summary = aggregate_results(evaluate_router(FakeRouter(), [example]))

    assert summary["invalid_tool_rate"] == 0.0
    assert summary["validation_fallback_rate"] == 0.0


def test_explicit_dataset_arguments_replace_default_dataset(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["route_evaluation", "--dataset", "one.csv", "--dataset", "two.csv"],
    )

    args = parse_args()

    assert args.dataset == ["one.csv", "two.csv"]
