from agent.baseline_evaluation import (
    BaselineToolResult,
    aggregate_baseline_results,
    extract_predicted_tool,
)


def test_extracts_first_structured_tool_call():
    class Response:
        tool_calls = [
            {"name": "get_weather", "args": {"city": "深圳"}},
            {"name": "web_search", "args": {"query": "深圳天气"}},
        ]

    assert extract_predicted_tool(Response()) == "get_weather"


def test_returns_empty_tool_for_normal_text_response():
    class Response:
        tool_calls = []
        content = "你好，我可以帮助你。"

    assert extract_predicted_tool(Response()) == ""


def test_aggregates_baseline_tool_metrics():
    results = [
        BaselineToolResult(
            query="天气",
            expected_tool="get_weather",
            predicted_tool="get_weather",
            tool_correct=True,
            invalid_tool=False,
            unnecessary_tool=False,
            latency_ms=10.0,
        ),
        BaselineToolResult(
            query="滤网",
            expected_tool="rag_summarize",
            predicted_tool="web_search",
            tool_correct=False,
            invalid_tool=False,
            unnecessary_tool=False,
            latency_ms=20.0,
        ),
        BaselineToolResult(
            query="你好",
            expected_tool="",
            predicted_tool="web_search",
            tool_correct=False,
            invalid_tool=False,
            unnecessary_tool=True,
            latency_ms=30.0,
        ),
    ]

    summary = aggregate_baseline_results(results)

    assert summary["tool_selection_accuracy"] == 0.5
    assert summary["unnecessary_tool_rate"] == 1 / 3
    assert summary["invalid_tool_rate"] == 0.0
    assert summary["p95_latency_ms"] == 30.0
