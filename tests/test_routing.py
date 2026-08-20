import pytest

from agent.routing import (
    HybridRouter,
    RouteDecision,
    RouteName,
    RouteStatus,
    apply_route_guidance,
    build_clarification_response,
    classify_query,
    route_tool_names,
    render_route_guidance,
)


@pytest.mark.parametrize(
    ("query", "route", "tools"),
    [
        ("扫地机器人滤网多久清洗一次？", RouteName.LOCAL_RAG, ("rag_summarize",)),
        ("2026 年扫地机器人主流品牌对比有哪些变化？", RouteName.WEB_RAG, ("web_search",)),
        ("深圳今天适合使用拖地功能吗？", RouteName.BUSINESS_QUERY, ("get_weather",)),
        (
            "帮我查询用户 1001 在 2025-01 的扫地机器人使用报告。",
            RouteName.BUSINESS_QUERY,
            ("fetch_external_data",),
        ),
        ("扫地机器人主刷缠绕并报错 E01，应该怎么处理？", RouteName.TROUBLESHOOTING, ("rag_summarize",)),
        ("你能提供哪些服务？", RouteName.DIRECT, ()),
    ],
)
def test_decisive_rules_return_route_without_pseudo_confidence(query, route, tools):
    decision = classify_query(query)

    assert decision.status is RouteStatus.DECISIVE
    assert decision.route is route
    assert decision.tool_candidates == tools
    assert not hasattr(decision, "confidence")


def test_missing_weather_city_requests_clarification_without_calling_tool():
    decision = classify_query("今天适合拖地吗？")

    assert decision.status is RouteStatus.NEEDS_CLARIFICATION
    assert decision.route is RouteName.BUSINESS_QUERY
    assert decision.tool_candidates == ("get_weather",)
    assert decision.missing_slots == ("city",)
    assert decision.requires_clarification is True


def test_missing_report_parameters_are_exposed_as_slots():
    decision = classify_query("帮我看看用户 1001 最近的设备使用情况")

    assert decision.status is RouteStatus.NEEDS_CLARIFICATION
    assert decision.route is RouteName.BUSINESS_QUERY
    assert decision.tool_candidates == ("fetch_external_data",)
    assert decision.missing_slots == ("month",)


def test_invalid_calendar_month_is_treated_as_missing_slot():
    decision = classify_query("查询用户 1001 在 2025-13 的使用报告")

    assert decision.status is RouteStatus.NEEDS_CLARIFICATION
    assert decision.missing_slots == ("month",)


def test_empty_query_requests_user_request_slot():
    decision = classify_query("  ")

    assert decision.status is RouteStatus.NEEDS_CLARIFICATION
    assert decision.route is None
    assert decision.missing_slots == ("user_request",)


def test_unmatched_query_returns_no_match_instead_of_guessing_direct_route():
    decision = classify_query("这个怎么样？")

    assert decision.status is RouteStatus.NO_MATCH
    assert decision.route is None
    assert decision.requires_clarification is True


def test_multi_intent_query_returns_conflict_with_evidence():
    decision = classify_query("主刷报错 E01，顺便查一下北京天气")

    assert decision.status is RouteStatus.CONFLICT
    assert decision.route is None
    assert set(decision.evidence_spans) >= {"报错", "天气"}


def test_fault_wording_is_not_conflict_when_recent_only_describes_frequency():
    decision = classify_query("主刷最近总是缠绕，怎么办？")

    assert decision.status is RouteStatus.DECISIVE
    assert decision.route is RouteName.TROUBLESHOOTING


def test_negated_fault_signal_does_not_trigger_troubleshooting():
    decision = classify_query("机器没有报错，只想知道滤网多久清洗一次")

    assert decision.status is RouteStatus.DECISIVE
    assert decision.route is RouteName.LOCAL_RAG


def test_negated_weather_intent_does_not_override_product_question():
    decision = classify_query("我不是要查天气，只想问滤网怎么清洗")

    assert decision.status is RouteStatus.DECISIVE
    assert decision.route is RouteName.LOCAL_RAG


def test_error_code_pattern_routes_to_troubleshooting():
    decision = classify_query("E01 是什么意思？")

    assert decision.status is RouteStatus.DECISIVE
    assert decision.route is RouteName.TROUBLESHOOTING
    assert decision.evidence_spans == ("E01",)


def test_common_stuck_typo_routes_to_troubleshooting_before_local_rag():
    decision = classify_query("扫地机主刷卡主了咋弄")

    assert decision.status is RouteStatus.DECISIVE
    assert decision.route is RouteName.TROUBLESHOOTING


def test_purchase_recommendation_keeps_web_route_with_local_constraints():
    decision = classify_query("最近推荐一款适合小户型的扫地机器人")

    assert decision.status is RouteStatus.DECISIVE
    assert decision.route is RouteName.WEB_RAG


def test_decisive_rule_does_not_call_small_model():
    calls = 0

    def classify_with_small_model(_: str) -> RouteDecision:
        nonlocal calls
        calls += 1
        return RouteDecision(
            status=RouteStatus.DECISIVE,
            route=RouteName.WEB_RAG,
            reason_code="freshness_required",
            tool_candidates=("web_search",),
            evidence_spans=("最新",),
            source="llm",
        )

    decision = HybridRouter(llm_classifier=classify_with_small_model).route(
        "扫地机器人滤网多久清洗一次？"
    )

    assert decision.route is RouteName.LOCAL_RAG
    assert decision.source == "rule"
    assert calls == 0


def test_missing_slots_do_not_call_small_model():
    calls = 0

    def classify_with_small_model(_: str) -> RouteDecision:
        nonlocal calls
        calls += 1
        raise AssertionError("missing slots should be handled deterministically")

    decision = HybridRouter(llm_classifier=classify_with_small_model).route(
        "今天适合拖地吗？"
    )

    assert decision.status is RouteStatus.NEEDS_CLARIFICATION
    assert calls == 0


def test_no_match_uses_validated_small_model_decision():
    def classify_with_small_model(_: str) -> RouteDecision:
        return RouteDecision(
            status=RouteStatus.DECISIVE,
            route=RouteName.LOCAL_RAG,
            reason_code="product_knowledge",
            tool_candidates=("rag_summarize",),
            evidence_spans=("这个",),
            source="llm",
        )

    decision = HybridRouter(llm_classifier=classify_with_small_model).route(
        "这个机器人怎么样？"
    )

    assert decision.route is RouteName.LOCAL_RAG
    assert decision.source == "llm"


def test_conflict_uses_small_model_instead_of_rule_priority():
    def classify_with_small_model(_: str) -> RouteDecision:
        return RouteDecision(
            status=RouteStatus.NEEDS_CLARIFICATION,
            route=None,
            reason_code="multi_intent_requires_split",
            missing_slots=("primary_intent",),
            evidence_spans=("报错", "天气"),
            source="llm",
        )

    decision = HybridRouter(llm_classifier=classify_with_small_model).route(
        "主刷报错 E01，顺便查一下北京天气"
    )

    assert decision.status is RouteStatus.NEEDS_CLARIFICATION
    assert decision.source == "llm"


def test_invalid_small_model_output_falls_back_to_clarification():
    def classify_with_small_model(_: str) -> RouteDecision:
        raise ValueError("invalid evidence span")

    decision = HybridRouter(llm_classifier=classify_with_small_model).route("这个怎么样？")

    assert decision.status is RouteStatus.NO_MATCH
    assert decision.route is None
    assert decision.source == "rule_fallback"
    assert decision.requires_clarification is True


def test_guidance_contains_status_and_never_renders_confidence():
    guidance = render_route_guidance(classify_query("扫地机器人滤网多久清洗一次？"))

    assert "当前路由：local_rag" in guidance
    assert "决策状态：decisive" in guidance
    assert "rag_summarize" in guidance
    assert "置信度" not in guidance


@pytest.mark.parametrize(
    ("query", "allowed_tools"),
    [
        ("你好", ()),
        ("扫地机器人滤网多久清洗一次？", ("rag_summarize",)),
        ("2026 年扫地机器人主流品牌对比有哪些变化？", ("web_search",)),
        ("深圳今天适合拖地吗？", ("get_weather",)),
        (
            "帮我查询用户 1001 在 2025-01 的扫地机器人使用报告。",
            (
                "fetch_external_data",
                "fill_context_for_report",
                "rag_summarize",
            ),
        ),
    ],
)
def test_route_exposes_only_tools_needed_by_the_selected_execution_chain(
    query, allowed_tools
):
    decision = classify_query(query)

    assert route_tool_names(decision) == allowed_tools


def test_route_guidance_declares_route_specific_tool_boundary():
    guidance = render_route_guidance(
        classify_query("2026 年扫地机器人主流品牌对比有哪些变化？")
    )

    assert "仅允许调用：web_search" in guidance
    assert "不得调用本地知识库或业务工具" in guidance


def test_guidance_uses_missing_slot_to_ask_for_city():
    guidance = render_route_guidance(classify_query("今天适合拖地吗？"))

    assert "先向用户询问城市" in guidance
    assert "get_weather" not in guidance


def test_builds_customer_facing_question_for_missing_city():
    response = build_clarification_response(classify_query("今天适合拖地吗？"))

    assert response == "请告诉我您所在的城市，我再结合当地天气判断是否适合拖地。"


def test_builds_customer_facing_question_for_missing_report_month():
    response = build_clarification_response(
        classify_query("帮我看看用户 1001 最近的设备使用情况")
    )

    assert "月份" in response
    assert "YYYY-MM" in response


def test_decisive_route_has_no_clarification_response():
    response = build_clarification_response(
        classify_query("扫地机器人滤网多久清洗一次？")
    )

    assert response is None


def test_injects_route_guidance_into_system_prompt():
    prompt = apply_route_guidance(
        "你是扫地机器人客服。",
        classify_query("扫地机器人滤网多久清洗一次？"),
    )

    assert prompt.startswith("你是扫地机器人客服。")
    assert "当前路由：local_rag" in prompt
    assert "rag_summarize" in prompt
