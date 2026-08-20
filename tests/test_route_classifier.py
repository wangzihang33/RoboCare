import pytest

from agent.route_classifier import SmallLLMRouteClassifier, parse_route_prediction
from agent.routing import RouteName, RouteStatus


def test_parses_structured_local_rag_prediction_with_grounded_evidence():
    decision = parse_route_prediction(
        '{"status":"decisive","route":"local_rag","tool":"",'
        '"evidence_spans":["滤网"],"missing_slots":[],"reason_code":"product_knowledge"}',
        query="这个滤网多久清洗一次？",
    )

    assert decision.status is RouteStatus.DECISIVE
    assert decision.route is RouteName.LOCAL_RAG
    assert decision.tool_candidates == ("rag_summarize",)
    assert decision.evidence_spans == ("滤网",)
    assert not hasattr(decision, "confidence")


def test_rejects_unknown_route_prediction():
    with pytest.raises(ValueError, match="未知路由"):
        parse_route_prediction(
            '{"status":"decisive","route":"unknown","tool":"",'
            '"evidence_spans":["滤网"],"missing_slots":[],"reason_code":"x"}',
            query="滤网怎么洗？",
        )


def test_rejects_self_reported_confidence_field():
    with pytest.raises(ValueError, match="不支持的字段"):
        parse_route_prediction(
            '{"status":"decisive","route":"local_rag","tool":"",'
            '"confidence":0.99,"evidence_spans":["滤网"],"missing_slots":[],'
            '"reason_code":"product_knowledge"}',
            query="滤网怎么洗？",
        )


def test_rejects_evidence_span_not_present_in_query():
    with pytest.raises(ValueError, match="证据片段"):
        parse_route_prediction(
            '{"status":"decisive","route":"local_rag","tool":"",'
            '"evidence_spans":["主刷"],"missing_slots":[],"reason_code":"product_knowledge"}',
            query="滤网怎么洗？",
        )


def test_rejects_decisive_weather_route_when_city_slot_is_missing():
    with pytest.raises(ValueError, match="缺少必要槽位"):
        parse_route_prediction(
            '{"status":"decisive","route":"business_query","tool":"get_weather",'
            '"evidence_spans":["天气"],"missing_slots":[],"reason_code":"weather_query"}',
            query="今天天气怎么样？",
        )


def test_parses_weather_route_when_city_slot_is_complete():
    decision = parse_route_prediction(
        '{"status":"decisive","route":"business_query","tool":"get_weather",'
        '"evidence_spans":["北京","天气"],"missing_slots":[],"reason_code":"weather_query"}',
        query="北京今天天气怎么样？",
    )

    assert decision.route is RouteName.BUSINESS_QUERY
    assert decision.tool_candidates == ("get_weather",)


def test_requires_missing_slots_for_clarification_prediction():
    with pytest.raises(ValueError, match="missing_slots"):
        parse_route_prediction(
            '{"status":"needs_clarification","route":null,"tool":"",'
            '"evidence_spans":["这个"],"missing_slots":[],"reason_code":"ambiguous_reference"}',
            query="这个怎么样？",
        )


def test_rejects_clarification_prediction_that_still_selects_a_tool():
    with pytest.raises(ValueError, match="route=null"):
        parse_route_prediction(
            '{"status":"needs_clarification","route":"business_query",'
            '"tool":"get_weather","evidence_spans":["雨"],'
            '"missing_slots":["city"],"reason_code":"missing_city"}',
            query="明天雨大吗",
        )


def test_rejects_decisive_route_for_out_of_domain_query():
    with pytest.raises(ValueError, match="客服领域"):
        parse_route_prediction(
            '{"status":"decisive","route":"local_rag","tool":"",'
            '"evidence_spans":["离职申请"],"missing_slots":[],'
            '"reason_code":"local_doc_generation"}',
            query="帮我写一份离职申请",
        )


def test_classifier_calls_model_and_returns_validated_prediction():
    class FakeModel:
        def invoke(self, prompt: str):
            assert "confidence" not in prompt
            return (
                '{"status":"decisive","route":"business_query","tool":"get_weather",'
                '"evidence_spans":["深圳","天气"],"missing_slots":[],'
                '"reason_code":"weather_query"}'
            )

    decision = SmallLLMRouteClassifier(FakeModel())("深圳今天天气怎么样？")

    assert decision.route is RouteName.BUSINESS_QUERY
    assert decision.tool_candidates == ("get_weather",)
    assert decision.source == "llm"
