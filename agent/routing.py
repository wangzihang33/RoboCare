from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
import re


class RouteName(StrEnum):
    DIRECT = "direct"
    LOCAL_RAG = "local_rag"
    WEB_RAG = "web_rag"
    BUSINESS_QUERY = "business_query"
    TROUBLESHOOTING = "troubleshooting"


class RouteStatus(StrEnum):
    DECISIVE = "decisive"
    NEEDS_CLARIFICATION = "needs_clarification"
    CONFLICT = "conflict"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class RouteDecision:
    status: RouteStatus
    route: RouteName | None
    reason_code: str
    tool_candidates: tuple[str, ...] = ()
    missing_slots: tuple[str, ...] = ()
    evidence_spans: tuple[str, ...] = ()
    source: str = "rule"

    @property
    def requires_clarification(self) -> bool:
        return self.status is not RouteStatus.DECISIVE


_ROUTE_TOOLS = {
    RouteName.DIRECT: (),
    RouteName.LOCAL_RAG: ("rag_summarize",),
    RouteName.WEB_RAG: ("web_search",),
    RouteName.TROUBLESHOOTING: ("rag_summarize",),
}
_BUSINESS_TOOLS = {"get_weather", "fetch_external_data"}

_REPORT_TERMS = (
    "使用报告",
    "个人报告",
    "使用记录",
    "使用情况",
    "设备使用",
    "清洁效率",
    "耗材状态",
    "同类对比",
)
_WEATHER_TERMS = (
    "天气",
    "湿度",
    "降雨",
    "下雨",
    "雨天",
    "风力",
    "温度",
    "适合拖地",
    "适合使用",
)
_FRESHNESS_TERMS = (
    "最新",
    "当前",
    "今年",
    "近期",
    "最近",
    "品牌",
    "品牌对比",
    "行业趋势",
    "市场",
    "新闻",
    "新款",
    "值得买",
    "换一台",
    "推荐",
    "固件更新",
)
_FAULT_TERMS = (
    "故障",
    "报错",
    "错误码",
    "不工作",
    "无法",
    "不能",
    "卡住",
    "卡主",
    "缠绕",
    "漏水",
    "异响",
    "失联",
    "不出水",
    "不回充",
    "不吸尘",
)
_LOCAL_TERMS = (
    "滤网",
    "主刷",
    "边刷",
    "拖布",
    "水箱",
    "清洁",
    "清洗",
    "维护",
    "保养",
    "选购",
    "小户型",
    "地毯",
    "木地板",
    "禁拖区",
    "充电",
)
_CAPABILITY_TERMS = ("你能做什么", "你能提供", "有哪些服务", "有什么功能", "你是谁")
_DOMAIN_TERMS = (
    "扫地",
    "扫拖",
    "机器人",
    "设备",
    "滤网",
    "主刷",
    "边刷",
    "拖布",
    "水箱",
    "天气",
    "湿度",
    "下雨",
    "雨天",
    "品牌",
    "固件",
    "用户",
    "耗材",
    "使用",
    "故障",
    "报错",
    "错误码",
    "回充",
    "缠绕",
    "漏水",
    "不出水",
)
_WEB_PURCHASE_OR_MARKET_TERMS = {
    "品牌",
    "品牌对比",
    "行业趋势",
    "市场",
    "新闻",
    "新款",
    "值得买",
    "换一台",
    "推荐",
}
_CITY_PATTERN = re.compile(
    r"北京|上海|广州|深圳|杭州|南京|成都|重庆|武汉|西安|苏州|天津|青岛|厦门|"
    r"佛山|东莞|郑州|长沙|合肥|昆明|济南|沈阳|哈尔滨|大连|宁波|无锡|福州|"
    r"南昌|石家庄|太原|贵阳|南宁|海口|兰州|[\u4e00-\u9fff]{2,8}[市县]"
)
_USER_ID_PATTERN = re.compile(r"用户?\s*\d{3,}")
_MONTH_PATTERN = re.compile(r"20\d{2}[-/](?:0?[1-9]|1[0-2])(?!\d)")


class HybridRouter:
    """Use rules for deterministic cases and a small model for semantic ambiguity."""

    def __init__(
        self,
        *,
        llm_classifier: Callable[[str], RouteDecision] | None = None,
    ) -> None:
        self.llm_classifier = llm_classifier

    def route(self, query: str) -> RouteDecision:
        rule_decision = classify_query(query)
        if (
            rule_decision.status
            not in {RouteStatus.CONFLICT, RouteStatus.NO_MATCH}
            or self.llm_classifier is None
        ):
            return rule_decision

        try:
            llm_decision = self.llm_classifier(query)
            validate_model_route_decision(llm_decision, query)
            return llm_decision
        except Exception as exc:
            return replace(
                rule_decision,
                reason_code=f"llm_validation_failed:{type(exc).__name__}",
                source="rule_fallback",
            )


def classify_query(query: str) -> RouteDecision:
    text = " ".join((query or "").split())
    if not text:
        return RouteDecision(
            status=RouteStatus.NEEDS_CLARIFICATION,
            route=None,
            reason_code="empty_request",
            missing_slots=("user_request",),
        )

    report_evidence = _find_non_negated_evidence(text, _REPORT_TERMS)
    weather_evidence = _find_non_negated_evidence(text, _WEATHER_TERMS)
    fault_evidence = _find_non_negated_evidence(text, _FAULT_TERMS)
    error_code_match = re.search(r"\bE\d{2,4}\b", text, re.IGNORECASE)
    if error_code_match:
        fault_evidence = _append_unique(fault_evidence, (error_code_match.group(),))
    web_evidence = _find_non_negated_evidence(text, _FRESHNESS_TERMS)
    local_evidence = _find_evidence(text, _LOCAL_TERMS)
    if re.search(r"20\d{2}年", text):
        web_evidence = _append_unique(web_evidence, (re.search(r"20\d{2}年", text).group(),))

    strong_intents: list[tuple[str, tuple[str, ...]]] = []
    if report_evidence:
        strong_intents.append(("usage_report", report_evidence))
    if weather_evidence:
        strong_intents.append(("weather", weather_evidence))
    if fault_evidence:
        strong_intents.append(("troubleshooting", fault_evidence))

    contextual_web_terms = {"当前", "近期", "最近"}
    web_is_contextual = bool(web_evidence) and set(web_evidence) <= contextual_web_terms
    if web_evidence and not (
        web_is_contextual
        and (report_evidence or weather_evidence or fault_evidence or local_evidence)
    ):
        strong_intents.append(("web", web_evidence))

    local_web_conflict = bool(
        web_evidence
        and local_evidence
        and not web_is_contextual
        and not (_WEB_PURCHASE_OR_MARKET_TERMS & set(web_evidence))
    )
    if len(strong_intents) > 1 or local_web_conflict:
        evidence = tuple(
            span
            for _, spans in strong_intents
            for span in spans
        )
        if local_web_conflict:
            evidence = _append_unique(evidence, local_evidence)
        return RouteDecision(
            status=RouteStatus.CONFLICT,
            route=None,
            reason_code="multiple_route_signals",
            evidence_spans=_deduplicate(evidence),
        )

    if report_evidence:
        missing = _missing_report_slots(text)
        return RouteDecision(
            status=(
                RouteStatus.NEEDS_CLARIFICATION if missing else RouteStatus.DECISIVE
            ),
            route=RouteName.BUSINESS_QUERY,
            reason_code="usage_report_missing_slots" if missing else "usage_report",
            tool_candidates=("fetch_external_data",),
            missing_slots=missing,
            evidence_spans=report_evidence,
        )

    if weather_evidence:
        missing = () if _has_city(text) else ("city",)
        return RouteDecision(
            status=(
                RouteStatus.NEEDS_CLARIFICATION if missing else RouteStatus.DECISIVE
            ),
            route=RouteName.BUSINESS_QUERY,
            reason_code="weather_missing_city" if missing else "weather_query",
            tool_candidates=("get_weather",),
            missing_slots=missing,
            evidence_spans=weather_evidence,
        )

    if fault_evidence:
        return RouteDecision(
            status=RouteStatus.DECISIVE,
            route=RouteName.TROUBLESHOOTING,
            reason_code="device_fault",
            tool_candidates=_ROUTE_TOOLS[RouteName.TROUBLESHOOTING],
            evidence_spans=fault_evidence,
        )

    if web_evidence:
        return RouteDecision(
            status=RouteStatus.DECISIVE,
            route=RouteName.WEB_RAG,
            reason_code="freshness_required",
            tool_candidates=_ROUTE_TOOLS[RouteName.WEB_RAG],
            evidence_spans=web_evidence,
        )

    if local_evidence:
        return RouteDecision(
            status=RouteStatus.DECISIVE,
            route=RouteName.LOCAL_RAG,
            reason_code="product_knowledge",
            tool_candidates=_ROUTE_TOOLS[RouteName.LOCAL_RAG],
            evidence_spans=local_evidence,
        )

    capability_evidence = _find_evidence(text, _CAPABILITY_TERMS)
    if capability_evidence:
        return RouteDecision(
            status=RouteStatus.DECISIVE,
            route=RouteName.DIRECT,
            reason_code="capability_intro",
            evidence_spans=capability_evidence,
        )

    if _is_greeting(text):
        return RouteDecision(
            status=RouteStatus.DECISIVE,
            route=RouteName.DIRECT,
            reason_code="greeting",
            evidence_spans=(text,),
        )

    return RouteDecision(
        status=RouteStatus.NO_MATCH,
        route=None,
        reason_code="no_rule_match",
    )


def validate_model_route_decision(decision: RouteDecision, query: str) -> None:
    if decision.status not in {
        RouteStatus.DECISIVE,
        RouteStatus.NEEDS_CLARIFICATION,
    }:
        raise ValueError("小模型只能输出 decisive 或 needs_clarification")

    if decision.status is RouteStatus.DECISIVE:
        if decision.route is None:
            raise ValueError("decisive 路由不能为空")
        if decision.missing_slots:
            raise ValueError("decisive 路由不能包含 missing_slots")
    elif not decision.missing_slots:
        raise ValueError("needs_clarification 必须包含 missing_slots")
    if decision.status is RouteStatus.NEEDS_CLARIFICATION and (
        decision.route is not None or decision.tool_candidates
    ):
        raise ValueError("needs_clarification 必须使用 route=null 且 tool=\"\"")

    if not decision.evidence_spans:
        raise ValueError("小模型路由必须提供证据片段")
    invalid_spans = [span for span in decision.evidence_spans if span not in query]
    if invalid_spans:
        raise ValueError(f"证据片段不在原问题中: {invalid_spans}")

    if decision.status is RouteStatus.DECISIVE:
        if decision.route is RouteName.DIRECT:
            if not (_is_greeting(query) or _contains_any(query, _CAPABILITY_TERMS)):
                raise ValueError("direct 路由超出客服领域")
        elif not _contains_any(query, _DOMAIN_TERMS):
            raise ValueError("路由问题超出客服领域")

    if decision.route is None:
        if decision.tool_candidates:
            raise ValueError("无路由时不能指定工具")
        return

    if decision.route is RouteName.BUSINESS_QUERY:
        if len(decision.tool_candidates) != 1:
            raise ValueError("业务路由必须指定一个工具")
        tool = decision.tool_candidates[0]
        if tool not in _BUSINESS_TOOLS:
            raise ValueError(f"业务路由工具不受支持: {tool}")
        missing = (
            (() if _has_city(query) else ("city",))
            if tool == "get_weather"
            else _missing_report_slots(query)
        )
        if decision.status is RouteStatus.DECISIVE and missing:
            raise ValueError(f"业务路由缺少必要槽位: {missing}")
        if decision.status is RouteStatus.NEEDS_CLARIFICATION and not set(missing).issubset(
            decision.missing_slots
        ):
            raise ValueError(f"missing_slots 未覆盖业务必填槽位: {missing}")
        return

    expected_tools = _ROUTE_TOOLS[decision.route]
    if decision.tool_candidates != expected_tools:
        raise ValueError(
            f"路由 {decision.route.value} 的工具应为 {expected_tools}"
        )


def route_tool_names(decision: RouteDecision) -> tuple[str, ...]:
    """Return the narrow tool set exposed to the selected execution chain."""
    if decision.status is not RouteStatus.DECISIVE or decision.route is None:
        return ()
    tool_names = list(decision.tool_candidates)
    if "fetch_external_data" in tool_names:
        tool_names.append("fill_context_for_report")
        tool_names.append("rag_summarize")
    return tuple(dict.fromkeys(tool_names))


def render_route_guidance(decision: RouteDecision) -> str:
    """Render deterministic routing guidance for the main Agent."""
    if decision.status is RouteStatus.NEEDS_CLARIFICATION:
        if "city" in decision.missing_slots:
            return "当前路由为业务查询，但天气查询缺少城市；先向用户询问城市，不要调用天气工具。"
        if decision.route is RouteName.BUSINESS_QUERY:
            labels = {"user_id": "用户 ID", "month": "月份（YYYY-MM）"}
            missing = "、".join(labels.get(slot, slot) for slot in decision.missing_slots)
            return f"当前路由为业务查询，但缺少{missing}；先补全参数，不要调用用户数据工具。"
        return "当前问题信息不足；根据缺失信息向用户提出一个具体问题，不要强行调用工具。"

    if decision.status is RouteStatus.CONFLICT:
        return "检测到多个业务意图；先确认用户希望优先处理的任务，再选择单一执行链路。"
    if decision.status is RouteStatus.NO_MATCH or decision.route is None:
        return "当前问题未匹配到可靠路由；先澄清具体设备、场景或目标，不要猜测或调用工具。"

    route_descriptions = {
        RouteName.DIRECT: "直接回答，不调用外部工具；不得调用任何检索或业务工具",
        RouteName.LOCAL_RAG: "仅使用本地知识库工具 rag_summarize；不得调用联网搜索或业务工具",
        RouteName.WEB_RAG: "仅使用联网检索工具 web_search，并保留来源信息；不得调用本地知识库或业务工具",
        RouteName.BUSINESS_QUERY: "仅按参数要求调用当前候选业务工具；报告解释阶段可使用本地知识库，不调用联网搜索",
        RouteName.TROUBLESHOOTING: "仅使用本地知识库工具 rag_summarize 获取故障处理依据；不得调用天气、业务查询或联网搜索工具",
    }
    allowed_tools = route_tool_names(decision)
    allowed_text = "、".join(allowed_tools) if allowed_tools else "无"
    return (
        f"当前路由：{decision.route.value}；决策状态：{decision.status.value}；"
        f"仅允许调用：{allowed_text}。{route_descriptions[decision.route]}。"
        "若证据不足，不要编造事实，先补充检索或说明信息不足。"
    )


def build_clarification_response(decision: RouteDecision) -> str | None:
    """Build a user-facing follow-up without invoking the main Agent."""
    if decision.status is RouteStatus.DECISIVE:
        return None
    if "city" in decision.missing_slots:
        return "请告诉我您所在的城市，我再结合当地天气判断是否适合拖地。"
    if decision.route is RouteName.BUSINESS_QUERY:
        labels = {"user_id": "用户 ID", "month": "月份（YYYY-MM）"}
        missing = "和".join(labels.get(slot, slot) for slot in decision.missing_slots)
        return f"请补充{missing}，我再为您查询对应的设备使用数据。"
    if decision.status is RouteStatus.CONFLICT or "primary_intent" in decision.missing_slots:
        return "您这次提到了多个需求，请先告诉我希望优先处理哪一个，我会按顺序协助您。"
    return "请再补充一下具体设备、遇到的现象或想了解的功能，我再为您处理。"


def apply_route_guidance(base_prompt: str, decision: RouteDecision | None) -> str:
    if decision is None:
        return base_prompt
    return f"{base_prompt.rstrip()}\n\n### 当前路由决策\n{render_route_guidance(decision)}"


def _find_evidence(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(term for term in terms if term in text)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _find_non_negated_evidence(
    text: str,
    terms: tuple[str, ...],
) -> tuple[str, ...]:
    evidence: list[str] = []
    for term in terms:
        for match in re.finditer(re.escape(term), text):
            prefix = text[max(0, match.start() - 8) : match.start()]
            if not re.search(
                r"(?:没有|并未|不是(?:要|想)?(?:查|问|看)?|"
                r"不(?:想|要|需要)(?:查|问|看)?|无需(?:查|问|看)?|无|没|未)\s*$",
                prefix,
            ):
                evidence.append(term)
                break
    return tuple(evidence)


def _append_unique(current: tuple[str, ...], extra: tuple[str, ...]) -> tuple[str, ...]:
    return _deduplicate(current + extra)


def _deduplicate(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _has_city(text: str) -> bool:
    return bool(_CITY_PATTERN.search(text))


def _missing_report_slots(text: str) -> tuple[str, ...]:
    missing: list[str] = []
    if not _USER_ID_PATTERN.search(text):
        missing.append("user_id")
    if not _MONTH_PATTERN.search(text):
        missing.append("month")
    return tuple(missing)


def _is_greeting(text: str) -> bool:
    return text.strip().lower() in {"你好", "您好", "嗨", "谢谢", "谢谢你", "再见"}
