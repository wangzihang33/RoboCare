from __future__ import annotations

import json
import re
from typing import Any

from agent.routing import (
    RouteDecision,
    RouteName,
    RouteStatus,
    validate_model_route_decision,
)


ROUTE_CLASSIFIER_PROMPT = """你是客服请求路由器，只负责选择执行链路，不回答用户问题。
可选 route：direct、local_rag、web_rag、business_query、troubleshooting。
可选 status：decisive、needs_clarification。
当 status=needs_clarification 时，route 必须为 null、tool 必须为空，只提出澄清问题；不要在澄清时选择任何工具。
business_query 可选 tool：get_weather、fetch_external_data；其他 route 的 tool 必须为空。
evidence_spans 必须逐字来自用户问题；信息不足时将缺失字段写入 missing_slots。
只输出以下 JSON，不要输出 Markdown、概率或额外字段：
{{"status":"...","route":"...","tool":"","evidence_spans":["..."],"missing_slots":[],"reason_code":"..."}}

用户问题：
<query>
{query}
</query>
""".strip()

_ALLOWED_FIELDS = {
    "status",
    "route",
    "tool",
    "evidence_spans",
    "missing_slots",
    "reason_code",
}
_ROUTE_TOOLS = {
    RouteName.DIRECT: (),
    RouteName.LOCAL_RAG: ("rag_summarize",),
    RouteName.WEB_RAG: ("web_search",),
    RouteName.TROUBLESHOOTING: ("rag_summarize",),
}
_BUSINESS_TOOLS = {"get_weather", "fetch_external_data"}


class SmallLLMRouteClassifier:
    def __init__(self, model: Any):
        self.model = model

    def __call__(self, query: str) -> RouteDecision:
        response = self.model.invoke(ROUTE_CLASSIFIER_PROMPT.format(query=query))
        raw_content = getattr(response, "content", response)
        if isinstance(raw_content, list):
            raw_content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in raw_content
            )
        return parse_route_prediction(str(raw_content), query=query)


def parse_route_prediction(raw_output: str, *, query: str) -> RouteDecision:
    payload = _extract_json_object(raw_output)
    unknown_fields = set(payload) - _ALLOWED_FIELDS
    if unknown_fields:
        raise ValueError(f"小模型输出包含不支持的字段: {sorted(unknown_fields)}")
    missing_fields = _ALLOWED_FIELDS - set(payload)
    if missing_fields:
        raise ValueError(f"小模型输出缺少字段: {sorted(missing_fields)}")

    try:
        status = RouteStatus(str(payload["status"]).strip())
    except ValueError as exc:
        raise ValueError(f"未知路由状态: {payload.get('status')}") from exc

    raw_route = payload.get("route")
    if raw_route in {None, ""}:
        route = None
    else:
        try:
            route = RouteName(str(raw_route).strip())
        except ValueError as exc:
            raise ValueError(f"未知路由: {raw_route}") from exc

    tool = str(payload.get("tool") or "").strip()
    if route is RouteName.BUSINESS_QUERY:
        if tool not in _BUSINESS_TOOLS:
            raise ValueError(f"业务路由工具不受支持: {tool}")
        tool_candidates = (tool,)
    elif route is None:
        if tool:
            raise ValueError("无路由时不应指定工具")
        tool_candidates = ()
    else:
        if tool:
            raise ValueError(f"路由 {route.value} 不应指定工具: {tool}")
        tool_candidates = _ROUTE_TOOLS[route]

    evidence_spans = _parse_string_list(payload["evidence_spans"], "evidence_spans")
    missing_slots = _parse_string_list(payload["missing_slots"], "missing_slots")
    reason_code = str(payload.get("reason_code") or "").strip()
    if not reason_code:
        raise ValueError("reason_code 不能为空")

    decision = RouteDecision(
        status=status,
        route=route,
        reason_code=reason_code,
        tool_candidates=tool_candidates,
        missing_slots=missing_slots,
        evidence_spans=evidence_spans,
        source="llm",
    )
    validate_model_route_decision(decision, query)
    return decision


def _parse_string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} 必须是字符串数组")
    cleaned = tuple(item.strip() for item in value if item.strip())
    if len(cleaned) != len(value):
        raise ValueError(f"{field_name} 不能包含空字符串")
    return tuple(dict.fromkeys(cleaned))


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    cleaned = raw_output.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
    if fenced_match:
        cleaned = fenced_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("小模型路由输出中没有 JSON 对象")
        cleaned = cleaned[start : end + 1]
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("小模型路由输出必须是 JSON 对象")
    return payload
