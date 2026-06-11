from __future__ import annotations

import re
from typing import Any

import requests

from agent.tools.tool_result import error_result, ok_result
from agent.tools.user_data_store import UserUsageStore
from utils.config_handler import agent_conf
from utils.logger_handler import logger


MAX_QUERY_CHARS = 500
MAX_CITY_CHARS = 40
USER_ID_RE = re.compile(r"^\d{3,20}$")
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

user_usage_store = UserUsageStore()
_rag_service = None
_web_service = None


def rag_summarize_core(query: str) -> dict[str, Any]:
    cleaned_query = _clean_text(query)
    if not cleaned_query:
        return error_result("rag_summarize", "invalid_input", "query 不能为空")
    if len(cleaned_query) > MAX_QUERY_CHARS:
        return error_result(
            "rag_summarize",
            "invalid_input",
            f"query 过长，最大长度为 {MAX_QUERY_CHARS} 字符",
        )

    answer = _get_rag_service().rag_summarize(cleaned_query)
    return ok_result(
        "rag_summarize",
        {
            "query": cleaned_query,
            "answer": answer,
        },
    )


def web_search_core(query: str) -> dict[str, Any]:
    cleaned_query = _clean_text(query)
    if not cleaned_query:
        return error_result("web_search", "invalid_input", "query 不能为空")
    if len(cleaned_query) > MAX_QUERY_CHARS:
        return error_result(
            "web_search",
            "invalid_input",
            f"query 过长，最大长度为 {MAX_QUERY_CHARS} 字符",
        )

    answer = _get_web_service().search_summarize(cleaned_query)
    trace = get_web_search_trace()
    return ok_result(
        "web_search",
        {
            "query": cleaned_query,
            "answer": answer,
            "trace": trace,
        },
        meta={
            "cache_hit": trace.get("cache_hit"),
            "source_count": trace.get("source_count"),
        },
    )


def get_weather_core(city: str) -> dict[str, Any]:
    cleaned_city = _clean_text(city)
    if not cleaned_city:
        return error_result("get_weather", "invalid_input", "city 不能为空，请提供标准城市名称")
    if len(cleaned_city) > MAX_CITY_CHARS:
        return error_result(
            "get_weather",
            "invalid_input",
            f"city 过长，最大长度为 {MAX_CITY_CHARS} 字符",
        )

    amap_key = agent_conf.get("amap_key")
    if not amap_key:
        return error_result("get_weather", "missing_config", "未配置 AMAP_KEY，无法查询真实天气")

    api_url = agent_conf.get("amap_weather_api_url")
    if not api_url:
        return error_result(
            "get_weather",
            "missing_config",
            "未配置 AMAP_WEATHER_API_URL，无法查询真实天气",
        )

    try:
        response = requests.get(
            api_url,
            params={"city": cleaned_city, "key": amap_key, "extensions": "base"},
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.error(f"[get_weather] failed city={cleaned_city}: {exc}")
        return error_result(
            "get_weather",
            "request_failed",
            "高德天气接口请求失败",
            details={"exception": str(exc)},
        )

    if payload.get("status") != "1":
        return error_result(
            "get_weather",
            "provider_error",
            "高德天气接口返回失败",
            details={
                "info": payload.get("info"),
                "infocode": payload.get("infocode"),
            },
        )

    lives = payload.get("lives") or []
    if not lives:
        return error_result(
            "get_weather",
            "empty_result",
            f"未查询到城市 {cleaned_city} 的实时天气",
        )

    live = lives[0]
    return ok_result(
        "get_weather",
        {
            "city": live.get("city") or cleaned_city,
            "adcode": live.get("adcode"),
            "province": live.get("province"),
            "weather": live.get("weather"),
            "temperature": live.get("temperature"),
            "humidity": live.get("humidity"),
            "winddirection": live.get("winddirection"),
            "windpower": live.get("windpower"),
            "reporttime": live.get("reporttime"),
        },
        meta={"provider": "amap"},
    )


def fetch_external_data_core(user_id: str, month: str) -> dict[str, Any]:
    cleaned_user_id = _clean_text(user_id)
    cleaned_month = _clean_text(month)

    if not USER_ID_RE.match(cleaned_user_id):
        return error_result(
            "fetch_external_data",
            "invalid_input",
            "user_id 必须是 3 到 20 位数字字符串",
        )
    if not MONTH_RE.match(cleaned_month):
        return error_result(
            "fetch_external_data",
            "invalid_input",
            "month 必须符合 YYYY-MM 格式",
        )

    record = user_usage_store.get_record(cleaned_user_id, cleaned_month)
    if record is None:
        return error_result(
            "fetch_external_data",
            "record_not_found",
            f"未查询到用户 {cleaned_user_id} 在 {cleaned_month} 的使用记录",
            details={"user_id": cleaned_user_id, "month": cleaned_month},
        )

    return ok_result(
        "fetch_external_data",
        {
            "record": record,
        },
        meta={
            "data_source": "sqlite",
            "table": "user_usage_records",
        },
    )


def get_web_search_trace() -> dict[str, Any]:
    if _web_service is None:
        return {}
    return dict(getattr(_web_service, "last_trace", {}) or {})


def get_user_report_schema_summary() -> dict[str, Any]:
    return user_usage_store.get_schema_summary()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _get_rag_service():
    global _rag_service
    if _rag_service is None:
        from rag.rag_service import RagSummarizeservice

        _rag_service = RagSummarizeservice()
    return _rag_service


def _get_web_service():
    global _web_service
    if _web_service is None:
        from websearch.web_search_service import WebSearchService

        _web_service = WebSearchService()
    return _web_service
