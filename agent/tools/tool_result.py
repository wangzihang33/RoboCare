from __future__ import annotations

import json
from typing import Any
from uuid import uuid4


def make_trace_id(tool_name: str) -> str:
    return f"{tool_name}_{uuid4().hex[:12]}"


def ok_result(
    tool_name: str,
    data: dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "tool_name": tool_name,
        "trace_id": trace_id or make_trace_id(tool_name),
        "data": data,
        "error": None,
        "meta": meta or {},
    }


def error_result(
    tool_name: str,
    error_type: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "tool_name": tool_name,
        "trace_id": trace_id or make_trace_id(tool_name),
        "data": None,
        "error": {
            "type": error_type,
            "message": message,
            "details": details or {},
        },
        "meta": {},
    }


def result_to_text(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        error = result.get("error") or {}
        message = error.get("message", "工具调用失败")
        error_type = error.get("type", "tool_error")
        return f"工具调用失败：{message}（错误类型：{error_type}）"

    data = result.get("data") or {}
    if isinstance(data, dict):
        if "answer" in data:
            return str(data["answer"])
        if "weather" in data:
            return format_weather(data)
        if "record" in data:
            return format_user_usage_record(data["record"])

    return json.dumps(data, ensure_ascii=False, indent=2)


def format_weather(data: dict[str, Any]) -> str:
    parts = [
        f"城市{data.get('city', '')}天气为{data.get('weather', '')}",
        f"气温{data.get('temperature', '')}摄氏度",
        f"空气湿度{data.get('humidity', '')}%",
    ]
    winddirection = data.get("winddirection")
    windpower = data.get("windpower")
    if winddirection:
        parts.append(f"风向{winddirection}")
    if windpower:
        parts.append(f"风力{windpower}级")
    reporttime = data.get("reporttime")
    if reporttime:
        parts.append(f"发布时间{reporttime}")
    return "，".join(part for part in parts if part)


def format_user_usage_record(record: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"用户ID：{record.get('user_id', '')}",
            f"月份：{record.get('month', '')}",
            f"家庭/设备特征：{record.get('feature_profile', '')}",
            f"清洁效率：{record.get('cleaning_efficiency', '')}",
            f"耗材状态：{record.get('consumables_status', '')}",
            f"同类对比：{record.get('comparison_summary', '')}",
        ]
    )
