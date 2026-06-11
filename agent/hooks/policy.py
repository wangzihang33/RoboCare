from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.hooks.sanitizer import detect_sensitive_types


@dataclass(frozen=True)
class ToolPolicy:
    tool_type: str
    risk_level: str
    requires_network: bool = False
    high_cost: bool = False
    accesses_user_data: bool = False
    allow_sensitive_input: bool = False


@dataclass(frozen=True)
class HookDecision:
    allowed: bool
    reason: str
    risk_level: str
    requires_network: bool
    high_cost: bool
    accesses_user_data: bool
    sensitive_types: list[str]
    fallback_message: str = ""


TOOL_POLICIES: dict[str, ToolPolicy] = {
    "rag_summarize": ToolPolicy(tool_type="local_rag", risk_level="low", high_cost=True),
    "web_search": ToolPolicy(tool_type="web_search", risk_level="medium", requires_network=True, high_cost=True),
    "get_weather": ToolPolicy(tool_type="weather", risk_level="low", requires_network=True),
    "get_user_location": ToolPolicy(tool_type="user_context", risk_level="low"),
    "get_user_id": ToolPolicy(tool_type="user_context", risk_level="low"),
    "get_current_month": ToolPolicy(tool_type="system_context", risk_level="low"),
    "fetch_external_data": ToolPolicy(
        tool_type="user_data",
        risk_level="high",
        accesses_user_data=True,
        allow_sensitive_input=False,
    ),
    "fill_context_for_report": ToolPolicy(tool_type="report_context", risk_level="low"),
}


def get_tool_policy(tool_name: str) -> ToolPolicy:
    return TOOL_POLICIES.get(tool_name, ToolPolicy(tool_type="unknown", risk_level="medium"))


def evaluate_tool_call(tool_name: str, args: Any) -> HookDecision:
    policy = get_tool_policy(tool_name)
    sensitive_types = detect_sensitive_types(args)

    if sensitive_types and not policy.allow_sensitive_input:
        return HookDecision(
            allowed=False,
            reason="工具入参包含敏感信息或敏感字段请求",
            risk_level=policy.risk_level,
            requires_network=policy.requires_network,
            high_cost=policy.high_cost,
            accesses_user_data=policy.accesses_user_data,
            sensitive_types=sensitive_types,
            fallback_message="该请求涉及敏感个人信息或凭证信息，当前工具无权限处理。",
        )

    return HookDecision(
        allowed=True,
        reason="allowed",
        risk_level=policy.risk_level,
        requires_network=policy.requires_network,
        high_cost=policy.high_cost,
        accesses_user_data=policy.accesses_user_data,
        sensitive_types=sensitive_types,
    )
