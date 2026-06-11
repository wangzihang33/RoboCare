from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from agent.hooks.policy import HookDecision, evaluate_tool_call, get_tool_policy
from agent.hooks.recorder import write_hook_event
from agent.hooks.sanitizer import summarize_value


@dataclass
class ToolCallContext:
    session_id: str
    tool_call_id: str
    tool_name: str
    args: Any
    started_at: float
    decision: HookDecision


class ToolHookManager:
    """Lifecycle hooks for LangChain tool calls."""

    def before_tool_call(
        self,
        *,
        tool_name: str,
        args: Any,
        tool_call_id: str | None = None,
        runtime_context: dict | None = None,
    ) -> ToolCallContext:
        session_id = self._ensure_session_id(runtime_context)
        normalized_tool_call_id = tool_call_id or f"tool_{uuid.uuid4().hex[:12]}"
        decision = evaluate_tool_call(tool_name, args)
        policy = get_tool_policy(tool_name)
        context = ToolCallContext(
            session_id=session_id,
            tool_call_id=normalized_tool_call_id,
            tool_name=tool_name,
            args=args,
            started_at=time.perf_counter(),
            decision=decision,
        )

        write_hook_event(
            {
                "stage": "before_tool_call",
                "session_id": session_id,
                "tool_call_id": normalized_tool_call_id,
                "tool_name": tool_name,
                "tool_type": policy.tool_type,
                "status": "allowed" if decision.allowed else "blocked",
                "risk_level": decision.risk_level,
                "requires_network": decision.requires_network,
                "high_cost": decision.high_cost,
                "accesses_user_data": decision.accesses_user_data,
                "sensitive_types": decision.sensitive_types,
                "decision_reason": decision.reason,
                "input_summary": summarize_value(args),
            }
        )
        return context

    def after_tool_call(self, context: ToolCallContext, output: Any) -> None:
        write_hook_event(
            {
                "stage": "after_tool_call",
                "session_id": context.session_id,
                "tool_call_id": context.tool_call_id,
                "tool_name": context.tool_name,
                "status": "success",
                "latency_ms": self._latency_ms(context),
                "input_summary": summarize_value(context.args),
                "output_summary": summarize_value(self._extract_output(output)),
            }
        )

    def on_tool_error(self, context: ToolCallContext, error: Exception) -> str:
        message = self.safe_error_message(context.tool_name)
        write_hook_event(
            {
                "stage": "on_tool_error",
                "session_id": context.session_id,
                "tool_call_id": context.tool_call_id,
                "tool_name": context.tool_name,
                "status": "error",
                "latency_ms": self._latency_ms(context),
                "error_type": type(error).__name__,
                "error_message": summarize_value(str(error)),
                "fallback_message": message,
            }
        )
        return message

    @staticmethod
    def safe_error_message(tool_name: str) -> str:
        return f"工具 {tool_name} 调用失败，已记录审计日志。请稍后重试或换一种问法。"

    @staticmethod
    def blocked_message(decision: HookDecision) -> str:
        return decision.fallback_message or "该工具调用被安全策略拦截。"

    @staticmethod
    def _extract_output(output: Any) -> Any:
        content = getattr(output, "content", None)
        return content if content is not None else output

    @staticmethod
    def _latency_ms(context: ToolCallContext) -> float:
        return round((time.perf_counter() - context.started_at) * 1000, 3)

    @staticmethod
    def _ensure_session_id(runtime_context: dict | None) -> str:
        if runtime_context is not None:
            session_id = runtime_context.get("session_id")
            if not session_id:
                session_id = f"sess_{uuid.uuid4().hex[:12]}"
                runtime_context["session_id"] = session_id
            return str(session_id)
        return f"sess_{uuid.uuid4().hex[:12]}"
