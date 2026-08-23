from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from agent.hooks.policy import (
    TOOL_POLICIES,
    HookDecision,
    ToolPolicy,
    evaluate_tool_call,
    get_tool_policy,
)
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


class ToolOutputError(RuntimeError):
    """Raised when a tool returns an empty or invalid structured result."""


class CircuitOpenError(RuntimeError):
    """Raised when repeated transient failures temporarily open a tool circuit."""


class ToolHookManager:
    """Lifecycle hooks for LangChain tool calls."""

    def __init__(
        self,
        *,
        policies: dict[str, ToolPolicy] | None = None,
        sleeper=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.policies = policies or TOOL_POLICIES
        self._sleeper = sleeper
        self._clock = clock
        self._circuit_failures: dict[str, int] = {}
        self._circuit_opened_at: dict[str, float] = {}

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
        policy = self.policies.get(tool_name, get_tool_policy(tool_name))
        decision = evaluate_tool_call(tool_name, args, policy=policy)
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

    def execute_tool(
        self,
        context: ToolCallContext,
        handler,
        *,
        request: Any,
    ) -> Any:
        """Execute a tool with bounded retries, timeout, validation and a circuit breaker."""
        policy = self.policies.get(context.tool_name, get_tool_policy(context.tool_name))
        self._check_circuit(context, policy)
        attempts = max(0, int(policy.max_retries)) + 1

        for attempt in range(1, attempts + 1):
            try:
                result = self._invoke_with_timeout(handler, request, policy.timeout_seconds)
                self._validate_result(result, policy)
                self._circuit_failures.pop(context.tool_name, None)
                self._circuit_opened_at.pop(context.tool_name, None)
                return result
            except Exception as exc:
                retryable = self._is_retryable(exc, policy)
                if retryable and attempt < attempts:
                    write_hook_event(
                        {
                            "stage": "tool_retry",
                            "session_id": context.session_id,
                            "tool_call_id": context.tool_call_id,
                            "tool_name": context.tool_name,
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "error_type": type(exc).__name__,
                        }
                    )
                    if policy.retry_backoff_seconds > 0:
                        self._sleeper(policy.retry_backoff_seconds * attempt)
                    continue

                if self._counts_toward_circuit(exc, policy):
                    self._record_circuit_failure(context, policy)
                write_hook_event(
                    {
                        "stage": "tool_failure",
                        "session_id": context.session_id,
                        "tool_call_id": context.tool_call_id,
                        "tool_name": context.tool_name,
                        "attempts": attempt,
                        "error_type": type(exc).__name__,
                        "error_class": (
                            "retryable_exhausted" if retryable else "non_retryable"
                        ),
                    }
                )
                raise

        raise RuntimeError("tool execution exhausted without a result")

    @staticmethod
    def _invoke_with_timeout(handler, request: Any, timeout_seconds: float | None) -> Any:
        if timeout_seconds is None or timeout_seconds <= 0:
            return handler(request)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(handler, request)
        try:
            return future.result(timeout=timeout_seconds)
        finally:
            # Do not wait for a timed-out provider call before returning to the Agent.
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _validate_result(result: Any, policy: ToolPolicy) -> None:
        if not policy.validate_result:
            return
        if result is None or (isinstance(result, str) and not result.strip()):
            raise ToolOutputError("工具返回了空结果")
        if isinstance(result, dict) and "ok" in result:
            if result.get("ok") is not True:
                raise ToolOutputError("工具返回了失败状态")
            if "data" not in result or result.get("data") is None:
                raise ToolOutputError("工具成功结果缺少 data 字段")
        content = getattr(result, "content", None)
        if content is not None and isinstance(content, str) and not content.strip():
            raise ToolOutputError("工具消息内容为空")

    @staticmethod
    def _is_retryable(error: Exception, policy: ToolPolicy) -> bool:
        return bool(getattr(error, "retryable", False)) or isinstance(
            error, policy.retryable_errors
        )

    @staticmethod
    def _counts_toward_circuit(error: Exception, policy: ToolPolicy) -> bool:
        if isinstance(error, ToolOutputError):
            return False
        return ToolHookManager._is_retryable(error, policy) or isinstance(
            error, (ConnectionError, TimeoutError, RuntimeError)
        )

    def _record_circuit_failure(self, context: ToolCallContext, policy: ToolPolicy) -> None:
        tool_name = context.tool_name
        failures = self._circuit_failures.get(tool_name, 0) + 1
        self._circuit_failures[tool_name] = failures
        if failures >= max(1, policy.circuit_failure_threshold):
            self._circuit_opened_at[tool_name] = self._clock()
            write_hook_event(
                {
                    "stage": "circuit_open",
                    "session_id": context.session_id,
                    "tool_call_id": context.tool_call_id,
                    "tool_name": tool_name,
                    "failure_count": failures,
                    "recovery_seconds": policy.circuit_recovery_seconds,
                }
            )

    def _check_circuit(self, context: ToolCallContext, policy: ToolPolicy) -> None:
        opened_at = self._circuit_opened_at.get(context.tool_name)
        if opened_at is None:
            return
        elapsed = self._clock() - opened_at
        if elapsed < policy.circuit_recovery_seconds:
            write_hook_event(
                {
                    "stage": "circuit_blocked",
                    "session_id": context.session_id,
                    "tool_call_id": context.tool_call_id,
                    "tool_name": context.tool_name,
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
            raise CircuitOpenError(f"工具 {context.tool_name} 熔断中")
        self._circuit_opened_at.pop(context.tool_name, None)
        self._circuit_failures.pop(context.tool_name, None)
        write_hook_event(
            {
                "stage": "circuit_half_open",
                "session_id": context.session_id,
                "tool_call_id": context.tool_call_id,
                "tool_name": context.tool_name,
            }
        )

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

    def record_tool_trace(self, context: ToolCallContext, trace: dict[str, Any]) -> None:
        if not trace:
            return

        write_hook_event(
            {
                "stage": "websearch_trace",
                "session_id": context.session_id,
                "tool_call_id": context.tool_call_id,
                "tool_name": context.tool_name,
                **trace,
            }
        )

    def on_tool_error(self, context: ToolCallContext, error: Exception) -> str:
        policy = self.policies.get(context.tool_name, get_tool_policy(context.tool_name))
        message = policy.fallback_message or self.safe_error_message(context.tool_name)
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
                "recovery_action": "safe_fallback",
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
