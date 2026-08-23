from __future__ import annotations

import time

import pytest

from agent.hooks.lifecycle import (
    CircuitOpenError,
    ToolHookManager,
    ToolOutputError,
)
from agent.hooks.policy import TOOL_POLICIES, ToolPolicy


def _manager(monkeypatch, policy: ToolPolicy) -> tuple[ToolHookManager, list[dict]]:
    events: list[dict] = []
    monkeypatch.setattr("agent.hooks.lifecycle.write_hook_event", events.append)
    policies = dict(TOOL_POLICIES)
    policies["test_tool"] = policy
    return ToolHookManager(policies=policies), events


def _context(manager: ToolHookManager):
    return manager.before_tool_call(tool_name="test_tool", args={})


def test_retries_transient_failure_and_records_retry_event(monkeypatch):
    manager, events = _manager(
        monkeypatch,
        ToolPolicy(
            tool_type="test",
            risk_level="low",
            max_retries=2,
            retry_backoff_seconds=0,
            retryable_errors=(ConnectionError,),
        ),
    )
    attempts = 0

    def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary")
        return {"ok": True, "data": "done"}

    result = manager.execute_tool(_context(manager), handler, request=None)

    assert result["ok"] is True
    assert attempts == 3
    assert [event["stage"] for event in events].count("tool_retry") == 2


def test_non_retryable_failure_uses_safe_fallback_without_retry(monkeypatch):
    manager, events = _manager(
        monkeypatch,
        ToolPolicy(
            tool_type="test",
            risk_level="low",
            max_retries=2,
            retry_backoff_seconds=0,
            retryable_errors=(ConnectionError,),
            fallback_message="测试工具暂时不可用。",
        ),
    )
    attempts = 0

    def handler(_request):
        nonlocal attempts
        attempts += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        manager.execute_tool(_context(manager), handler, request=None)

    assert attempts == 1
    assert events[-1]["stage"] == "tool_failure"
    assert events[-1]["error_class"] == "non_retryable"


def test_empty_tool_result_is_rejected(monkeypatch):
    manager, _events = _manager(
        monkeypatch,
        ToolPolicy(tool_type="test", risk_level="low", validate_result=True),
    )

    with pytest.raises(ToolOutputError):
        manager.execute_tool(_context(manager), lambda _request: {"ok": True}, request=None)


def test_repeated_failures_open_circuit(monkeypatch):
    manager, events = _manager(
        monkeypatch,
        ToolPolicy(
            tool_type="test",
            risk_level="medium",
            max_retries=0,
            circuit_failure_threshold=2,
            circuit_recovery_seconds=60,
        ),
    )

    def handler(_request):
        raise RuntimeError("provider down")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            manager.execute_tool(_context(manager), handler, request=None)

    with pytest.raises(CircuitOpenError):
        manager.execute_tool(_context(manager), handler, request=None)

    assert any(event["stage"] == "circuit_open" for event in events)


def test_timeout_is_classified_as_retryable_when_configured(monkeypatch):
    manager, events = _manager(
        monkeypatch,
        ToolPolicy(
            tool_type="test",
            risk_level="low",
            timeout_seconds=0.01,
            max_retries=0,
            retryable_errors=(TimeoutError,),
        ),
    )

    def handler(_request):
        time.sleep(0.05)
        return {"ok": True, "data": "late"}

    with pytest.raises(TimeoutError):
        manager.execute_tool(_context(manager), handler, request=None)

    assert events[-1]["error_class"] == "retryable_exhausted"


def test_tool_error_marker_can_be_retried_without_broad_exception_matching(monkeypatch):
    manager, events = _manager(
        monkeypatch,
        ToolPolicy(
            tool_type="test",
            risk_level="low",
            max_retries=1,
            retry_backoff_seconds=0,
        ),
    )
    attempts = 0

    class MarkedToolError(RuntimeError):
        retryable = True

    def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise MarkedToolError("provider error")
        return "recovered"

    result = manager.execute_tool(_context(manager), handler, request=None)

    assert result == "recovered"
    assert attempts == 2
    assert events[-1]["stage"] == "tool_retry"
