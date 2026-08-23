from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import ModelRequest, before_model, dynamic_prompt, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.hooks.lifecycle import ToolHookManager
from agent.routing import RouteDecision, apply_route_guidance
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


tool_hook_manager = ToolHookManager()


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})
    tool_call_id = request.tool_call.get("id")
    hook_context = tool_hook_manager.before_tool_call(
        tool_name=tool_name,
        args=tool_args,
        tool_call_id=tool_call_id,
        runtime_context=request.runtime.context,
    )

    logger.info(f"[tool monitor] start tool={tool_name}, args={tool_args}")

    if not hook_context.decision.allowed:
        blocked_message = tool_hook_manager.blocked_message(hook_context.decision)
        logger.warning(f"[tool hook] blocked tool={tool_name}: {hook_context.decision.reason}")
        return ToolMessage(
            content=blocked_message,
            tool_call_id=hook_context.tool_call_id,
            name=tool_name,
            status="error",
        )

    try:
        result = tool_hook_manager.execute_tool(
            hook_context,
            handler,
            request=request,
        )
        tool_hook_manager.after_tool_call(hook_context, result)
        tool_hook_manager.record_tool_trace(hook_context, _extract_tool_trace(tool_name))
        logger.info(f"[tool monitor] success tool={tool_name}")

        if tool_name == "fill_context_for_report":
            request.runtime.context["is_report"] = True

        return result
    except Exception as e:
        fallback_message = tool_hook_manager.on_tool_error(hook_context, e)
        logger.error(f"[tool monitor] failed tool={tool_name}: {str(e)}")
        return ToolMessage(
            content=fallback_message,
            tool_call_id=hook_context.tool_call_id,
            name=tool_name,
            status="error",
        )


def _extract_tool_trace(tool_name: str) -> dict:
    if tool_name != "web_search":
        return {}

    try:
        from agent.tools.core_tools import get_web_search_trace

        return get_web_search_trace()
    except Exception as exc:
        logger.warning(f"[tool hook] failed to extract web_search trace: {exc}")
        return {}


@before_model
def log_before_model(
    state: AgentState,
    runtime: Runtime,
):
    logger.info(f"[before model] calling model with {len(state['messages'])} messages")
    logger.debug(
        "[before model] last message: "
        f"{type(state['messages'][-1]).__name__} | {state['messages'][-1].content}"
    )

    return None


@dynamic_prompt
def report_prompt_switch(Request: ModelRequest):
    is_report = Request.runtime.context.get("is_report", False)
    if is_report:
        base_prompt = load_report_prompts()
    else:
        tool_names = Request.runtime.context.get("tool_names")
        base_prompt = load_system_prompts(tool_names)
    route_decision = Request.runtime.context.get("route_decision")
    if isinstance(route_decision, RouteDecision):
        return apply_route_guidance(base_prompt, route_decision)
    return base_prompt
