from typing import Callable
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from agent.hooks.lifecycle import ToolHookManager
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts
from langgraph.runtime import Runtime

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

    logger.info(f"工具执行：{tool_name}")
    logger.info(f"工具参数：{tool_args}")

    if not hook_context.decision.allowed:
        blocked_message = tool_hook_manager.blocked_message(hook_context.decision)
        logger.warning(f"[tool hook]工具{tool_name}调用被拦截：{hook_context.decision.reason}")
        return ToolMessage(
            content=blocked_message,
            tool_call_id=hook_context.tool_call_id,
            name=tool_name,
            status="error",
        )

    try:
        result = handler(request)
        tool_hook_manager.after_tool_call(hook_context, result)
        logger.info(f"[tool monitor]工具{tool_name}调用成功")

        if tool_name == "fill_context_for_report":
            request.runtime.context["is_report"] = True

        return result
    except Exception as e:
        fallback_message = tool_hook_manager.on_tool_error(hook_context, e)
        logger.error(f"[tool monitor]工具{tool_name}调用失败：{str(e)}")
        return ToolMessage(
            content=fallback_message,
            tool_call_id=hook_context.tool_call_id,
            name=tool_name,
            status="error",
        )


@before_model
def log_before_model(
    state: AgentState,
    runtime: Runtime,
):
    logger.info(f"[before model]模型即将调用，并附带{len(state['messages'])}消息")
    logger.debug(f"[before model]消息内容：{type(state['messages'][-1]).__name__} | {state['messages'][-1].content}")

    return None


@dynamic_prompt
def report_prompt_switch(Request: ModelRequest):
    is_report = Request.runtime.context.get("is_report", False)
    if is_report:
        return load_report_prompts()
    return load_system_prompts()
