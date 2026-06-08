from typing import Callable
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts
from langgraph.runtime import Runtime


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    logger.info(f"工具执行：{request.tool_call['name']}")
    logger.info(f"工具参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["is_report"] = True

        
        return result
    except Exception as e:
        logger.error(f"[tool monitor]工具{request.tool_call['name']}调用失败：{str(e)}")
        raise e


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