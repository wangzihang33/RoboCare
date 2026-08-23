from langchain_core.tools import tool

from agent.tools.core_tools import (
    fetch_external_data_core,
    get_weather_core,
    rag_summarize_core,
    web_search_core,
)
from agent.tools.tool_result import result_to_text


class ToolExecutionError(RuntimeError):
    """Preserve structured core-tool failures for lifecycle reliability policies."""

    def __init__(self, tool_name: str, error_type: str, message: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.error_type = error_type
        self.retryable = error_type in {
            "request_failed",
            "provider_error",
            "service_unavailable",
            "timeout",
        }


def _render_core_result(result: dict) -> str:
    if not result.get("ok"):
        error = result.get("error") or {}
        raise ToolExecutionError(
            str(result.get("tool_name") or "unknown_tool"),
            str(error.get("type") or "tool_error"),
            str(error.get("message") or "工具调用失败"),
        )
    return result_to_text(result)


@tool(description="从互联网抓取相关内容并基于 LLM 生成回答")
def web_search(query: str) -> str:
    return _render_core_result(web_search_core(query))


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return _render_core_result(rag_summarize_core(query))


@tool(description="获取指定城市的实时天气。必须显式传入 city 参数；缺少城市时应先向用户追问")
def get_weather(city: str) -> str:
    return _render_core_result(get_weather_core(city))


@tool(description="从用户设备数据服务中获取指定用户在指定月份的使用记录。必须传入 user_id 和 month")
def fetch_external_data(user_id: str, month: str) -> str:
    return _render_core_result(fetch_external_data_core(user_id, month))


@tool(description="无入参，调用后触发中间件为报告生成场景动态切换提示词；该工具仅供 Agent 内部流程使用")
def fill_context_for_report() -> str:
    return "fill_context_for_report已调用"


if __name__ == "__main__":
    print(result_to_text(fetch_external_data_core("1001", "2025-01")))
