from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.middleware import log_before_model, monitor_tool, report_prompt_switch
from agent.tools.agent_tools import (
    fetch_external_data,
    fill_context_for_report,
    get_weather,
    rag_summarize,
    web_search,
)



class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_weather,
                fetch_external_data,
                fill_context_for_report,
                web_search,
            ],
            middleware=[log_before_model, monitor_tool, report_prompt_switch]
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
        res = self.agent.stream(input_dict, stream_mode="values", context={"report": False})
        for chunk in res:
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("我这个地区适合咱们智扫通品牌的扫地机器人吗？有什么优势？有什么劣势？"):
        print(chunk, end="", flush=True)
