import os
import requests
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeservice
from websearch.web_search_service import WebSearchService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

rag = RagSummarizeservice()
web = WebSearchService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}

@tool(description="从互联网抓取相关内容并基于 LLM 生成回答")
def web_search(query: str) -> str:
    return web.search_summarize(query)


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    AMAP_KEY = agent_conf.get("amap_key")
    if not AMAP_KEY:
        return f"未配置高德地图Key，无法获取天气。默认返回虚拟天气: 城市{city}天气为晴天，气温26摄氏度"

    try:
        url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={city}&key={AMAP_KEY}&extensions=base"
        resp = requests.get(url, timeout=5).json()
        if resp.get("status") != "1":
            logger.warning(f"[get_weather] 高德API返回失败: {resp}")
            return f"城市{city}天气信息获取失败，返回虚拟天气: 晴天，气温26摄氏度"

        live = resp.get("lives", [{}])[0]
        if not live:
            return f"城市{city}天气信息为空，返回虚拟天气: 晴天，气温26摄氏度"

        weather = live.get("weather", "晴")
        temp = live.get("temperature", "26")
        humidity = live.get("humidity", "50")
        winddirection = live.get("winddirection", "南")
        windpower = live.get("windpower", "1")
        aqi = live.get("aqi", "21")

        return (f"城市{city}天气为{weather}，气温{temp}摄氏度，"
                f"空气湿度{humidity}%，风向{winddirection}，风力{windpower}级，AQI{aqi}")

    except Exception as e:
        logger.error(f"[get_weather] 获取城市{city}天气失败: {e}")
        return f"城市{city}天气信息获取异常，返回虚拟天气: 晴天，气温26摄氏度"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州", "保定"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"[generate_external_data]指定的外部数据文件不存在: {external_data_path}")
        
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id : str = arr[0].replace('"', "")
                featture : str = arr[1].replace('"', "")
                efficiency : str = arr[2].replace('"', "")
                consumables : str = arr[3].replace('"', "")
                comparison : str = arr[4].replace('"', "")
                time : str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": featture,
                    "效率": efficiency,
                    "消耗品": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[get_external_data]未检索到用户{user_id}在{month}的记录")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"

if __name__ == "__main__":
    print(fetch_external_data("1001", "2025-01"))