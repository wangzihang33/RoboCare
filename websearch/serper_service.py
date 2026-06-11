import requests
import re
import json
from utils.config_handler import websearch_conf
from utils.logger_handler import logger
from websearch.models import SearchResult
from websearch.source_filter import filter_search_results

class SerperClient:
    """
    Serper 搜索客户端，支持中文/英文查询，并解析返回的标题、链接和摘要
    """
    def __init__(self):
        # 从统一配置中读取 API Key
        self.url = websearch_conf.get("serper_api_url", None)
        if not self.url:
            raise ValueError("[SerperClient] 配置文件中缺少 serper_api_url")

        api_key = websearch_conf.get("serper_api_key", None)
        if not api_key:
            raise ValueError("[SerperClient] 配置文件中缺少 serper_api_key")

        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        # 可配置最大页码、超时等
        self.page = websearch_conf.get("serper_page", 2)
        self.timeout = websearch_conf.get("timeout", 8)

    def serper(self, query: str) -> dict:
        """
        调用 Serper API 执行搜索
        """
        serper_settings = {"q": query, "page": self.page}

        # 如果 query 中包含中文，修改请求参数
        if self._contains_chinese(query):
            serper_settings.update({"gl": "cn", "hl": "zh-cn"})

        payload = json.dumps(serper_settings)

        try:
            response = requests.post(self.url, headers=self.headers, data=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"[SerperClient] 搜索请求失败: {str(e)}")
            return {}

    @staticmethod
    def _contains_chinese(query: str) -> bool:
        """
        判断字符串是否包含中文
        """
        pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(pattern.search(query))

    @staticmethod
    def extract_components(serper_response: dict) -> dict:
        """
        从 Serper API 响应中提取标题、链接、摘要等信息
        """
        raw_results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                rank=index,
            )
            for index, item in enumerate(serper_response.get("organic", []), 1)
        ]
        filtered_results, filter_stats = filter_search_results(raw_results)

        query = serper_response.get("searchParameters", {}).get("q", "")
        count = len(filtered_results)
        language = "zh-cn" if SerperClient._contains_chinese(query) else "en-us"

        output_dict = {
            "query": query,
            "language": language,
            "count": count,
            "original_count": len(raw_results),
            "filter_stats": filter_stats.to_dict(),
            "results": [result.to_dict() for result in filtered_results],
            "titles": [result.title for result in filtered_results],
            "links": [result.url for result in filtered_results],
            "snippets": [result.snippet for result in filtered_results],
        }
        logger.info(
            "[SerperClient] filtered search results: "
            f"{filter_stats.kept_count}/{filter_stats.original_count}"
        )
        return output_dict


# 测试用例
if __name__ == "__main__":
    client = SerperClient()
    query = "目前市面上最好的智能扫地机器人是什么品牌？有什么优势？"
    response = client.serper(query)
    components = client.extract_components(response)
    print(json.dumps(components, ensure_ascii=False, indent=2))
