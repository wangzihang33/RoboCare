# websearch/fetch_web_content.py
import threading
import time
from websearch.web_crawler import WebScraper
from websearch.serper_service import SerperClient
from utils.logger_handler import logger  # 使用项目统一日志

class WebContentFetcher:
    def __init__(self, query):
        self.query = query
        self.web_contents = []  # 存储抓取到的网页内容
        self.error_urls = []    # 存储抓取失败的 URL
        self.web_contents_lock = threading.Lock()
        self.error_urls_lock = threading.Lock()
        logger.info(f"[WebContentFetcher] 初始化抓取器，查询: {self.query}")

    def _web_crawler_thread(self, thread_id: int, urls: list):
        try:
            url = urls[thread_id]
            logger.info(f"[Thread-{thread_id}] 开始抓取 URL: {url}")
            start_time = time.time()

            scraper = WebScraper()
            content = scraper.scrape_url(url, 0)

            # 如果正文太短，尝试扩展规则
            if 0 < len(content) < 800:
                logger.info(f"[Thread-{thread_id}] 内容太短，尝试扩展抓取规则")
                content = scraper.scrape_url(url, 1)

            # 如果抓取的内容长度足够，加入列表
            if len(content) > 300:
                with self.web_contents_lock:
                    self.web_contents.append({"url": url, "content": content})
                logger.info(f"[Thread-{thread_id}] 成功抓取，正文长度: {len(content)}")

            end_time = time.time()
            logger.info(f"[Thread-{thread_id}] 完成抓取 URL: {url}，耗时: {end_time - start_time:.2f}s")

        except Exception as e:
            with self.error_urls_lock:
                self.error_urls.append(url)
            logger.error(f"[Thread-{thread_id}] 抓取失败 URL: {url}, 错误: {str(e)}")

    def _serper_launcher(self):
        serper_client = SerperClient()
        serper_results = serper_client.serper(self.query)
        if not serper_results:
            logger.warning(f"[WebContentFetcher] Serper 搜索返回空结果: {self.query}")
            return {}
        return serper_client.extract_components(serper_results)

    def _crawl_threads_launcher(self, url_list):
        threads = []
        for i in range(len(url_list)):
            thread = threading.Thread(target=self._web_crawler_thread, args=(i, url_list))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def fetch(self):
        logger.info(f"[WebContentFetcher] 开始执行查询: {self.query}")
        serper_response = self._serper_launcher()
        if serper_response:
            url_list = serper_response.get("links", [])
            logger.info(f"[WebContentFetcher] Serper 返回 {len(url_list)} 个链接")
            self._crawl_threads_launcher(url_list)

            # 按 URL 顺序返回抓取内容
            ordered_contents = [
                next((item['content'] for item in self.web_contents if item['url'] == url), '') 
                for url in url_list
            ]
            logger.info(f"[WebContentFetcher] 抓取完成，共抓取 {len(ordered_contents)} 条网页内容")
            return ordered_contents, serper_response

        logger.warning(f"[WebContentFetcher] 查询未获取到搜索结果: {self.query}")
        return [], None


# Example usage
if __name__ == "__main__":
    fetcher = WebContentFetcher("What happened to Silicon Valley Bank")
    contents, serper_response = fetcher.fetch()

    logger.info(f"Serper Response: {serper_response}")
    logger.info(f"抓取到 {len(contents)} 条网页内容")
    for i, c in enumerate(contents[:3], 1):
        logger.info(f"\n--- 网页 {i} ---\n{c[:500]}...")  # 打印前 500 字符