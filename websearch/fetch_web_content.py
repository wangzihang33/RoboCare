from __future__ import annotations

import threading
import time

from utils.logger_handler import logger
from websearch.models import FetchedPage, SearchResult
from websearch.serper_service import SerperClient
from websearch.web_crawler import WebScraper


class WebContentFetcher:
    def __init__(self, query: str):
        self.query = query
        self.web_pages: list[FetchedPage] = []
        self.error_urls: list[str] = []
        self.web_pages_lock = threading.Lock()
        self.error_urls_lock = threading.Lock()
        logger.info(f"[WebContentFetcher] initialized, query={self.query}")

    def _web_crawler_thread(self, thread_id: int, results: list[SearchResult]) -> None:
        result = results[thread_id]
        url = result.url
        start_time = time.time()
        logger.info(f"[Thread-{thread_id}] start crawling URL: {url}")

        try:
            scraper = WebScraper()
            content = scraper.scrape_url(url, 0)

            if 0 < len(content) < 800:
                logger.info(f"[Thread-{thread_id}] content too short, retrying with expanded rule")
                content = scraper.scrape_url(url, 1)

            if len(content) > 300:
                page = FetchedPage.from_search_result(result, content=content, status="success")
                logger.info(f"[Thread-{thread_id}] crawled successfully, content_length={len(content)}")
            else:
                page = FetchedPage.from_search_result(
                    result,
                    content="",
                    status="empty",
                    error_message="content_too_short",
                )
                logger.warning(f"[Thread-{thread_id}] crawled content too short, URL: {url}")

            with self.web_pages_lock:
                self.web_pages.append(page)

            elapsed = time.time() - start_time
            logger.info(f"[Thread-{thread_id}] finished URL: {url}, latency={elapsed:.2f}s")

        except Exception as e:
            with self.error_urls_lock:
                self.error_urls.append(url)
            with self.web_pages_lock:
                self.web_pages.append(
                    FetchedPage.from_search_result(
                        result,
                        content="",
                        status="error",
                        error_message=str(e),
                    )
                )
            logger.error(f"[Thread-{thread_id}] failed to crawl URL: {url}, error={str(e)}")

    def _serper_launcher(self) -> dict:
        serper_client = SerperClient()
        serper_results = serper_client.serper(self.query)
        if not serper_results:
            logger.warning(f"[WebContentFetcher] Serper returned empty results: {self.query}")
            return {}
        return serper_client.extract_components(serper_results)

    def _crawl_threads_launcher(self, results: list[SearchResult]) -> None:
        threads = []
        for i in range(len(results)):
            thread = threading.Thread(target=self._web_crawler_thread, args=(i, results))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def fetch_pages(self) -> tuple[list[FetchedPage], dict | None]:
        logger.info(f"[WebContentFetcher] start query: {self.query}")
        self.web_pages = []
        self.error_urls = []

        serper_response = self._serper_launcher()
        if not serper_response:
            logger.warning(f"[WebContentFetcher] query produced no search results: {self.query}")
            return [], None

        results = [SearchResult(**item) for item in serper_response.get("results", [])]
        logger.info(f"[WebContentFetcher] Serper returned {len(results)} filtered links")
        self._crawl_threads_launcher(results)

        page_by_url = {page.url: page for page in self.web_pages}
        ordered_pages = [
            page_by_url.get(
                result.url,
                FetchedPage.from_search_result(
                    result,
                    content="",
                    status="error",
                    error_message="not_fetched",
                ),
            )
            for result in results
        ]
        serper_response["fetch_stats"] = self._build_fetch_stats(ordered_pages)
        logger.info(
            "[WebContentFetcher] fetch completed, "
            f"success={serper_response['fetch_stats']['success_count']}, "
            f"errors={serper_response['fetch_stats']['error_count']}"
        )
        return ordered_pages, serper_response

    def fetch(self) -> tuple[list[str], dict | None]:
        pages, serper_response = self.fetch_pages()
        return [page.content for page in pages], serper_response

    @staticmethod
    def _build_fetch_stats(pages: list[FetchedPage]) -> dict:
        return {
            "page_count": len(pages),
            "success_count": sum(1 for page in pages if page.status == "success"),
            "empty_count": sum(1 for page in pages if page.status == "empty"),
            "error_count": sum(1 for page in pages if page.status == "error"),
            "content_chars": sum(page.content_length for page in pages),
        }


if __name__ == "__main__":
    fetcher = WebContentFetcher("What happened to Silicon Valley Bank")
    pages, serper_response = fetcher.fetch_pages()

    logger.info(f"Serper Response: {serper_response}")
    logger.info(f"Fetched {len(pages)} pages")
    for i, page in enumerate(pages[:3], 1):
        logger.info(f"\n--- Page {i} {page.url} ---\n{page.content[:500]}...")
