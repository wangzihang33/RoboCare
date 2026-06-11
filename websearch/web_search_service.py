from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts
from websearch.cache import WebSearchCache
from websearch.fetch_web_content import WebContentFetcher
from websearch.retrieval import EmbeddingRetriever


def print_prompt(full_prompt):
    print("=" * 20, full_prompt.to_string(), "=" * 20)
    return full_prompt


class WebSearchService:
    """
    WebSearch orchestration service:
    query -> Serper -> filtered sources -> fetched pages -> page-level retrieval -> LLM answer.
    """

    def __init__(self):
        self.fetcher = WebContentFetcher("")
        self.retriever_class = EmbeddingRetriever()
        self.cache = WebSearchCache()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
        self.last_trace: dict = {}
        logger.info("[WebSearchService] initialized")

    def _init_chain(self):
        return self.prompt_template | print_prompt | self.model | StrOutputParser()

    def retriever_docs(self, query: str) -> list[Document]:
        docs, _ = self.retriever_docs_with_trace(query)
        return docs

    def retriever_docs_with_trace(self, query: str) -> tuple[list[Document], dict]:
        self.fetcher.query = query
        pages, serper_response = self.fetcher.fetch_pages()
        if not serper_response:
            logger.warning(f"[WebSearchService] no search results for query: {query}")
            return [], {
                "cache_hit": False,
                "error_reason": "empty_search_results",
                "source_count": 0,
            }

        relevant_docs = self.retriever_class.retrieve_page_embeddings(pages, query)
        sources = self._extract_sources(relevant_docs)
        trace = {
            "cache_hit": False,
            "original_count": serper_response.get("original_count", 0),
            "urls_after_filter": serper_response.get("count", 0),
            "filter_stats": serper_response.get("filter_stats", {}),
            "fetch_stats": serper_response.get("fetch_stats", {}),
            "retrieved_docs": len(relevant_docs),
            "source_count": len(sources),
        }
        logger.info(f"[WebSearchService] retrieved {len(relevant_docs)} web docs")
        return relevant_docs, trace

    def search_summarize(self, query: str) -> str:
        cached = self.cache.get(query, top_k=self.retriever_class.TOP_K)
        if cached:
            trace = {
                "cache_hit": True,
                "cache_key": cached.get("cache_key"),
                "source_count": len(cached.get("sources", [])),
                "cached_created_at": cached.get("created_at"),
                "cached_expires_at": cached.get("expires_at"),
            }
            self.last_trace = trace
            return cached.get("answer", "")

        context_docs, trace = self.retriever_docs_with_trace(query)
        sources = self._extract_sources(context_docs)
        context = self._format_context(context_docs)
        raw_answer = self.chain.invoke({"input": query, "context": context})
        answer = self._append_references(raw_answer, sources)
        cache_payload = self.cache.set(
            query,
            top_k=self.retriever_class.TOP_K,
            answer=answer,
            sources=sources,
            trace=trace,
        )
        trace["cache_key"] = cache_payload.get("cache_key")
        self.last_trace = trace
        return answer

    @staticmethod
    def _format_context(context_docs: list[Document]) -> str:
        context_parts: list[str] = []
        for idx, doc in enumerate(context_docs, 1):
            metadata = doc.metadata or {}
            source_id = metadata.get("source_id") or f"R{idx}"
            context_parts.append(
                "\n".join(
                    [
                        f"[{source_id}]",
                        f"Title: {metadata.get('title', '')}",
                        f"URL: {metadata.get('url', '')}",
                        f"Domain: {metadata.get('domain', '')}",
                        f"Search Rank: {metadata.get('search_rank', '')}",
                        f"Fetched At: {metadata.get('fetched_at', '')}",
                        f"Snippet: {metadata.get('snippet', '')}",
                        f"Content: {doc.page_content}",
                    ]
                )
            )
        return "\n\n".join(context_parts)

    @staticmethod
    def _extract_sources(context_docs: list[Document]) -> list[dict]:
        sources: list[dict] = []
        seen_urls: set[str] = set()
        for doc in context_docs:
            metadata = doc.metadata or {}
            url = metadata.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append(
                {
                    "source_id": metadata.get("source_id", ""),
                    "title": metadata.get("title", ""),
                    "url": url,
                    "domain": metadata.get("domain", ""),
                    "snippet": metadata.get("snippet", ""),
                    "search_rank": metadata.get("search_rank", ""),
                    "fetched_at": metadata.get("fetched_at", ""),
                }
            )
        return sources

    @staticmethod
    def _append_references(answer: str, sources: list[dict]) -> str:
        if not sources:
            return answer

        reference_lines = ["", "", "参考来源："]
        for index, source in enumerate(sources, 1):
            source_id = source.get("source_id") or f"R{index}"
            title = source.get("title") or source.get("domain") or "Untitled"
            url = source.get("url", "")
            fetched_at = source.get("fetched_at", "")
            fetched_text = f" | fetched_at={fetched_at}" if fetched_at else ""
            reference_lines.append(f"[{source_id}] {title} - {url}{fetched_text}")

        return answer.rstrip() + "\n".join(reference_lines)


if __name__ == "__main__":
    websearch = WebSearchService()
    print(websearch.search_summarize("robot vacuum buying guide"))
