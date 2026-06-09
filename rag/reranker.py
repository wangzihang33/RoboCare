"""DashScope reranker integration for two-stage RAG retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from langchain_core.documents import Document

from utils.config_handler import rag_conf


DEFAULT_RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"


@dataclass
class RerankResult:
    index: int
    relevance_score: float


class DashScopeReranker:
    """Rerank retrieved documents with DashScope qwen3-rerank."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        instruction: str | None = None,
        timeout_seconds: int | None = None,
        max_document_chars: int = 4000,
    ):
        self.model_name = model_name or rag_conf.get("reranker_model_name", "qwen3-rerank")
        self.api_url = api_url or os.getenv("DASHSCOPE_RERANK_API_URL", DEFAULT_RERANK_URL)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.instruction = instruction or rag_conf.get(
            "reranker_instruction",
            "Given a customer-service question, retrieve relevant product-support passages that answer the question.",
        )
        self.timeout_seconds = int(
            timeout_seconds or rag_conf.get("reranker_timeout_seconds", 30)
        )
        self.max_document_chars = max_document_chars

        if not self.api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY，无法调用 qwen3-rerank")

    def rerank(
        self,
        *,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[Document]:
        if not documents:
            return []

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [self._document_text(doc) for doc in documents],
            "top_n": min(top_n, len(documents)),
            "instruct": self.instruction,
        }
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response_data = self._parse_response(response)
        rerank_results = self._extract_results(response_data)

        reranked_docs: list[Document] = []
        for new_rank, result in enumerate(rerank_results, start=1):
            if result.index < 0 or result.index >= len(documents):
                continue
            source_doc = documents[result.index]
            reranked_docs.append(
                Document(
                    page_content=source_doc.page_content,
                    metadata={
                        **source_doc.metadata,
                        "retrieval_strategy": "fusion_rerank",
                        "rerank_model": self.model_name,
                        "rerank_score": result.relevance_score,
                        "rerank_original_rank": result.index + 1,
                        "rerank_rank": new_rank,
                    },
                )
            )
        return reranked_docs

    def _document_text(self, doc: Document) -> str:
        text = doc.page_content.replace("\x00", "").strip()
        if len(text) <= self.max_document_chars:
            return text
        return text[: self.max_document_chars]

    def _parse_response(self, response: requests.Response) -> dict:
        try:
            response_data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"DashScope rerank 返回非 JSON 响应: HTTP {response.status_code}") from exc

        if response.status_code >= 400:
            code = response_data.get("code", response.status_code)
            message = response_data.get("message", "DashScope rerank 调用失败")
            request_id = response_data.get("request_id", "")
            detail = f"{code}: {message}"
            if request_id:
                detail += f" (request_id={request_id})"
            raise RuntimeError(detail)
        return response_data

    def _extract_results(self, response_data: dict) -> list[RerankResult]:
        raw_results = response_data.get("results")
        if raw_results is None:
            raw_results = (response_data.get("output") or {}).get("results", [])

        results: list[RerankResult] = []
        for item in raw_results or []:
            try:
                index = int(item["index"])
                relevance_score = float(item.get("relevance_score", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            results.append(RerankResult(index=index, relevance_score=relevance_score))
        return results
