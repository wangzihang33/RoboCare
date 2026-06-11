import yaml
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from websearch.fetch_web_content import WebContentFetcher
from websearch.models import FetchedPage


class EmbeddingRetriever:
    TOP_K = 20

    def __init__(self):
        config_path = get_abs_path("config/websearch.yml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        logger.info(f"[EmbeddingRetriever] loaded config: {config_path}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
        )
        logger.info("[EmbeddingRetriever] text splitter initialized")

    def retrievel_embeddings(self, contents_list: list, link_list: list, query: str):
        metadatas = [{"url": link} for link in link_list]
        texts = self.text_splitter.create_documents(contents_list, metadatas=metadatas)
        if not texts:
            logger.warning(f"[EmbeddingRetriever] no web chunks available for query: {query}")
            return []

        db = Chroma.from_documents(documents=texts, embedding=embed_model)
        retriever = db.as_retriever(search_kwargs={"k": min(self.TOP_K, len(texts))})
        relevant_docs = retriever.invoke(query)
        logger.info(f"[EmbeddingRetriever] legacy retrieval returned {len(relevant_docs)} docs")
        return relevant_docs

    def retrieve_page_embeddings(self, pages: list[FetchedPage], query: str):
        valid_pages = [
            page
            for page in pages
            if page.status == "success" and page.content.strip()
        ]
        if not valid_pages:
            logger.warning(f"[EmbeddingRetriever] no valid fetched pages for query: {query}")
            return []

        contents = [page.content for page in valid_pages]
        metadatas = [page.to_metadata() for page in valid_pages]
        texts = self.text_splitter.create_documents(contents, metadatas=metadatas)
        if not texts:
            logger.warning(f"[EmbeddingRetriever] no chunks generated for query: {query}")
            return []

        db = Chroma.from_documents(documents=texts, embedding=embed_model)
        retriever = db.as_retriever(search_kwargs={"k": min(self.TOP_K, len(texts))})
        relevant_docs = retriever.invoke(query)
        logger.info(f"[EmbeddingRetriever] page-level retrieval returned {len(relevant_docs)} docs")
        return relevant_docs


def web_retrieval(query):
    logger.info(f"[web_retrieval] start WebSearch query: {query}")
    fetcher = WebContentFetcher(query)
    pages, _ = fetcher.fetch_pages()
    logger.info(f"[web_retrieval] fetched {len(pages)} pages")

    retriever = EmbeddingRetriever()
    return retriever.retrieve_page_embeddings(pages, query)


if __name__ == "__main__":
    query = "目前市面上最好的智能扫地机器人是什么品牌？有什么优势？"
    relevant_docs_list = web_retrieval(query)
    logger.info(f"\n\nRelevant Documents from VectorDB:\n{relevant_docs_list}")
