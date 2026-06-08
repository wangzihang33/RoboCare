# websearch/retrieval.py
import os
import yaml
from .fetch_web_content import WebContentFetcher
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from model.factory import embed_model
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

chroma_client = chromadb.Client()


class EmbeddingRetriever:
    TOP_K = 20  # Number of top K documents to retrieve

    def __init__(self):
        # 使用项目路径工具加载配置
        config_path = get_abs_path("config/websearch.yml")
        with open(config_path, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        logger.info(f"[EmbeddingRetriever] 加载配置文件: {config_path}")

        # 初始化文本切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        logger.info(f"[EmbeddingRetriever] 文本切分器初始化完成")

    def retrievel_embeddings(self, contents_list: list, link_list: list, query: str):
        logger.info(f"[EmbeddingRetriever] 开始向量化检索, 查询: {query}")
        # 1️⃣ 生成文档对象
        metadatas = [{'url': link} for link in link_list]
        texts = self.text_splitter.create_documents(contents_list, metadatas=metadatas)
        logger.info(f"[EmbeddingRetriever] 创建 {len(texts)} 个文档块")

        # 3️⃣ 创建 Chroma 数据库
        db = Chroma.from_documents(documents=texts, embedding=embed_model)
        logger.info(f"[EmbeddingRetriever] Chroma 向量库创建完成")

        # 4️⃣ 创建检索器
        retriever = db.as_retriever(search_kwargs={"k": min(self.TOP_K, len(texts))})
        relevant_docs = retriever.invoke(query)
        logger.info(f"[EmbeddingRetriever] 检索完成, 返回 {len(relevant_docs)} 条相关文档")
        return relevant_docs


def web_retrieval(query):
    logger.info(f"[web_retrieval] 开始执行 WebSearch 查询: {query}")
    fetcher = WebContentFetcher(query)
    web_contents, serper_response = fetcher.fetch()
    logger.info(f"[web_retrieval] 抓取完成, 获取 {len(web_contents)} 条网页内容")

    retriever = EmbeddingRetriever()
    relevant_docs_list = retriever.retrievel_embeddings(web_contents, serper_response['links'], query)
    return relevant_docs_list


# Example usage
if __name__ == "__main__":
    query = "目前市面上最好的智能扫地机器人是什么品牌？有什么优势？"
    relevant_docs_list = web_retrieval(query)
    logger.info(f"\n\nRelevant Documents from VectorDB:\n{relevant_docs_list}")