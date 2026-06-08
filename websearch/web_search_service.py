from websearch.fetch_web_content import WebContentFetcher
from websearch.retrieval import EmbeddingRetriever
from model.factory import chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts

def print_prompt(full_prompt):
    print("="*20, full_prompt.to_string(), "="*20)
    return full_prompt

class WebSearchService:
    """
    WebSearch 封装服务
    - 查询 -> 网页抓取 -> 向量检索 -> 上下文拼接 -> LLM 输出
    """
    def __init__(self):
        self.fetcher = WebContentFetcher("")
        self.retriever_class = EmbeddingRetriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
        logger.info("[WebSearchService] 初始化完成")

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        # 获取向量检索文档
        self.fetcher.query = query
        web_contents, serper_response = self.fetcher.fetch()
        if not serper_response:
            logger.warning(f"[WebSearchService] 未获取到搜索结果: {query}")
            return []

        relevant_docs = self.retriever_class.retrievel_embeddings(
            web_contents, serper_response['links'], query
        )
        logger.info(f"[WebSearchService] 向量检索到 {len(relevant_docs)} 条文档")
        return relevant_docs

    def search_summarize(self, query: str) -> str:
        # 将检索结果拼接成上下文
        context_docs = self.retriever_docs(query)
        context = ""
        for idx, doc in enumerate(context_docs, 1):
            context += f"【参考资料{idx}】：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        # 调用 LLM 输出
        return self.chain.invoke({"input": query, "context": context})


# 测试
if __name__ == "__main__":
    websearch = WebSearchService()
    print(websearch.search_summarize("现在市面上有什么性价比好的扫地机器人？"))