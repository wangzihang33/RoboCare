from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


def build_local_rag_retriever(
    vector_service: VectorStoreService,
    top_k: int = 3,
    *,
    use_reranker: bool = True,
):
    """Build Hybrid RAG, optionally using the faster RRF-only evaluation mode."""
    if not use_reranker:
        return vector_service.get_fusion_retriever(k=top_k)
    return vector_service.get_fusion_rerank_retriever(k=top_k)


def print_prompt(full_prompt):
    print("="*20, full_prompt.to_string(), "="*20)
    return full_prompt

class RagSummarizeservice(object):
    def __init__(self, *, use_reranker: bool = True):
        self.vector_service = VectorStoreService()
        self.retriever = build_local_rag_retriever(
            self.vector_service,
            top_k=3,
            use_reranker=use_reranker,
        )
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()


    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain
    
    def retriever_docs(self, query: str) -> list[Document]:
        if callable(self.retriever):
            return self.retriever(query)
        return self.retriever.invoke(query)
    
    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke({"input": query, "context": context})
    

if __name__ == "__main__":
    rag = RagSummarizeservice()
    print(rag.rag_summarize("小户型适合哪些扫地机器人？"))
