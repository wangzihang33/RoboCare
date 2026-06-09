from langchain_chroma import Chroma
from utils.config_handler import chroma_conf, rag_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
import os
import re
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document
from rag.reranker import DashScopeReranker

from rank_bm25 import BM25Okapi
import numpy as np
import jieba


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"]
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件中读取数据文件，转为向量存入向量库，使用 md5 去重
        且为每个 chunk 添加唯一 doc_id
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), 'w', encoding="utf-8").close()
                return False
            with open(get_abs_path(chroma_conf["md5_hex_store"]), 'r', encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), 'a', encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_txt_card_documents(read_path: str) -> list[Document]:
            with open(read_path, "r", encoding="utf-8") as f:
                text = f.read()

            card_pattern = re.compile(r"^##\s+([A-Z0-9-]+)\s+(.+)$", re.M)
            matches = list(card_pattern.finditer(text))
            if not matches:
                return txt_loader(read_path)

            documents: list[Document] = []
            for idx, match in enumerate(matches):
                card_id = match.group(1).strip()
                card_title = match.group(2).strip()
                start = match.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                body = text[start:end].strip()
                page_content = f"卡片ID：{card_id}\n标题：{card_title}\n{body}"
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata={
                            "source": read_path,
                            "card_id": card_id,
                            "card_title": card_title,
                        },
                    )
                )
            return documents

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return get_txt_card_documents(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            chroma_conf["data_path"],
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载数据库]{path}已经处理过了")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.info(f"[加载数据库]{path}没有内容")
                    continue
                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.info(f"[加载数据库]分片后{path}没有内容")
                    continue

                # 为每个分片添加唯一 ID
                for idx, doc in enumerate(split_document):
                    doc.metadata["doc_id"] = f"{md5_hex}_{idx}"

                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                logger.info(f"[加载数据库]{path}处理完成")
            except Exception as e:
                logger.error(f"[加载数据库]{path}处理失败: {str(e)}", exc_info=True)
                continue

    #### RRF 融合检索方法 ####
    def get_fusion_retriever(self, k=None, rrf_k=60, candidate_k=None):
        """
        向量检索 + BM25 检索的 Reciprocal Rank Fusion。

        RRF 不直接混合不同检索器的原始分数，而是融合各自的排名：
        score = sum(1 / (rrf_k + rank))
        """
        k = k or chroma_conf["k"]

        # 获取所有文档及元数据（包含 doc_id）
        all_docs_data = self.vector_store.get(include=["documents", "metadatas"])
        docs = []
        for idx, (text, meta) in enumerate(zip(all_docs_data["documents"], all_docs_data["metadatas"])):
            doc_id = meta.get("doc_id", f"doc_{idx}")
            docs.append({"text": text, "metadata": meta, "doc_id": doc_id})

        if not docs:
            logger.warning("[fusion retriever] 向量库中没有可检索文档")
            return lambda query: []

        # 构建 BM25 索引
        tokenized_texts = [list(jieba.cut(d["text"])) for d in docs]
        bm25 = BM25Okapi(tokenized_texts)
        doc_by_id = {d["doc_id"]: d for d in docs}
        fetch_k = candidate_k or min(len(docs), max(k * 10, 20))

        def fusion_invoke(query: str):
            candidates: dict[str, dict] = {}

            def add_ranked_doc(doc_id: str, rank: int, channel: str):
                if not doc_id:
                    return
                candidate = candidates.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "score": 0.0,
                        "channels": [],
                    },
                )
                candidate["score"] += 1 / (rrf_k + rank)
                candidate["channels"].append(channel)

            # 1. 向量检索候选，并按向量排名贡献 RRF 分数
            vector_results = self.vector_store.similarity_search(query, k=fetch_k)
            for rank, doc in enumerate(vector_results, start=1):
                add_ranked_doc(str(doc.metadata.get("doc_id") or ""), rank, "vector")

            # 2. BM25 检索候选，并按 BM25 排名贡献 RRF 分数
            query_tokens = list(jieba.cut(query))
            bm25_scores = bm25.get_scores(query_tokens)
            bm25_top_idx = np.argsort(-bm25_scores)[:fetch_k]
            for rank, idx in enumerate(bm25_top_idx, start=1):
                if bm25_scores[idx] <= 0:
                    continue
                add_ranked_doc(docs[idx]["doc_id"], rank, "bm25")

            if not candidates:
                return []

            ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
            top_docs: list[Document] = []
            for item in ranked[:k]:
                raw_doc = doc_by_id.get(item["doc_id"])
                if not raw_doc:
                    continue
                top_docs.append(
                    Document(
                        page_content=raw_doc["text"],
                        metadata={
                            **raw_doc["metadata"],
                            "retrieval_strategy": "rrf_fusion",
                            "rrf_score": item["score"],
                            "rrf_channels": ",".join(item["channels"]),
                        },
                    )
                )
            return top_docs

        return fusion_invoke

    def get_fusion_rerank_retriever(self, k=None, candidate_k=None):
        """
        RRF 粗召回 + DashScope qwen3-rerank 二阶段重排。

        candidate_k 控制进入 reranker 的候选文档数，k 控制最终返回文档数。
        """
        k = k or chroma_conf["k"]
        candidate_k = candidate_k or int(rag_conf.get("reranker_candidate_k", 20))
        candidate_k = max(candidate_k, k)

        fusion_retriever = self.get_fusion_retriever(k=candidate_k)
        reranker = DashScopeReranker()

        def fusion_rerank_invoke(query: str):
            candidate_docs = fusion_retriever(query)
            return reranker.rerank(query=query, documents=candidate_docs, top_n=k)

        return fusion_rerank_invoke

    #### BM25 单独检索 ####
    def bm25_retriever(self):
        all_docs_data = self.vector_store.get(include=["documents", "metadatas"])
        docs = [{"text": doc, "metadata": meta, "doc_id": meta.get("doc_id", f"doc_{i}")}
                for i, (doc, meta) in enumerate(zip(all_docs_data["documents"], all_docs_data["metadatas"]))]

        if not docs:
            logger.warning("[bm25 retriever] 向量库中没有可检索文档")
            return lambda query, k=5: []

        tokenized_texts = [list(jieba.cut(d["text"])) for d in docs]
        bm25 = BM25Okapi(tokenized_texts)

        def retrieve(query: str, k=5):
            query_tokens = list(jieba.cut(query))
            scores = bm25.get_scores(query_tokens)
            top_idx = np.argsort(-scores)[:k]
            return [Document(page_content=docs[i]["text"], metadata=docs[i]["metadata"]) for i in top_idx]

        return retrieve


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()

    query = "机器人在低温环境下工作会对电池产生什么影响？"

    # 向量检索
    retriever = vs.get_retriever()
    res = retriever.invoke(query)
    print("\n=== Vector-only 检索结果 ===")
    for r in res:
        print(r.page_content)
        print("="*20)

    # BM25-only 检索
    bm25_retriever = vs.bm25_retriever()
    bm25_res = bm25_retriever(query, k=3)
    print("\n=== BM25-only 检索结果 ===")
    for r in bm25_res:
        print(r.page_content)
        print("="*20)

    # 融合检索
    fusion_retriever = vs.get_fusion_retriever(k=3)
    fusion_res = fusion_retriever(query)
    print("\n=== RRF 融合检索结果 ===")
    for r in fusion_res:
        print(r.page_content)
        print("="*20)
