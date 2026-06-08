from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
import os
from utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_core.documents import Document

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

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
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

    #### 新增融合检索方法 ####
    def get_fusion_retriever(self, alpha=0.7, k=None):
        """
        向量检索 + BM25 检索融合，通过 doc_id 对齐
        alpha: 向量权重, 1-alpha 为 BM25 权重
        k: 返回 top-k 文档
        """
        k = k or chroma_conf["k"]
        epsilon = 1e-8

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

        def fusion_invoke(query: str):
            # 1. 向量检索所有文档（返回距离）
            vector_results = self.vector_store.similarity_search_with_score(query, k=len(docs))

            # 2. BM25 得分（原始顺序）
            query_tokens = list(jieba.cut(query))
            bm25_scores = bm25.get_scores(query_tokens)
            bm25_score_by_id = {docs[i]["doc_id"]: bm25_scores[i] for i in range(len(docs))}

            # 3. 对齐 doc_id 并将距离转为相似度
            combined = []
            for doc, vec_distance in vector_results:
                doc_id = doc.metadata.get("doc_id")
                if doc_id is None or doc_id not in bm25_score_by_id:
                    continue
                # 距离转相似度
                vec_similarity = 1 / (1 + vec_distance)
                combined.append({
                    "doc": doc,
                    "doc_id": doc_id,
                    "vector_score": vec_similarity,
                    "bm25_score": bm25_score_by_id[doc_id]
                })

            if not combined:
                return []

            # 4. 归一化
            vec_scores = np.array([c["vector_score"] for c in combined])
            bm25_scores_arr = np.array([c["bm25_score"] for c in combined])
            norm_vec = (vec_scores - np.min(vec_scores)) / (np.max(vec_scores) - np.min(vec_scores) + epsilon)
            norm_bm25 = (bm25_scores_arr - np.min(bm25_scores_arr)) / (np.max(bm25_scores_arr) - np.min(bm25_scores_arr) + epsilon)

            # 5. 加权融合
            for i, c in enumerate(combined):
                c["combined_score"] = alpha * norm_vec[i] + (1 - alpha) * norm_bm25[i]

            # 6. 返回 top-k 文档
            combined.sort(key=lambda x: x["combined_score"], reverse=True)
            top_docs = [c["doc"] for c in combined[:k]]
            return top_docs

        return fusion_invoke
    #### BM25 单独检索 ####
    def bm25_retriever(self):
        all_docs_data = self.vector_store.get(include=["documents", "metadatas"])
        docs = [{"text": doc, "metadata": meta, "doc_id": meta.get("doc_id", f"doc_{i}")}
                for i, (doc, meta) in enumerate(zip(all_docs_data["documents"], all_docs_data["metadatas"]))]

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
    fusion_retriever = vs.get_fusion_retriever(alpha=0.7, k=3)
    fusion_res = fusion_retriever(query)
    print("\n=== 融合检索结果 ===")
    for r in fusion_res:
        print(r.page_content)
        print("="*20)
