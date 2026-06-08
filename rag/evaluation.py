# evaluation.py
import os
import pandas as pd
import numpy as np
from rag.vector_store import VectorStoreService
from agent.react_agent import ReactAgent
from openai import OpenAI  # DeepSeek SDK 接口兼容
from utils.path_tool import get_abs_path

# ==================== 配置 DeepSeek ====================
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY', '').strip(),
    base_url="https://api.deepseek.com"
)

# ==================== 路径 ====================
TEST_CSV_PATH = get_abs_path("data/rag_eval_dataset.csv")

# ==================== 初始化 VectorStore & Agent ====================
vs = VectorStoreService()
vs.load_document()
agent = ReactAgent()

vector_retriever = vs.get_retriever()
bm25_retriever = vs.bm25_retriever()
fusion_retriever = vs.get_fusion_retriever(alpha=0.7, k=3)

# ==================== 读取测试集 ====================
df = pd.read_csv(TEST_CSV_PATH)

# ==================== 辅助函数 ====================
def answer_with_agent(docs, query):
    """使用 Agent 对检索到的文档生成回答"""
    content_list = [d.page_content for d in docs]
    input_query = query + "\n参考文档:\n" + "\n".join(content_list)
    
    response = ""
    for chunk in agent.execute_stream(input_query):
        response += chunk
    return response.strip()

def evaluate_responses(query, vector_resp, bm25_resp, fusion_resp, reference_answer=None):
    """使用 DeepSeek LLM 对三种检索回答进行自动对比评估"""
    system_prompt = """You are an expert evaluator of RAG systems. Compare responses from three different retrieval approaches:
1. Vector-based retrieval: Uses semantic similarity for document retrieval
2. BM25 keyword retrieval: Uses keyword matching for document retrieval
3. Fusion retrieval: Combines both vector and keyword approaches

Evaluate the responses based on:
- Relevance to the query
- Factual correctness
- Comprehensiveness
- Clarity and coherence
"""

    user_prompt = f"""Query: {query}

Vector-based response:
{vector_resp}

BM25 keyword response:
{bm25_resp}

Fusion response:
{fusion_resp}
"""

    if reference_answer:
        user_prompt += f"\nReference answer:\n{reference_answer}\n"

    user_prompt += "\nPlease provide a detailed comparison of these three responses. Which approach performed best and why?"

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=False
    )

    return response.choices[0].message.content

def generate_overall_analysis(results):
    """使用 DeepSeek LLM 对整个测试集生成整体分析报告"""
    system_prompt = """You are an expert at evaluating information retrieval systems. 
    Based on multiple test queries, provide an overall analysis comparing three retrieval approaches:
    1. Vector-based retrieval (semantic similarity)
    2. BM25 keyword retrieval (keyword matching)
    3. Fusion retrieval (combination of both)

    Focus on:
    1. Types of queries where each approach performs best
    2. Overall strengths and weaknesses of each approach
    3. How fusion retrieval balances the trade-offs
    4. Recommendations for when to use each approach
    """

    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Comparison Summary: {result['comparison'][:200]}...\n\n"

    user_prompt = f"""Based on the following evaluations of different retrieval methods across {len(results)} queries, 
    provide an overall analysis comparing these three approaches:

    {evaluations_summary}

    Please provide a comprehensive analysis of vector-based, BM25, and fusion retrieval approaches,
    highlighting when and why fusion retrieval provides advantages over the individual methods.
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=False
    )

    return response.choices[0].message.content

# ==================== 主评估函数 ====================
def evaluate_rag(test_df):
    """遍历测试集，执行三种检索 + Agent回答 + LLM评估"""
    results = []

    for i, row in test_df.iterrows():
        query = row["query"]
        reference = row.get("reference_answer", None)

        print(f"\n=== Evaluating Query {i+1}/{len(test_df)} ===")
        print(f"Query: {query}")

        # 向量-only
        vector_docs = vector_retriever.invoke(query)
        vector_resp = answer_with_agent(vector_docs, query)
        print("\n--- Vector-only Response ---")
        print(vector_resp)

        # BM25-only
        bm25_docs = bm25_retriever(query, k=3)
        bm25_resp = answer_with_agent(bm25_docs, query)
        print("\n--- BM25-only Response ---")
        print(bm25_resp)

        # 融合检索
        fusion_docs = fusion_retriever(query)
        fusion_resp = answer_with_agent(fusion_docs, query)
        print("\n--- Fusion Response ---")
        print(fusion_resp)

        # LLM 对比
        comparison = evaluate_responses(query, vector_resp, bm25_resp, fusion_resp, reference)
        print("\n--- LLM Comparison ---")
        print(comparison)

        results.append({
            "query": query,
            "vector_resp": vector_resp,
            "bm25_resp": bm25_resp,
            "fusion_resp": fusion_resp,
            "comparison": comparison
        })

    return results

# ==================== 主程序 ====================
if __name__ == "__main__":
    results = evaluate_rag(df)
    print("\n=== Overall Analysis ===")
    overall_analysis = generate_overall_analysis(results)
    print(overall_analysis)
