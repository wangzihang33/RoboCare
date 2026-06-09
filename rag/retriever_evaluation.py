"""Retriever-level RAG evaluation CLI.

This script intentionally keeps only two resume-level retrieval metrics:
- Recall@K
- MRR
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Callable

from langchain_core.documents import Document

from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf, rag_conf
from utils.path_tool import get_abs_path


DEFAULT_DATASET = "data/rag_eval_dataset.csv"
DEFAULT_OUTPUT_DIR = "outputs/evaluations"
DEFAULT_STRATEGIES = ("vector", "bm25", "fusion")
SUPPORTED_STRATEGIES = ("vector", "bm25", "fusion", "fusion_rerank")


@dataclass
class EvalExample:
    query: str
    reference_answer: str
    query_type: str
    expected_source: str
    expected_card_id: str
    expected_evidence: str
    expected_keywords: str


@dataclass
class RetrieverEvalResult:
    run_id: str
    strategy: str
    query: str
    query_type: str
    reference_answer: str
    expected_source: str
    expected_card_id: str
    expected_evidence: str
    expected_keywords: str
    latency_ms: float
    retrieved_count: int
    first_relevant_rank: int
    recall_at_k: float
    mrr: float
    top_sources: str
    context_preview: str


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(get_abs_path(path_value))


def path_is_inside(child: Path, parent: Path) -> bool:
    child = child.resolve()
    parent = parent.resolve()
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def reset_local_index() -> None:
    """Remove local Chroma and MD5 state before rebuilding eval indexes."""
    project_root = resolve_project_path(".").resolve()
    targets = [
        resolve_project_path(chroma_conf["persist_directory"]),
        resolve_project_path(chroma_conf["md5_hex_store"]),
    ]

    for target in targets:
        if not path_is_inside(target, project_root):
            raise ValueError(f"拒绝清理项目目录外的索引路径: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def load_eval_dataset(dataset_path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = (row.get("query") or "").strip()
            if not query:
                continue
            examples.append(
                EvalExample(
                    query=query,
                    reference_answer=(row.get("reference_answer") or "").strip(),
                    query_type=(row.get("type") or "未分类").strip(),
                    expected_source=(row.get("expected_source") or "").strip(),
                    expected_card_id=(row.get("expected_card_id") or "").strip(),
                    expected_evidence=(row.get("expected_evidence") or "").strip(),
                    expected_keywords=(row.get("expected_keywords") or "").strip(),
                )
            )
    return examples


def extract_terms(text: str) -> set[str]:
    """Extract lightweight Chinese n-gram and alphanumeric terms."""
    terms = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", text)}
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", text)
    for chunk in chinese_chunks:
        for n in (2, 3):
            if len(chunk) < n:
                continue
            for idx in range(len(chunk) - n + 1):
                terms.add(chunk[idx : idx + n])
    return {term for term in terms if len(term) >= 2}


def split_expected_terms(value: str) -> list[str]:
    parts = re.split(r"[;；|,，]", value or "")
    return [part.strip() for part in parts if part.strip()]


def expected_terms_for(example: EvalExample) -> set[str]:
    explicit_terms = set()
    for term in split_expected_terms(example.expected_keywords):
        explicit_terms.update(extract_terms(term))
    if explicit_terms:
        return explicit_terms
    return extract_terms(example.reference_answer)


def normalize_source(value: str) -> str:
    if not value:
        return ""
    normalized = value.replace("\\", "/").strip().lower()
    return Path(normalized).name


def doc_source(doc: Document) -> str:
    source = doc.metadata.get("source") or doc.metadata.get("url") or ""
    return str(source)


def card_id_matches(doc: Document, expected_card_id: str) -> bool:
    if not expected_card_id:
        return True
    actual_card_id = str(doc.metadata.get("card_id") or "").strip()
    return actual_card_id == expected_card_id or expected_card_id in doc.page_content


def source_matches(doc: Document, expected_source: str) -> bool:
    if not expected_source:
        return True

    expected_name = normalize_source(expected_source)
    actual = doc_source(doc)
    actual_name = normalize_source(actual)
    actual_normalized = actual.replace("\\", "/").strip().lower()
    expected_normalized = expected_source.replace("\\", "/").strip().lower()

    return (
        expected_name == actual_name
        or expected_normalized in actual_normalized
        or expected_name in actual_normalized
    )


def evidence_phrases_for(example: EvalExample) -> list[str]:
    if not example.expected_evidence:
        return []
    return split_expected_terms(example.expected_evidence)


def evidence_matches_text(text: str, evidence_phrases: list[str]) -> bool:
    if not evidence_phrases:
        return False
    return any(phrase in text for phrase in evidence_phrases)


def keyword_coverage(text: str, expected_terms: set[str]) -> float:
    if not expected_terms:
        return 0.0
    matched = sum(1 for term in expected_terms if term in text)
    return matched / len(expected_terms)


def is_relevant_doc(
    doc: Document,
    example: EvalExample,
    expected_terms: set[str],
    evidence_phrases: list[str],
    threshold: float,
) -> bool:
    if not source_matches(doc, example.expected_source):
        return False
    if example.expected_card_id:
        return card_id_matches(doc, example.expected_card_id)

    evidence_hit = evidence_matches_text(doc.page_content, evidence_phrases)
    keyword_hit = keyword_coverage(doc.page_content, expected_terms) >= threshold
    return evidence_hit or keyword_hit


def first_relevant_rank(
    docs: list[Document],
    example: EvalExample,
    expected_terms: set[str],
    evidence_phrases: list[str],
    threshold: float,
) -> int:
    for idx, doc in enumerate(docs, start=1):
        if is_relevant_doc(doc, example, expected_terms, evidence_phrases, threshold):
            return idx
    return 0


def summarize_sources(docs: list[Document], limit: int = 3) -> str:
    sources: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source") or doc.metadata.get("url") or ""
        if source and source not in sources:
            sources.append(str(source))
        if len(sources) >= limit:
            break
    return " | ".join(sources)


def truncate_text(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def evaluate_docs(
    *,
    run_id: str,
    strategy: str,
    example: EvalExample,
    docs: list[Document],
    latency_ms: float,
    relevance_threshold: float,
) -> RetrieverEvalResult:
    context = "\n".join(doc.page_content for doc in docs)
    expected_terms = expected_terms_for(example)
    evidence_phrases = evidence_phrases_for(example)
    rank = first_relevant_rank(
        docs,
        example,
        expected_terms,
        evidence_phrases,
        relevance_threshold,
    )

    return RetrieverEvalResult(
        run_id=run_id,
        strategy=strategy,
        query=example.query,
        query_type=example.query_type,
        reference_answer=example.reference_answer,
        expected_source=example.expected_source,
        expected_card_id=example.expected_card_id,
        expected_evidence=example.expected_evidence,
        expected_keywords=example.expected_keywords,
        latency_ms=latency_ms,
        retrieved_count=len(docs),
        first_relevant_rank=rank,
        recall_at_k=1.0 if rank else 0.0,
        mrr=1 / rank if rank else 0.0,
        top_sources=summarize_sources(docs),
        context_preview=truncate_text(context),
    )


def build_retrievers(
    vector_service: VectorStoreService,
    *,
    top_k: int,
) -> dict[str, Callable[[str], list[Document]]]:
    bm25_retriever = vector_service.bm25_retriever()
    fusion_retriever = vector_service.get_fusion_retriever(k=top_k)
    fusion_rerank_retriever: Callable[[str], list[Document]] | None = None

    def lazy_fusion_rerank(query: str) -> list[Document]:
        nonlocal fusion_rerank_retriever
        if fusion_rerank_retriever is None:
            fusion_rerank_retriever = vector_service.get_fusion_rerank_retriever(k=top_k)
        return fusion_rerank_retriever(query)

    return {
        "vector": lambda query: vector_service.vector_store.similarity_search(query, k=top_k),
        "bm25": lambda query: bm25_retriever(query, k=top_k),
        "fusion": lambda query: fusion_retriever(query),
        "fusion_rerank": lazy_fusion_rerank,
    }


def run_retriever_evaluation(
    *,
    examples: list[EvalExample],
    strategies: list[str],
    top_k: int,
    relevance_threshold: float,
    load_documents: bool,
    reset_index: bool,
) -> list[RetrieverEvalResult]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if reset_index:
        reset_local_index()

    vector_service = VectorStoreService()
    if load_documents:
        vector_service.load_document()

    retrievers = build_retrievers(vector_service, top_k=top_k)
    unknown_strategies = [name for name in strategies if name not in retrievers]
    if unknown_strategies:
        raise ValueError(f"未知检索策略: {', '.join(unknown_strategies)}")

    results: list[RetrieverEvalResult] = []
    for example in examples:
        for strategy in strategies:
            retriever = retrievers[strategy]
            started_at = time.perf_counter()
            docs = retriever(example.query)
            latency_ms = (time.perf_counter() - started_at) * 1000
            results.append(
                evaluate_docs(
                    run_id=run_id,
                    strategy=strategy,
                    example=example,
                    docs=docs,
                    latency_ms=latency_ms,
                    relevance_threshold=relevance_threshold,
                )
            )
    return results


def as_float(value: float) -> str:
    return f"{value:.4f}"


def aggregate_by_strategy(results: list[RetrieverEvalResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RetrieverEvalResult]] = {}
    for result in results:
        grouped.setdefault(result.strategy, []).append(result)

    summary: dict[str, dict[str, float]] = {}
    for strategy, items in grouped.items():
        summary[strategy] = {
            "queries": float(len(items)),
            "avg_latency_ms": mean(item.latency_ms for item in items),
            "avg_retrieved_count": mean(item.retrieved_count for item in items),
            "recall_at_k": mean(item.recall_at_k for item in items),
            "mrr": mean(item.mrr for item in items),
        }
    return summary


def aggregate_by_type(results: list[RetrieverEvalResult]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[RetrieverEvalResult]] = {}
    for result in results:
        grouped.setdefault((result.query_type, result.strategy), []).append(result)

    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key, items in grouped.items():
        summary[key] = {
            "queries": float(len(items)),
            "recall_at_k": mean(item.recall_at_k for item in items),
            "mrr": mean(item.mrr for item in items),
            "avg_latency_ms": mean(item.latency_ms for item in items),
        }
    return summary


def save_csv_report(results: list[RetrieverEvalResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RetrieverEvalResult.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def escape_md(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().replace("|", "/")


def save_markdown_report(
    *,
    results: list[RetrieverEvalResult],
    output_path: Path,
    dataset_path: Path,
    top_k: int,
    relevance_threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_summary = aggregate_by_strategy(results)
    type_summary = aggregate_by_type(results)
    run_id = results[0].run_id if results else "empty"

    lines: list[str] = [
        "# RAG Retriever 评测报告",
        "",
        f"- run_id: `{run_id}`",
        f"- dataset: `{dataset_path}`",
        f"- top_k: `{top_k}`",
        "- fusion_method: `rrf`",
        f"- relevance_threshold: `{relevance_threshold}`",
        f"- total_rows: `{len(results)}`",
        "",
        "## 指标说明",
        "",
        "- `recall_at_k`: Top-K 中是否命中标准来源，且满足证据短语或关键词覆盖条件。",
        "- `mrr`: 第一条相关文档的排名倒数；第一名命中为 1.0，第二名命中为 0.5，未命中为 0。",
        "",
        "## 策略汇总",
        "",
        "| strategy | queries | recall_at_k | mrr | avg_retrieved_count | avg_latency_ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if any(result.strategy == "fusion_rerank" for result in results):
        lines.insert(7, f"- reranker_model: `{rag_conf.get('reranker_model_name', 'qwen3-rerank')}`")

    for strategy, values in sorted(strategy_summary.items()):
        lines.append(
            "| {strategy} | {queries:.0f} | {recall} | {mrr} | {retrieved} | {latency} |".format(
                strategy=strategy,
                queries=values["queries"],
                recall=as_float(values["recall_at_k"]),
                mrr=as_float(values["mrr"]),
                retrieved=as_float(values["avg_retrieved_count"]),
                latency=as_float(values["avg_latency_ms"]),
            )
        )

    lines.extend(
        [
            "",
            "## 按问题类型汇总",
            "",
            "| type | strategy | queries | recall_at_k | mrr | avg_latency_ms |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    for (query_type, strategy), values in sorted(type_summary.items()):
        lines.append(
            "| {query_type} | {strategy} | {queries:.0f} | {recall} | {mrr} | {latency} |".format(
                query_type=escape_md(query_type),
                strategy=strategy,
                queries=values["queries"],
                recall=as_float(values["recall_at_k"]),
                mrr=as_float(values["mrr"]),
                latency=as_float(values["avg_latency_ms"]),
            )
        )

    lines.extend(
        [
            "",
            "## 单题结果",
            "",
            "| type | query | strategy | expected_source | expected_card_id | first_relevant_rank | recall_at_k | mrr | retrieved | latency_ms |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )

    for result in results:
        lines.append(
            "| {query_type} | {query} | {strategy} | {expected_source} | {expected_card_id} | {rank} | {recall} | {mrr} | {retrieved} | {latency} |".format(
                query_type=escape_md(result.query_type),
                query=escape_md(result.query),
                strategy=result.strategy,
                expected_source=escape_md(result.expected_source),
                expected_card_id=escape_md(result.expected_card_id),
                rank=result.first_relevant_rank,
                recall=as_float(result.recall_at_k),
                mrr=as_float(result.mrr),
                retrieved=result.retrieved_count,
                latency=as_float(result.latency_ms),
            )
        )

    lines.extend(
        [
            "",
            "## V1 边界",
            "",
            "- 当前版本只评估 Retriever，不评估生成答案质量和完整 Agent 工具路由。",
            "- 如果评测集提供 `expected_card_id`，相关文档优先按知识卡片 ID 判断；否则回退到 `expected_source`、`expected_evidence` 和 `expected_keywords`。",
            "- 当前 card_id 还不是最终 chunk-level 标准文档 ID，后续可以继续细化。",
            "- Generator 层评测请运行 `python -m rag.generation_evaluation`。",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG retriever evaluation.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="评测集 CSV 路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="评测报告输出目录")
    parser.add_argument("--top-k", type=int, default=3, help="每种检索策略返回的文档数")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.25,
        help="相关文档的关键词覆盖阈值",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help=f"逗号分隔的检索策略: {','.join(SUPPORTED_STRATEGIES)}",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="最多评测多少条样本；0 表示全量评测",
    )
    parser.add_argument(
        "--skip-load-documents",
        action="store_true",
        help="跳过知识库入库步骤，直接使用已有 chroma_db",
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="评测前清理 chroma_db 和 md5.text，并重新加载 data 目录知识库",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reset_index and args.skip_load_documents:
        raise ValueError("--reset-index 不能和 --skip-load-documents 同时使用")

    dataset_path = resolve_project_path(args.dataset)
    output_dir = resolve_project_path(args.output_dir)
    examples = load_eval_dataset(dataset_path)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError(f"评测集为空: {dataset_path}")

    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    results = run_retriever_evaluation(
        examples=examples,
        strategies=strategies,
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
        load_documents=not args.skip_load_documents,
        reset_index=args.reset_index,
    )

    run_id = results[0].run_id
    csv_path = output_dir / f"rag_retriever_eval_{run_id}.csv"
    md_path = output_dir / f"rag_retriever_eval_{run_id}.md"
    save_csv_report(results, csv_path)
    save_markdown_report(
        results=results,
        output_path=md_path,
        dataset_path=dataset_path,
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
    )

    summary = aggregate_by_strategy(results)
    print(f"Retriever 评测完成: {len(results)} rows")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    for strategy, values in sorted(summary.items()):
        print(
            f"- {strategy}: recall@k={values['recall_at_k']:.4f}, "
            f"mrr={values['mrr']:.4f}, latency={values['avg_latency_ms']:.2f}ms"
        )


if __name__ == "__main__":
    main()
