"""Generator-level RAG evaluation with LLM-as-judge metrics.

V1 keeps the metric set intentionally small:
- faithfulness: whether the generated answer is supported by retrieved context.
- answer_relevance: whether the generated answer addresses the user query.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rag.retriever_evaluation import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    EvalExample,
    SUPPORTED_STRATEGIES,
    as_float,
    build_retrievers,
    load_eval_dataset,
    reset_local_index,
    resolve_project_path,
    summarize_sources,
    truncate_text,
)
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_conf
from utils.prompt_loader import load_rag_prompts


DEFAULT_STRATEGIES = ("fusion",)
DEFAULT_GENERATOR_PROVIDER = "dashscope"
DEFAULT_JUDGE_PROVIDER = "deepseek"
DEFAULT_DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class JudgeScore:
    faithfulness: float
    answer_relevance: float
    faithfulness_reason: str
    answer_relevance_reason: str
    unsupported_claims: str
    raw_output: str
    error: str


@dataclass
class GenerationEvalResult:
    run_id: str
    strategy: str
    query: str
    query_type: str
    reference_answer: str
    expected_source: str
    expected_card_id: str
    expected_evidence: str
    expected_keywords: str
    generated_answer: str
    faithfulness: float
    answer_relevance: float
    pass_threshold: float
    passed: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    judge_latency_ms: float
    retrieved_count: int
    top_sources: str
    context_preview: str
    faithfulness_reason: str
    answer_relevance_reason: str
    unsupported_claims: str
    generation_error: str
    judge_error: str
    raw_judge_output: str


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"缺少 {name}，请在项目 .env 中配置后重新运行")
    return value


def exception_summary(exc: Exception) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def build_chat_model(provider: str, model_name: str):
    provider = provider.lower().strip()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "生成评测需要安装 langchain-openai，请先运行 pip install -r requirements.txt"
        ) from exc

    if provider == "dashscope":
        return ChatOpenAI(
            model=model_name,
            api_key=require_env("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL),
            temperature=0,
        )
    if provider == "deepseek":
        return ChatOpenAI(
            model=model_name,
            api_key=require_env("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            temperature=0,
        )
    raise ValueError(f"不支持的模型 provider: {provider}")


def resolve_chat_model_name(provider: str, model_name: str) -> str:
    if model_name:
        return model_name
    if provider.lower().strip() == "deepseek":
        return DEFAULT_DEEPSEEK_CHAT_MODEL
    return rag_conf["chat_model_name"]


def invoke_model_text(model, prompt: str) -> str:
    response = model.invoke(prompt)
    content = getattr(response, "content", None)
    return str(content if content is not None else response).strip()


def format_context(docs: list[Document]) -> str:
    lines: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        lines.append(
            f"【参考资料{idx}】：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}"
        )
    return "\n".join(lines)


def build_generation_chain(generator_model):
    prompt_template = PromptTemplate.from_template(load_rag_prompts())
    return prompt_template | generator_model | StrOutputParser()


def generate_answer(chain, query: str, docs: list[Document]) -> str:
    return chain.invoke({"input": query, "context": format_context(docs)}).strip()


def build_judge_prompt(
    *,
    example: EvalExample,
    context: str,
    generated_answer: str,
) -> str:
    return f"""
你是一个严格的 RAG 客服系统评测员。请只根据给定信息评分，不要补充外部知识。

评分指标：
1. faithfulness：生成答案中的事实性陈述是否都能被 retrieval_context 支撑。只看是否有根据，不因为答案简短而扣分。
2. answer_relevance：生成答案是否直接回答 user_query，是否覆盖 reference_answer 中的核心需求。

评分范围：
- 1.0 表示完全满足。
- 0.7 表示基本满足，但有轻微遗漏或表达不够直接。
- 0.4 表示部分满足，有明显遗漏、跑题或缺少关键事实。
- 0.0 表示基本不满足，或回答与上下文冲突。

请严格输出 JSON，不要输出 Markdown、解释性前后缀或代码块。JSON 字段如下：
{{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "faithfulness_reason": "一句话说明事实是否被上下文支撑",
  "answer_relevance_reason": "一句话说明是否回答了用户问题",
  "unsupported_claims": ["没有上下文支撑的关键陈述，如无则为空数组"]
}}

user_query:
{example.query}

reference_answer:
{example.reference_answer}

expected_evidence:
{example.expected_evidence}

retrieval_context:
{context}

generated_answer:
{generated_answer}
""".strip()


def clamp_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, score))


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
    if fenced_match:
        cleaned = fenced_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("judge 输出中没有 JSON 对象")
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def parse_judge_output(raw_output: str) -> JudgeScore:
    payload = extract_json_object(raw_output)
    unsupported_claims = payload.get("unsupported_claims", [])
    if isinstance(unsupported_claims, list):
        unsupported_claims_text = "; ".join(str(item) for item in unsupported_claims)
    else:
        unsupported_claims_text = str(unsupported_claims or "")

    return JudgeScore(
        faithfulness=clamp_score(payload.get("faithfulness")),
        answer_relevance=clamp_score(payload.get("answer_relevance")),
        faithfulness_reason=str(payload.get("faithfulness_reason") or "").strip(),
        answer_relevance_reason=str(payload.get("answer_relevance_reason") or "").strip(),
        unsupported_claims=unsupported_claims_text.strip(),
        raw_output=raw_output,
        error="",
    )


def judge_answer(
    *,
    judge_model,
    example: EvalExample,
    docs: list[Document],
    generated_answer: str,
) -> JudgeScore:
    context = format_context(docs)
    prompt = build_judge_prompt(
        example=example,
        context=context,
        generated_answer=generated_answer,
    )
    raw_output = invoke_model_text(judge_model, prompt)
    try:
        return parse_judge_output(raw_output)
    except Exception as exc:
        return JudgeScore(
            faithfulness=0.0,
            answer_relevance=0.0,
            faithfulness_reason="Judge 输出解析失败，无法确认事实一致性。",
            answer_relevance_reason="Judge 输出解析失败，无法确认回答相关性。",
            unsupported_claims="",
            raw_output=raw_output,
            error=str(exc),
        )


def run_generation_evaluation(
    *,
    examples: list[EvalExample],
    strategies: list[str],
    top_k: int,
    generator_provider: str,
    generator_model_name: str,
    judge_provider: str,
    judge_model_name: str,
    pass_threshold: float,
    load_documents: bool,
    reset_index: bool,
) -> list[GenerationEvalResult]:
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

    generation_chain = build_generation_chain(
        build_chat_model(generator_provider, generator_model_name)
    )
    judge_model = build_chat_model(judge_provider, judge_model_name)

    results: list[GenerationEvalResult] = []
    for example in examples:
        for strategy in strategies:
            retriever = retrievers[strategy]

            started_at = time.perf_counter()
            docs = retriever(example.query)
            retrieval_latency_ms = (time.perf_counter() - started_at) * 1000

            generation_started_at = time.perf_counter()
            generation_error = ""
            try:
                generated_answer = generate_answer(generation_chain, example.query, docs)
            except Exception as exc:
                generated_answer = ""
                generation_error = exception_summary(exc)
            generation_latency_ms = (time.perf_counter() - generation_started_at) * 1000

            judge_started_at = time.perf_counter()
            if generated_answer:
                judge_score = judge_answer(
                    judge_model=judge_model,
                    example=example,
                    docs=docs,
                    generated_answer=generated_answer,
                )
            else:
                judge_score = JudgeScore(
                    faithfulness=0.0,
                    answer_relevance=0.0,
                    faithfulness_reason="生成失败，无法评估事实一致性。",
                    answer_relevance_reason="生成失败，无法评估回答相关性。",
                    unsupported_claims="",
                    raw_output="",
                    error="",
                )
            judge_latency_ms = (time.perf_counter() - judge_started_at) * 1000

            passed = float(
                judge_score.faithfulness >= pass_threshold
                and judge_score.answer_relevance >= pass_threshold
                and not generation_error
                and not judge_score.error
            )

            results.append(
                GenerationEvalResult(
                    run_id=run_id,
                    strategy=strategy,
                    query=example.query,
                    query_type=example.query_type,
                    reference_answer=example.reference_answer,
                    expected_source=example.expected_source,
                    expected_card_id=example.expected_card_id,
                    expected_evidence=example.expected_evidence,
                    expected_keywords=example.expected_keywords,
                    generated_answer=generated_answer,
                    faithfulness=judge_score.faithfulness,
                    answer_relevance=judge_score.answer_relevance,
                    pass_threshold=pass_threshold,
                    passed=passed,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=generation_latency_ms,
                    judge_latency_ms=judge_latency_ms,
                    retrieved_count=len(docs),
                    top_sources=summarize_sources(docs),
                    context_preview=truncate_text(format_context(docs), limit=240),
                    faithfulness_reason=judge_score.faithfulness_reason,
                    answer_relevance_reason=judge_score.answer_relevance_reason,
                    unsupported_claims=judge_score.unsupported_claims,
                    generation_error=generation_error,
                    judge_error=judge_score.error,
                    raw_judge_output=judge_score.raw_output,
                )
            )

    return results


def aggregate_by_strategy(results: list[GenerationEvalResult]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[GenerationEvalResult]] = {}
    for result in results:
        grouped.setdefault(result.strategy, []).append(result)

    summary: dict[str, dict[str, float]] = {}
    for strategy, items in grouped.items():
        summary[strategy] = {
            "queries": float(len(items)),
            "faithfulness": mean(item.faithfulness for item in items),
            "answer_relevance": mean(item.answer_relevance for item in items),
            "pass_rate": mean(item.passed for item in items),
            "avg_retrieval_latency_ms": mean(item.retrieval_latency_ms for item in items),
            "avg_generation_latency_ms": mean(item.generation_latency_ms for item in items),
            "avg_judge_latency_ms": mean(item.judge_latency_ms for item in items),
        }
    return summary


def aggregate_by_type(results: list[GenerationEvalResult]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[GenerationEvalResult]] = {}
    for result in results:
        grouped.setdefault((result.query_type, result.strategy), []).append(result)

    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key, items in grouped.items():
        summary[key] = {
            "queries": float(len(items)),
            "faithfulness": mean(item.faithfulness for item in items),
            "answer_relevance": mean(item.answer_relevance for item in items),
            "pass_rate": mean(item.passed for item in items),
        }
    return summary


def save_csv_report(results: list[GenerationEvalResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(GenerationEvalResult.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def escape_md(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().replace("|", "/")


def save_markdown_report(
    *,
    results: list[GenerationEvalResult],
    output_path: Path,
    dataset_path: Path,
    top_k: int,
    generator_provider: str,
    generator_model_name: str,
    judge_provider: str,
    judge_model_name: str,
    pass_threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_summary = aggregate_by_strategy(results)
    type_summary = aggregate_by_type(results)
    run_id = results[0].run_id if results else "empty"

    lines: list[str] = [
        "# RAG Generator 评测报告",
        "",
        f"- run_id: `{run_id}`",
        f"- dataset: `{dataset_path}`",
        f"- top_k: `{top_k}`",
        "- fusion_method: `rrf`",
        f"- generator_provider: `{generator_provider}`",
        f"- generator_model: `{generator_model_name}`",
        f"- judge_provider: `{judge_provider}`",
        f"- judge_model: `{judge_model_name}`",
        f"- pass_threshold: `{pass_threshold}`",
        f"- total_rows: `{len(results)}`",
        "",
        "## 指标说明",
        "",
        "- `faithfulness`: 生成答案中的事实性陈述是否被检索上下文支撑。",
        "- `answer_relevance`: 生成答案是否直接回答用户问题，并覆盖参考答案中的核心需求。",
        "- `pass_rate`: `faithfulness` 和 `answer_relevance` 同时达到阈值，且生成/Judge 没有失败的样本占比。",
        "",
        "## 策略汇总",
        "",
        "| strategy | queries | faithfulness | answer_relevance | pass_rate | retrieval_ms | generation_ms | judge_ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if any(result.strategy == "fusion_rerank" for result in results):
        lines.insert(7, f"- reranker_model: `{rag_conf.get('reranker_model_name', 'qwen3-rerank')}`")

    for strategy, values in sorted(strategy_summary.items()):
        lines.append(
            "| {strategy} | {queries:.0f} | {faith} | {answer_rel} | {pass_rate} | {retrieval} | {generation} | {judge} |".format(
                strategy=strategy,
                queries=values["queries"],
                faith=as_float(values["faithfulness"]),
                answer_rel=as_float(values["answer_relevance"]),
                pass_rate=as_float(values["pass_rate"]),
                retrieval=as_float(values["avg_retrieval_latency_ms"]),
                generation=as_float(values["avg_generation_latency_ms"]),
                judge=as_float(values["avg_judge_latency_ms"]),
            )
        )

    lines.extend(
        [
            "",
            "## 按问题类型汇总",
            "",
            "| type | strategy | queries | faithfulness | answer_relevance | pass_rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    for (query_type, strategy), values in sorted(type_summary.items()):
        lines.append(
            "| {query_type} | {strategy} | {queries:.0f} | {faith} | {answer_rel} | {pass_rate} |".format(
                query_type=escape_md(query_type),
                strategy=strategy,
                queries=values["queries"],
                faith=as_float(values["faithfulness"]),
                answer_rel=as_float(values["answer_relevance"]),
                pass_rate=as_float(values["pass_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## 单题结果",
            "",
            "| type | query | strategy | faithfulness | answer_relevance | passed | answer | judge_reason |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
    )

    for result in results:
        judge_reason = (
            f"F: {result.faithfulness_reason} A: {result.answer_relevance_reason}"
        )
        lines.append(
            "| {query_type} | {query} | {strategy} | {faith} | {answer_rel} | {passed} | {answer} | {reason} |".format(
                query_type=escape_md(result.query_type),
                query=escape_md(result.query),
                strategy=result.strategy,
                faith=as_float(result.faithfulness),
                answer_rel=as_float(result.answer_relevance),
                passed=as_float(result.passed),
                answer=escape_md(truncate_text(result.generated_answer, limit=120)),
                reason=escape_md(truncate_text(judge_reason, limit=160)),
            )
        )

    lines.extend(
        [
            "",
            "## V1 边界",
            "",
            "- 当前版本使用 LLM-as-judge，分数会受 Judge 模型稳定性影响；建议关注趋势，不把单次分数当绝对真值。",
            "- 当前默认使用 `fusion` 检索策略生成答案，不评估完整 Agent 工具路由。",
            "- 当前只保留两个生成层指标，避免指标过多导致简历表达分散。",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG generator evaluation.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="评测集 CSV 路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="评测报告输出目录")
    parser.add_argument("--top-k", type=int, default=3, help="生成答案时使用的检索文档数")
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help=f"逗号分隔的生成评测策略，支持 {','.join(SUPPORTED_STRATEGIES)}；默认只跑 fusion 以控制 LLM 成本",
    )
    parser.add_argument(
        "--generator-provider",
        default=DEFAULT_GENERATOR_PROVIDER,
        choices=("dashscope", "deepseek"),
        help="生成答案的模型提供方，默认 dashscope",
    )
    parser.add_argument(
        "--generator-model",
        default="",
        help="用于生成答案的模型名；DashScope 默认读取 config/rag.yml，DeepSeek 默认 deepseek-chat",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="用于 LLM Judge 的模型名；DeepSeek 默认 deepseek-chat，DashScope 默认读取 config/rag.yml",
    )
    parser.add_argument(
        "--judge-provider",
        default=DEFAULT_JUDGE_PROVIDER,
        choices=("deepseek", "dashscope"),
        help="LLM Judge 提供方，默认 deepseek",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.7,
        help="faithfulness 和 answer_relevance 的通过阈值",
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
    generator_model_name = resolve_chat_model_name(args.generator_provider, args.generator_model)
    judge_model_name = resolve_chat_model_name(args.judge_provider, args.judge_model)
    results = run_generation_evaluation(
        examples=examples,
        strategies=strategies,
        top_k=args.top_k,
        generator_provider=args.generator_provider,
        generator_model_name=generator_model_name,
        judge_provider=args.judge_provider,
        judge_model_name=judge_model_name,
        pass_threshold=args.pass_threshold,
        load_documents=not args.skip_load_documents,
        reset_index=args.reset_index,
    )

    run_id = results[0].run_id
    csv_path = output_dir / f"rag_generation_eval_{run_id}.csv"
    md_path = output_dir / f"rag_generation_eval_{run_id}.md"
    save_csv_report(results, csv_path)
    save_markdown_report(
        results=results,
        output_path=md_path,
        dataset_path=dataset_path,
        top_k=args.top_k,
        generator_provider=args.generator_provider,
        generator_model_name=generator_model_name,
        judge_provider=args.judge_provider,
        judge_model_name=judge_model_name,
        pass_threshold=args.pass_threshold,
    )

    summary = aggregate_by_strategy(results)
    print(f"Generator 评测完成: {len(results)} rows")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    for strategy, values in sorted(summary.items()):
        print(
            f"- {strategy}: faithfulness={values['faithfulness']:.4f}, "
            f"answer_relevance={values['answer_relevance']:.4f}, "
            f"pass_rate={values['pass_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
