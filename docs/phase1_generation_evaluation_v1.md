# Phase 1 Generator Evaluation V1 说明文档

版本目标：在已有 Retriever 评测闭环基础上，新增 Generator 层面的答案质量评估，让项目不仅能证明“检索命中”，也能证明“生成答案可信且相关”。

## 本版本实现了什么

- 新增生成层评测入口：`rag/generation_evaluation.py`。
- 复用 `data/rag_eval_dataset.csv`，不新增第二套评测集；当前 CSV 是 60 条 card-level hard set。
- 知识库入库语料包含 90 张 txt 知识卡片：60 张标准卡片和 30 张 `*-DIST-*` 近邻干扰卡片，使生成评测的 Top-K context 更接近真实客服场景。
- 复用当前 RAG 生成链路：
  - 使用 `VectorStoreService` 构建检索器。
  - 默认使用 `fusion` 检索策略获取 Top-K context。
  - 使用 `prompts/rag_summarize.txt` 生成最终答案。
- 使用 LLM-as-judge 评估两个指标：
  - `faithfulness`
  - `answer_relevance`
- 自动输出 CSV/Markdown 报告。
- 增加 `--max-examples`，方便先用少量样本试跑，控制 LLM 调用成本。
- 增加 `--reset-index`，可在评测前显式清理 `chroma_db/` 和 `md5.text`，避免旧语料污染新数据指标。

## 为什么只保留两个指标

本项目当前已经有 Retriever 层面的两个核心指标：`Recall@K` 和 `MRR`。

Generator 层面如果继续堆很多指标，简历和面试表达都会变散。因此 V1 只保留最能代表客服质量的两个指标：

| 指标 | 解决的问题 | 简历价值 |
|---|---|---|
| `faithfulness` | 答案是否被检索上下文支撑，是否减少幻觉 | 证明回答可信、可追溯 |
| `answer_relevance` | 答案是否直接回应用户问题 | 证明回答不是只“有根据”，而是真正有用 |

最终形成清晰的三层评测表达：

- Retriever：`Recall@K`、`MRR`
- Generator：`Faithfulness`、`Answer Relevance`
- Agent：后续增加 `Tool Selection Accuracy` 或 `Trajectory Match Rate`

## 指标定义

### Faithfulness

判断生成答案中的事实性陈述是否都能被 `retrieval_context` 支撑。

如果答案包含上下文没有提供的参数、承诺、诊断结论或售后政策，应扣分。

### Answer Relevance

判断生成答案是否直接回答 `query`，并覆盖 `reference_answer` 中的核心需求。

它不要求逐字匹配参考答案，但要求语义上回答了用户真正问的问题。

### Pass Rate

当 `faithfulness >= pass_threshold` 且 `answer_relevance >= pass_threshold`，并且生成和 Judge 都没有报错时，该样本记为通过。

默认阈值：

```text
pass_threshold = 0.7
```

## 怎么运行

前置条件：

- `.env` 中已配置 `DASHSCOPE_API_KEY`，用于生成答案和向量检索。
- 如果使用 DashScope 作为生成模型，`.env` 可配置 `DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`；未配置时使用该默认地址。
- `.env` 中已配置 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL`，用于默认 LLM Judge。
- 当前 Python 环境已安装项目依赖，例如 `langchain_community`、`langchain_chroma`、`langchain_openai`、`rank_bm25`、`jieba` 等。

先小样本试跑：

```bash
python -m rag.generation_evaluation --max-examples 5 --skip-load-documents
```

正式评测建议重建索引：

```bash
python -m rag.generation_evaluation --reset-index
```

指定生成模型和 Judge 模型：

```bash
python -m rag.generation_evaluation --generator-model qwen3.6-plus --judge-provider deepseek --judge-model deepseek-chat
python -m rag.generation_evaluation --generator-provider deepseek --generator-model deepseek-chat --judge-provider deepseek --judge-model deepseek-chat
python -m rag.generation_evaluation --generator-model qwen3.6-plus --judge-provider dashscope --judge-model qwen3.6-plus
```

如果 DashScope 返回 `AllocationQuota.FreeTierOnly`，说明当前生成模型免费额度已耗尽。可以在百炼控制台关闭 “use free tier only”，或临时切换到有额度的 DashScope 模型；当前项目默认生成模型已切换为 `qwen3.6-plus`，也可以用 `--generator-provider deepseek` 先完成生成评测链路验证。

比较不同检索策略对最终答案的影响：

```bash
python -m rag.generation_evaluation --strategies vector,bm25,fusion --max-examples 10
python -m rag.generation_evaluation --strategies fusion,fusion_rerank --max-examples 10
```

注意：`fusion_rerank` 会调用 DashScope `qwen3-rerank`，需要 `.env` 中配置 `DASHSCOPE_API_KEY`，并会额外产生 Reranker API 调用。

输出位置：

```text
outputs/evaluations/rag_generation_eval_YYYYMMDD_HHMMSS.csv
outputs/evaluations/rag_generation_eval_YYYYMMDD_HHMMSS.md
```

## 输出字段

| 字段 | 说明 |
|---|---|
| `query` | 用户问题 |
| `reference_answer` | 人工参考答案 |
| `expected_card_id` | 标准知识卡片 ID |
| `generated_answer` | RAG 链路生成答案 |
| `faithfulness` | Judge 给出的事实一致性评分 |
| `answer_relevance` | Judge 给出的回答相关性评分 |
| `passed` | 是否同时达到两个指标阈值 |
| `faithfulness_reason` | Judge 对事实一致性的简短解释 |
| `answer_relevance_reason` | Judge 对回答相关性的简短解释 |
| `unsupported_claims` | Judge 认为没有上下文支撑的关键陈述 |
| `retrieval_latency_ms` | 检索耗时 |
| `generation_latency_ms` | 生成耗时 |
| `judge_latency_ms` | Judge 耗时 |

## 与 Retriever 评测的关系

`rag/retriever_evaluation.py` 是 Retriever 评测入口，负责 `Recall@K` 和 `MRR`。

`rag/generation_evaluation.py` 是 Generator 评测入口，负责 `faithfulness` 和 `answer_relevance`。

两者复用同一份 `data/rag_eval_dataset.csv`，因此可以对同一批业务问题分别观察：

- 检索是否命中正确资料。
- 命中资料后生成答案是否可信。

## 本版本边界

- 当前使用 LLM-as-judge，分数不是绝对真值，应看趋势和失败样例。
- 默认生成模型读取 `config/rag.yml` 中的 `chat_model_name`，默认 Judge 使用 DeepSeek `deepseek-chat`。
- DeepSeek Judge 与生成模型分离，可以减少同模型自评偏置。
- 当前没有引入 DeepEval、Ragas 或 TruLens 依赖，避免为了 V1 增加额外框架复杂度。
- 当前不评估完整 Agent 工具路由，Agent 层评估留到 tracing 接入后推进。
- 当前 Judge 已支持 DeepSeek 和 DashScope 两种 provider；后续可以继续接入更多独立评测模型。

## 可写进简历的阶段成果

> 构建 Generator 层评测闭环，基于业务问答集使用 LLM-as-judge 评估答案 Faithfulness 与 Answer Relevance，并自动输出明细与汇总报告，使 RAG 优化从检索命中扩展到答案可信度评估。
