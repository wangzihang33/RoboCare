# Phase 1 Retriever Evaluation V1 说明文档

版本目标：让项目从“能回答”升级到“能证明检索质量”。本版本先建立稳定、可重复运行、可落盘的 Retriever 评测基线，并将融合检索升级为 RRF，为后续 Reranker、WebSearch 缓存、Tracing 和 MCP 工具化提供可对比指标。

## 本版本实现了什么

- 基于现有 `data/rag_eval_dataset.csv` 构建自动评测入口。
- 重建 `data/rag_eval_dataset.csv`，每条样本增加标准来源、标准卡片 ID、证据短语和标准关键词。
- 将旧的实验脚本重构为命令行评测工具：`rag/retriever_evaluation.py`。
- 支持三种检索策略对比：
  - `vector`：纯向量检索。
  - `bm25`：关键词检索。
  - `fusion`：向量 + BM25 的 RRF 融合检索。
  - `fusion_rerank`：RRF 粗召回 + DashScope `qwen3-rerank` 二阶段重排。
- 每次运行自动生成两类报告：
  - CSV 明细报告：逐 query、逐策略记录指标。
  - Markdown 汇总报告：包含策略汇总、问题类型汇总和单题结果。
- 默认使用确定性 Retriever 指标，不依赖 DeepSeek/OpenAI 等 LLM Judge，因此没有评测模型 key 也能跑。
- 为 BM25 和 Fusion 检索增加空知识库保护，避免 Chroma 没有文档时直接报错。
- txt 知识文件按 `## CARD-ID 标题` 切分为知识卡片，并在 metadata 中写入 `card_id`，使评测从 source 级升级为 card 级。

## 评测集设计

旧版评测集只有 `query`、`reference_answer`、`type`，没有标准答案文档 ID 或标准来源，因此 Recall@K 和 MRR 只能依赖参考答案关键词覆盖，证据强度不足。

当前评测集已升级为 60 条 card-level hard set，覆盖 6 个主要 txt 知识文件，每个来源 10 条：

- `data/智扫通.txt`
- `data/选购指南.txt`
- `data/维护保养.txt`
- `data/故障排除.txt`
- `data/扫地机器人100问2.txt`
- `data/扫拖一体机器人100问.txt`

`data/扫地机器人100问.pdf` 仍保留在 `data/` 目录中参与入库，可作为额外干扰来源；但当前 hard set 不将 PDF 作为标准答案来源。扫描型 PDF 或图片型资料后续可单独扩展为 OCR/多模态文档检索评测集。

当前 6 个 txt 知识文件已整理为 90 张客服知识卡片，每个文件 15 张。其中 60 张是评测集标注的标准卡片，30 张是 `*-DIST-*` 近邻干扰卡片。干扰卡不会出现在 `expected_card_id` 中，但会参与向量检索、BM25 和融合检索，用来压低过于容易的 Top-3 命中率。

评测问题采用口语化、多条件、易混淆表达，并通过 `expected_card_id` 标注标准卡片，避免只命中同一来源文件就算正确。新增干扰卡后，`--top-k 3` 更适合作为日常 hard set 指标，`--top-k 5` 可以作为宽松召回观察。

字段说明：

| 字段 | 说明 |
|---|---|
| `query` | 用户问题 |
| `reference_answer` | 人工参考答案 |
| `type` | 问题类型 |
| `expected_source` | 标准来源文件 |
| `expected_card_id` | 标准知识卡片 ID |
| `expected_evidence` | 来源文件中的证据短语 |
| `expected_keywords` | 评测用关键词，使用 `;` 分隔 |

## 怎么运行

在项目根目录执行：

```bash
python -m rag.retriever_evaluation
```

常用参数：

```bash
python -m rag.retriever_evaluation --top-k 3
python -m rag.retriever_evaluation --strategies vector,bm25,fusion
python -m rag.retriever_evaluation --strategies fusion,fusion_rerank --max-examples 5 --skip-load-documents
python -m rag.retriever_evaluation --strategies vector,bm25,fusion,fusion_rerank
python -m rag.retriever_evaluation --skip-load-documents
python -m rag.retriever_evaluation --reset-index
python -m rag.retriever_evaluation --dataset data/rag_eval_dataset.csv --output-dir outputs/evaluations
```

注意：`fusion_rerank` 会调用 DashScope `qwen3-rerank`，需要 `.env` 中配置 `DASHSCOPE_API_KEY`。默认评测策略不包含 `fusion_rerank`，避免无意产生外部 API 调用和费用。

输出位置：

```text
outputs/evaluations/rag_retriever_eval_YYYYMMDD_HHMMSS.csv
outputs/evaluations/rag_retriever_eval_YYYYMMDD_HHMMSS.md
```

注意：当前入库逻辑基于文件 MD5 增量追加，不会自动清理旧 chunk。如果本地已经存在 `chroma_db/` 和 `md5.text`，正式评测前应先重建索引，避免旧版 txt 语料混入新知识卡片。

当前 `rag/retriever_evaluation.py` 已支持显式 `--reset-index`，会在评测前清理 `chroma_db/` 和 `md5.text` 并重新加载知识库。不要和 `--skip-load-documents` 同时使用。

Generator 层评测已拆分到独立入口：[Phase 1 Generator Evaluation V1](phase1_generation_evaluation_v1.md)。

## 指标如何实现

当前评测集标注了 `expected_source`、`expected_card_id`、`expected_evidence` 和 `expected_keywords`。如果样本存在 `expected_card_id`，V1 会优先按卡片 ID 判断是否命中；没有卡片 ID 时才回退到标准来源、证据短语和关键词覆盖。

## 检索策略如何实现

### Vector

使用 Chroma 向量库进行纯向量检索，作为普通 RAG baseline。

### BM25

使用 `rank_bm25` 和 `jieba` 对所有入库文档构建关键词检索器，作为词面匹配 baseline。

### Fusion

`fusion` 使用 Reciprocal Rank Fusion 融合 Vector 与 BM25 排名。RRF 不直接混合两种检索器的原始分数，而是按各自排名累加：

```text
score = sum(1 / (rrf_k + rank))
```

这样可以避免向量距离和 BM25 分数尺度不同导致的归一化偏差。

### Fusion Rerank

`fusion_rerank` 是二阶段检索策略：

1. 先使用 RRF 粗召回候选文档。
2. 再调用 DashScope `qwen3-rerank` 对 query 与候选文档重新打分排序。
3. 最终返回 Top-K 文档给 Retriever 评测或 Generator 生成链路。

`config/rag.yml` 中相关配置：

| 字段 | 说明 |
|---|---|
| `reranker_model_name` | 默认 `qwen3-rerank` |
| `reranker_candidate_k` | 进入 Reranker 的候选文档数，默认 20 |
| `reranker_timeout_seconds` | API 超时时间 |
| `reranker_instruction` | Reranker instruction，用于约束客服问答检索目标 |

Reranker 是外部 API 调用，延迟和费用会高于纯本地 RRF。当前失败时会显式报错，不静默回退到 RRF，避免评测指标被伪装成成功。

### Recall@K

判断 Top-K 检索结果中是否出现标准知识卡片。满足条件时，该条样本记为命中。

### MRR

逐条检查 Top-K 文档，找到第一条命中标准卡片的文档，计算其排名倒数。

例如第一条命中则 MRR 为 `1.0`，第二条命中则为 `0.5`。

## 为什么 Retriever V1 不直接上 LLM Judge 或 Ragas

Retriever V1 优先解决“能稳定复现检索评测结果”的问题。LLM Judge 和 Ragas 更适合用于生成答案评估或更复杂的语义级判断：

- 检索链路稳定。
- 输出答案链路稳定。
- 有 trace 可以记录每次工具调用和上下文。
- 有更明确的人工标注样本或标准答案。

因此 `rag/retriever_evaluation.py` 先建立低成本、可运行、可提交的检索评测基线；生成层 LLM Judge 已拆分到 `rag/generation_evaluation.py`，避免检索指标和生成指标混在同一个入口里。

## 本版本边界

- 不评估完整 Agent 的真实工具选择，只评估检索策略。
- 不调用 LLM Judge，不输出语义级事实一致性结论。
- 不输出 Generator 或 Agent 指标，只保留 `Recall@K` 和 `MRR`。
- 当前用 `expected_card_id` 作为主要相关性标准，但还不是最终 chunk-level 标准文档 ID；后续可以继续补充稳定 chunk 标识。
- 不比较历史运行结果；每次报告可作为后续改造前后的对比基线。

## 后续 V2 建议

- 为评测集增加稳定的 chunk-level 标识或 `expected_chunk_evidence` 多证据字段，进一步提升 Recall@K/MRR 可信度。
- 继续增强回答生成链路评测，例如增加人工抽样复核和失败样例聚类。
- 为 Generator 评测切换独立 Judge 模型，减少生成模型与评测模型相同带来的偏置。
- 接入 tracing 后评估真实 Agent 工具调用准确率。
- 引入 Ragas，对 faithfulness、context recall、answer relevancy 做更标准的评估。

## 可写进简历的阶段成果

> 构建 Retriever 评测闭环 V1，基于业务 hard set 对 Vector、BM25、RRF Fusion 与 qwen3-rerank 二阶段检索进行 Recall@K 与 MRR 评估，并自动输出 CSV/Markdown 报告，使检索优化从经验调参转为指标驱动。
