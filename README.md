# 智扫通智能客服系统

面向扫地机器人与扫拖一体机器人场景的智能客服 Agent。系统基于 LangChain Agent、本地 Hybrid RAG、动态 Web RAG、工具生命周期 Hook 和工具服务层标准化能力，覆盖售前咨询、产品知识问答、故障排查、维护保养、天气适配建议、市场信息补充和个性化使用报告生成。

这个项目不是单纯的知识库问答 Demo，而是围绕“客服 Agent 如何可靠回答、如何证明效果、如何治理工具调用、如何补充新鲜知识”构建的一套工程化实践。

## 核心功能

| 功能 | 说明 |
|---|---|
| 本地知识库问答 | 基于 Chroma、向量检索、BM25、RRF 融合和 `qwen3-rerank`，回答产品知识、使用指导、故障排查和维护保养问题。 |
| 动态 Web RAG | 基于 Serper 搜索、多线程网页抓取、URL 去重过滤、TTL 缓存、临时向量检索和引用输出，补充最新产品和品牌对比信息。 |
| 天气适配建议 | 通过高德天气 API 查询真实天气，根据城市、湿度、风力等信息给出扫地/拖地机器人使用建议。 |
| 用户使用报告 | 使用 SQLite 查询脱敏用户设备使用记录，生成清洁效率、耗材状态、同类对比和保养建议报告。 |
| 工具调用治理 | 基于 Hook 生命周期记录工具调用前策略、调用后审计、异常兜底、敏感信息拦截和 JSONL 日志。 |
| RAG 评测闭环 | 支持 Retriever 和 Generator 双层评测，输出 CSV/Markdown 报告，用指标比较不同检索策略效果。 |

## 技术亮点

- **评测驱动的 RAG 优化**：构建 60 条业务 hard set，使用 `Recall@K`、`MRR`、`Faithfulness`、`Answer Relevance` 对 Vector、BM25、RRF Fusion 和 Reranker 进行量化评估。
- **多阶段检索链路**：本地知识库支持向量检索、BM25、RRF 融合检索和 DashScope `qwen3-rerank` 二阶段重排。
- **动态 Web RAG 证据链**：联网搜索结果经过 URL 标准化、低质量过滤、重复域名控制、网页抓取、临时向量检索和来源 metadata 追踪，最终回答附带真实引用。
- **工具生命周期 Hook**：统一记录 `session_id`、`tool_call_id`、工具名、输入摘要、输出摘要、耗时、状态、异常和拦截原因。
- **工具服务层标准化**：将 RAG、WebSearch、天气查询和用户数据查询封装为可校验、可测试、可审计的 LangChain 工具，统一结构化错误和文本适配。
- **敏感信息治理**：工具调用前检测身份证号、手机号、API Key 等敏感输入，结合工具策略决定是否允许调用。

## 系统架构

```text
Streamlit Chat UI
        |
        v
LangChain ReAct Agent
        |
        +-- rag_summarize --------> Local Hybrid RAG -> Chroma / BM25 / RRF / Reranker
        |
        +-- web_search -----------> Serper -> URL Filter -> Web Crawler -> Temp Chroma -> LLM Summary
        |
        +-- get_weather ----------> AMap Weather API
        |
        +-- fetch_external_data --> SQLite user_usage_records
        |
        +-- fill_context_for_report -> Report Prompt Switch
        |
        v
Tool Hook Lifecycle
before_tool_call / after_tool_call / on_tool_error / websearch_trace
```

## 项目结构

```text
.
├── app.py                         # Streamlit 聊天入口
├── agent/
│   ├── react_agent.py             # LangChain Agent 构建与流式输出
│   ├── hooks/                     # 工具生命周期 Hook、策略、脱敏、日志和报表
│   └── tools/                     # LangChain 工具、核心工具层、SQLite 数据访问层
├── rag/
│   ├── vector_store.py            # Chroma 入库、Vector/BM25/RRF/Reranker 检索
│   ├── rag_service.py             # 本地 RAG 回答链路
│   ├── retriever_evaluation.py    # Retriever 评测
│   └── generation_evaluation.py   # Generator 评测
├── websearch/
│   ├── web_search_service.py      # 动态 Web RAG 主编排
│   ├── cache.py                   # TTL 缓存
│   ├── source_filter.py           # URL 标准化与过滤
│   ├── fetch_web_content.py       # 多线程网页抓取
│   └── retrieval.py               # 网页临时向量检索
├── data/                          # 产品知识库、评测集、用户设备 seed 数据
├── config/                        # RAG、Agent、WebSearch 配置
├── prompts/                       # 主 Agent、RAG、报告生成提示词
└── docs/                          # 各模块说明文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并按需填写：

```bash
AMAP_KEY=your_amap_key
AMAP_WEATHER_API_URL=https://restapi.amap.com/v3/weather/weatherInfo
SERPER_API_URL=https://google.serper.dev/search
SERPER_API_KEY=your_serper_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_RERANK_API_URL=https://dashscope.aliyuncs.com/compatible-api/v1/reranks
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

说明：

- `DASHSCOPE_API_KEY` 用于 Qwen 生成模型、Embedding 和 Reranker。
- `SERPER_API_KEY` 用于动态 Web RAG 的联网搜索。
- `AMAP_KEY` 用于真实天气查询。
- `DEEPSEEK_API_KEY` 用于 Generator 评测中的 Judge 模型。
- `.env` 不会被提交到 Git。

### 3. 启动应用

```bash
streamlit run app.py
```

## 常用命令

### Retriever 评测

```bash
python -m rag.retriever_evaluation --reset-index --top-k 1 --strategies vector,bm25,fusion,fusion_rerank
python -m rag.retriever_evaluation --top-k 3 --strategies vector,fusion_rerank --skip-load-documents
```

### Generator 评测

```bash
python -m rag.generation_evaluation --strategies vector,fusion_rerank --skip-load-documents
```

### Hook 报表

```bash
python -m agent.hooks.report
python -m agent.hooks.report --date 20260609
```

### 工具层验证

```bash
python -m py_compile agent/tools/core_tools.py agent/tools/user_data_store.py agent/tools/tool_result.py agent/tools/agent_tools.py
python -c "from agent.tools.core_tools import fetch_external_data_core; print(fetch_external_data_core('1001', '2025-01'))"
```

## 评测结果示例

在 60 条业务 hard set 上，`top_k=1` 的 Retriever 评测示例：

```text
bm25:          recall@k=0.5500, mrr=0.5500
vector:        recall@k=0.8500, mrr=0.8500
fusion_rerank: recall@k=0.9500, mrr=0.9500
```

Generator 评测示例：

```text
fusion:        faithfulness=1.0000, answer_relevance=0.9000, pass_rate=0.9000
fusion_rerank: faithfulness=1.0000, answer_relevance=1.0000, pass_rate=1.0000
```

评测报告会输出到：

```text
outputs/evaluations/
```

## 文档索引

- [项目总览](docs/project_overview.md)
- [Retriever 评测说明](docs/phase1_retriever_evaluation_v1.md)
- [Generator 评测说明](docs/phase1_generation_evaluation_v1.md)
- [Hook 生命周期管控](docs/phase2_hook_lifecycle_v1.md)
- [动态 Web RAG](docs/phase3_websearch_v1.md)
- [工具服务层标准化](docs/phase4_tool_service_v1.md)

## 数据与安全

- `.env`、日志、缓存、本地向量库和 SQLite 运行时数据库均已加入 `.gitignore`。
- `data/external/records.csv` 是脱敏模拟用户设备数据，用于初始化 SQLite。
- `data/external/user_usage.db` 会在首次查询时自动生成，不提交到仓库。
- 工具调用日志写入 `logs/hooks/tool_calls_YYYYMMDD.jsonl`，可用于审计和指标统计。

## 简历描述参考

> 智扫通智能客服系统：基于 Hybrid RAG、动态 Web RAG 与工具生命周期治理的扫地机器人智能客服 Agent。项目构建评测驱动的多阶段检索优化链路，结合 RRF、Reranker、WebSearch TTL 缓存、来源追踪、Hook 审计和 SQLite 用户数据工具层，实现客服问答、最新信息补充、天气适配和个性化报告生成。
