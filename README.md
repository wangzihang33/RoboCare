# 智扫通智能客服系统

目录名：`zhisaotong-customer-service-agent`

本文档记录项目当前已经完成的竞争力改造、下一步工程化路线，以及可用于简历和面试的表达。

## 项目定位

面向扫地机器人及扫拖一体机器人用户的智能客服 Agent，覆盖售前咨询、产品介绍、使用指导、维护保养、故障排查、天气适配建议、市场信息补充与个性化报告生成。

目标不是只做一个问答机器人，而是升级为一个可检索、可调用工具、可联网补充、可评测、可观测、可审计的客服 Agent 系统。

## 当前已有能力

- Streamlit 聊天入口，项目标题为“智扫通机器人智能客服”。
- 基于 LangChain `create_agent` 构建 ReAct 风格 Agent。
- 工具能力包括本地 RAG 检索、互联网搜索、天气查询、用户 ID 获取、月份获取、外部使用记录读取和报告上下文注入。
- 本地知识库基于 Chroma，已实现向量检索、BM25 检索、RRF 融合检索和 DashScope `qwen3-rerank` 二阶段重排。
- 已构建 Retriever 与 Generator 双层评测闭环，支持 CSV/Markdown 报告输出和多策略指标对比。
- 已实现工具调用生命周期 Hook 管控，统一记录 `session_id`、`tool_call_id`、输入输出摘要、耗时、成功/失败状态和敏感信息拦截结果。
- 动态网络知识补充链路已经存在：Serper 搜索、网页抓取、多线程并发抓取、网页内容切分、临时 Chroma 向量库、相关内容检索和 LLM 总结。
- Prompt 中已明确约束 `web_search` 在最新信息、品牌对比、本地知识库不足时调用。

## 环境安装

```bash
pip install -r requirements.txt
```

## 当前完成情况与下一步重点

- Phase 1 已完成：评测驱动的多阶段 RAG 检索优化，形成 `vector`、`bm25`、`fusion`、`fusion_rerank` 的可量化对比。
- Phase 2 已完成：工具调用生命周期 Hook 管控 V1，形成工具调用前拦截、调用后审计、异常兜底和日志统计闭环。
- Phase 3 已完成：动态 Web RAG V1，形成 TTL 缓存、URL 治理、来源追踪、引用输出和 Hook 指标统计闭环。
- Phase 4 已完成：工具服务层标准化 V1，完成核心工具解耦、SQLite 用户数据接入、真实天气错误处理和随机工具清理。
- 项目已形成 4 个可用于简历和面试展开的核心竞争点：评测驱动 RAG 优化、Hook 生命周期治理、动态 Web RAG、工具服务层标准化。

## 改造路线

## Claude Code 公开设计可借鉴点

以下内容只参考 Anthropic 官方公开仓库和公开文档，不参考任何非授权泄露材料。Claude Code 公开资料中最值得借鉴的是产品化 Agent 的组织方式：插件化能力包、子 Agent 上下文隔离、Hook 生命周期管控、可审计记忆和标准化工具治理。

- 插件化能力包：参考 Claude Code plugins 的结构，将客服能力拆成可组合模块，例如 `rag-search`、`web-freshness`、`user-report`、`weather-advisor` 和 `source-auditor`。短期可以先在项目内按模块组织，后期再视真实复用需求封装成可复用插件或外部工具服务。
- Hook 生命周期管控：参考 PreToolUse/PostToolUse 思路，在工具调用前后增加规则。调用前检查是否真的需要联网、是否包含敏感用户信息、是否访问用户数据、是否属于高成本外部调用；调用后记录耗时、错误、输入输出摘要和工具调用状态。
- 子 Agent 上下文隔离：将“本地知识库回答”“动态联网补充”“报告生成”“事实来源审查”拆成专职子 Agent，主 Agent 只做意图识别和路由，避免一个上下文里混入过多任务细节。
- 可审计记忆：参考 `CLAUDE.md` 与自动记忆思想，为项目建立人工维护的 `AGENT.md` 或 `PROJECT_MEMORY.md`，记录产品术语、工具边界、常见失败样例和评测结论。记忆必须是可读 Markdown，避免黑箱化。
- 背景监控：参考 background monitors 思路，监控日志、WebSearch 失败率、缓存命中率、评测指标变化，并在异常时生成可读报告。

映射到本项目后，评测闭环、RRF 融合、Reranker、Hook 生命周期管控 V1、动态 Web RAG 强化 V1 和工具服务层标准化 V1 已经完成。这样既能保持项目规模可控，也能把“Agent 工程化”主线讲清楚。

### Phase 1：评测驱动的多阶段 RAG 检索优化（已完成）

目标：让项目从“能回答”升级到“能证明回答质量”，并通过指标证明多阶段检索优化确实提升效果。

- 重建 `data/rag_eval_dataset.csv` 为 60 条 card-level hard set，覆盖 6 个 txt 知识文件。
- 将 6 个 txt 知识文件整理为 90 张客服知识卡片，其中 60 张为评测标准卡，30 张为 `*-DIST-*` 近邻干扰卡。
- 入库时写入 `card_id` metadata，使评测从 source 级升级为 card 级，避免“命中同一文件就算正确”的虚高指标。
- Retriever 评测支持 `Recall@K`、`MRR`，并自动输出 CSV/Markdown 报告。
- Generator 评测支持 `Faithfulness`、`Answer Relevance`、`Pass Rate`，默认使用 Qwen 生成、DeepSeek Judge。
- 检索策略已支持 `vector`、`bm25`、`fusion`、`fusion_rerank`：
  - `vector`：纯向量检索 baseline。
  - `bm25`：关键词检索 baseline。
  - `fusion`：Vector + BM25 的 RRF 融合。
  - `fusion_rerank`：RRF 粗召回 + DashScope `qwen3-rerank` 二阶段重排。

核心评测命令：

```bash
python -m rag.retriever_evaluation --top-k 1 --strategies vector,fusion_rerank --skip-load-documents
python -m rag.retriever_evaluation --top-k 3 --strategies vector,fusion_rerank --skip-load-documents
python -m rag.generation_evaluation --strategies vector,fusion_rerank --skip-load-documents
```

全量重建索引评测：

```bash
python -m rag.retriever_evaluation --reset-index --top-k 1 --strategies vector,bm25,fusion,fusion_rerank
python -m rag.generation_evaluation --reset-index --strategies vector,fusion_rerank
```

Generator 评测会调用 LLM 生成答案和 Judge 打分。默认生成模型使用 DashScope/Qwen，需要 `.env` 中配置 `DASHSCOPE_API_KEY`；默认 Judge 使用 DeepSeek，需要配置 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL`。如果 DashScope 默认生成模型额度不足，可通过 `--generator-provider deepseek` 临时切换生成模型。

输出目录：

```text
outputs/evaluations/
```

注意：如果本地已经加载过旧版知识库，正式跑评测前应重建 `chroma_db/` 和 `md5.text`，避免旧 chunk 污染新数据指标。

Retriever 评测说明文档：[Phase 1 Retriever Evaluation V1](docs/phase1_retriever_evaluation_v1.md)

生成评测说明文档：[Phase 1 Generator Evaluation V1](docs/phase1_generation_evaluation_v1.md)

可写进简历的成果：

> 构建评测驱动的多阶段 RAG 检索优化链路，基于 60 条业务 hard set 对 Vector、BM25、RRF Fusion 与 qwen3-rerank 进行 Recall@K、MRR、Faithfulness 和 Answer Relevance 评估，使系统优化从经验调参转为指标驱动。

### Phase 2：Hook 生命周期管控（V1 已完成）

目标：将粗糙日志升级为可审计的工具调用生命周期，让 Agent 每次调用工具都能被记录、复盘和约束。

- 已建立统一 Hook 管理模块，抽象 `before_tool_call`、`after_tool_call`、`on_tool_error` 生命周期。
- PreToolUse：检查工具是否需要联网、是否包含敏感信息、是否访问用户数据、是否属于高成本外部调用。
- PostToolUse：记录 `session_id`、`tool_call_id`、工具名、输入摘要、输出摘要、耗时、成功/失败状态。
- Error Hook：统一捕获异常、错误类型和降级提示，避免工具失败直接污染最终回答。
- 已为 RAG、WebSearch、天气、用户数据、报告工具统一接入 Hook，输出结构化 JSONL 日志。
- 已提供 Hook 日志统计脚本，支持统计工具调用次数、成功次数、失败次数、拦截次数和平均延迟。

Hook 日志位置：

```text
logs/hooks/tool_calls_YYYYMMDD.jsonl
```

Hook 统计命令：

```bash
python -m agent.hooks.report
python -m agent.hooks.report --date 20260609
```

Hook 说明文档：[Phase 2 Hook Lifecycle V1](docs/phase2_hook_lifecycle_v1.md)

可写进简历的成果：

> 构建 Agent 工具调用生命周期 Hook 管控层，基于 LangChain middleware 统一实现工具调用前策略拦截、调用后结构化审计、异常兜底与敏感信息治理，提升客服 Agent 的可观测性、安全性和工程化可复盘能力。

### Phase 3：动态网络知识补充强化（V1 已完成）

目标：构建可控、可追溯、低成本的动态 Web RAG 链路。

- 已为 `web_search(query)` 配置本地 JSON 缓存：基于 `query + top_k + date_bucket + cache_version` 生成 hash key，缓存最终回答、引用来源和 trace。
- 已设计 TTL 策略：市场、价格、排行、品牌对比类短缓存，故障、维护、使用指导类长缓存。
- 已实现 URL 标准化、追踪参数清理、PDF/图片等非网页资源过滤、重复 URL 去重和同域名限流。
- 已将 Serper 搜索结果结构化为 `SearchResult`，并将网页抓取结果结构化为 `FetchedPage`。
- 已对外部网页内容构建临时向量库时记录来源 metadata，最终回答稳定追加参考来源列表。
- 已将缓存命中率、来源数量、过滤后 URL 数、抓取成功数和失败数接入 Hook 日志统计。

WebSearch 说明文档：[Phase 3 WebSearch V1](docs/phase3_websearch_v1.md)

可写进简历的成果：

> 构建动态 Web RAG 证据链，基于 TTL 缓存、URL 去重过滤、网页来源 metadata 追踪、引用输出和 Hook 指标统计，实现可追溯、低成本、可观测的联网知识补充链路。

### Phase 4：工具服务层标准化（V1 已完成）

目标：把项目内工具整理为可维护、可测试、可审计的 LangChain 工具服务层。

- 已将 `rag_summarize`、`web_search`、`get_weather`、`fetch_external_data` 的核心逻辑下沉到 `core_tools.py`。
- 已保留 LangChain `@tool` 注册方式，由 `agent_tools.py` 负责薄包装和文本适配。
- 已为核心工具添加输入校验、统一结构化返回和 Hook 调用日志。
- 已将 `fetch_external_data` 从 CSV 内存字典改为 SQLite 用户设备数据访问层。
- 已移除随机 `get_user_id`、`get_current_month`、`get_user_location` 工具；`get_weather` 必须显式传入城市，`fill_context_for_report` 保留为 Agent 内部流程工具。
- 已将天气工具从“失败时返回虚拟天气”改为真实接口查询和结构化错误返回。

工具层验证命令：

```bash
python -m py_compile agent/tools/core_tools.py agent/tools/user_data_store.py agent/tools/tool_result.py agent/tools/agent_tools.py agent/tools/middleware.py agent/react_agent.py
python -c "from agent.tools.core_tools import fetch_external_data_core; print(fetch_external_data_core('1001', '2025-01'))"
```

工具服务层说明文档：[Phase 4 Tool Service Layer V1](docs/phase4_tool_service_v1.md)

可写进简历的成果：

> 重构 Agent 工具服务层，将 RAG、动态 Web RAG、真实天气查询和用户设备数据查询统一封装为可校验、可审计、可测试的 LangChain 工具，并将用户报告数据接入 SQLite，移除随机虚拟工具，提升系统工程可信度和业务可复盘能力。

## 建议后的简历项目标题

智扫通智能客服系统：基于 Hybrid RAG 与工具生命周期治理的扫地机器人智能客服 Agent

## 技术参考

- Anthropic Contextual Retrieval: https://www.anthropic.com/research/contextual-retrieval
- Anthropic Claude Code public repository: https://github.com/anthropics/claude-code
- Claude Code Plugins: https://code.claude.com/docs/en/plugins
- Claude Code Hooks: https://code.claude.com/docs/en/agent-sdk/hooks
- Claude Code Memory: https://code.claude.com/docs/zh-CN/memory
- Ragas Metrics: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
- Model Context Protocol: https://modelcontextprotocol.io/docs/learn/architecture
- Qwen3 Embedding and Reranker: https://arxiv.org/abs/2506.05176
