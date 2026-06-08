# 智扫通智能客服系统

目录名：`zhisaotong-customer-service-agent`

本文档是项目竞争力优化规划，不代表当前所有能力均已完成。当前阶段只整理项目定位、已有能力、改造路线和简历表达，不修改业务代码。

## 项目定位

面向扫地机器人及扫拖一体机器人用户的智能客服 Agent，覆盖售前咨询、产品介绍、使用指导、维护保养、天气适配建议、市场信息补充与个性化报告生成。

目标不是只做一个问答机器人，而是升级为一个可检索、可调用工具、可联网补充、可评测、可观测的客服 Agent 系统。

## 当前已有能力

- Streamlit 聊天入口，项目标题为“智扫通机器人智能客服”。
- 基于 LangChain `create_agent` 构建 ReAct 风格 Agent。
- 工具能力包括本地 RAG 检索、互联网搜索、天气查询、用户 ID 获取、月份获取、外部使用记录读取和报告上下文注入。
- 本地知识库基于 Chroma，已实现向量检索和 BM25 融合检索。
- 动态网络知识补充链路已存在：Serper 搜索、网页抓取、多线程并发抓取、网页内容切分、临时 Chroma 向量库、相关内容检索和 LLM 总结。
- Prompt 中已明确约束 `web_search` 在最新信息、品牌对比、本地知识库不足时调用。

## 当前短板

- 检索质量缺少系统评测闭环，虽然已有 `rag_eval_dataset.csv`，但还没有形成稳定评测报告。
- WebSearch 缺少查询缓存、URL 去重、来源可信度评分、TTL 过期策略和引用输出。
- 当前融合检索已有 BM25 + 向量，但缺少 RRF、Cross-Encoder 或 Qwen/BGE Reranker 的二阶段重排。
- 工具仍是项目内 Python 函数，尚未封装成 MCP Server，外部 Agent 客户端复用性不强。
- 日志有记录，但缺少可观测面板，不能直观看到工具调用、延迟、检索命中和回答质量变化。

## 改造路线

## Claude Code 公开设计可借鉴点

以下内容只参考 Anthropic 官方公开仓库和公开文档，不参考任何非授权泄露材料。Claude Code 公开资料中最值得借鉴的是产品化 Agent 的组织方式：插件化能力包、子 Agent 上下文隔离、hook 生命周期管控、可审计记忆和 MCP 工具生态。

- 插件化能力包：参考 Claude Code plugins 的结构，将客服能力拆成可组合模块，例如 `rag-search`、`web-freshness`、`user-report`、`weather-advisor` 和 `source-auditor`。短期可以先在项目内按模块组织，后期再封装成可复用插件或 MCP 工具包。
- Hook 生命周期管控：参考 PreToolUse/PostToolUse 思路，在工具调用前后增加规则。调用前检查是否真的需要联网、是否命中缓存、是否包含敏感用户信息；调用后记录耗时、来源、错误和工具输出摘要。
- 子 Agent 上下文隔离：将“本地知识库回答”“动态联网补充”“报告生成”“事实来源审查”拆成专职子 Agent，主 Agent 只做意图识别和路由，避免一个上下文里混入过多任务细节。
- 可审计记忆：参考 `CLAUDE.md` 与自动记忆思想，为项目建立人工维护的 `AGENT.md` 或 `PROJECT_MEMORY.md`，记录产品术语、工具边界、常见失败样例和评测结论。记忆必须是可读 Markdown，避免黑箱化。
- 背景监控：参考 background monitors 思路，监控日志、WebSearch 失败率、缓存命中率、评测指标变化，并在异常时生成可读报告。

映射到本项目后，优先级建议是：先做 Hook + 评测，再做缓存 + Reranker，最后做 MCP/插件化。这样既能保持项目规模可控，也能把“Agent 工程化”讲清楚。

### Phase 1：评测闭环

目标：让项目从“能回答”升级到“能证明回答质量”。

- 基于现有 `rag_eval_dataset.csv` 构建自动评测脚本。
- 评估指标：检索 Recall@K、MRR、回答相关性、事实一致性、上下文利用率、工具调用准确率。
- 输出 CSV/Markdown 评测报告，记录每次改造前后的指标变化。
- 将评测命令写入 README，方便面试时演示。

可写进简历的成果：

> 构建 RAG 评测闭环，基于业务问答集评估检索召回、回答事实性和工具调用准确率，使系统优化从经验调参转为指标驱动。

### Phase 2：检索增强

目标：提升知识库问答的稳定性和专业度。

- 将现有向量 + BM25 融合升级为 RRF 融合。
- 增加二阶段 Reranker，例如 `BAAI/bge-reranker-v2-m3` 或 `Qwen/Qwen3-Reranker-0.6B`。
- 尝试 Contextual Retrieval：为 chunk 添加文档标题、章节、产品型号、适用场景等上下文，再进行 embedding 和 BM25 建索引。
- 保留向量检索、BM25、融合检索、融合 + Reranker 的对比评测。

可写进简历的成果：

> 设计 Hybrid Retrieval + RRF + Reranker 二阶段检索链路，并通过离线评测对比向量检索、BM25 与融合检索效果。

### Phase 3：动态网络知识补充强化

目标：让 WebSearch 从“能搜”升级为“可控、可追溯、低成本”。

- 为 `web_search(query)` 增加缓存：`query + top_k + date_bucket` 生成 hash key，缓存搜索结果和网页摘要。
- 增加 TTL 策略：市场信息类短缓存，通用知识类长缓存。
- URL 去重，过滤 PDF、低质量页面、重复域名。
- 为返回结果附带来源 URL、网页标题、抓取时间和摘要。
- 对外部网页内容构建临时向量库时记录检索来源，最终回答中输出引用。

可写进简历的成果：

> 集成 Serper 与多线程网页抓取，构建带缓存、去重和来源追踪的动态 Web RAG，补充本地知识库缺失的最新产品对比信息。

### Phase 4：MCP 工具化

目标：把项目内工具升级为标准 Agent 工具服务。

- 将 `rag_summarize`、`web_search`、`get_weather`、`fetch_external_data` 封装成 MCP tools。
- 暴露资源：产品知识库摘要、用户报告字段说明、工具调用规范。
- 支持外部 MCP Client 或其他 Agent 复用这些工具。
- 为工具添加输入校验、错误结构化返回和调用日志。

可写进简历的成果：

> 将客服 Agent 工具封装为 MCP Server，实现 RAG、WebSearch、天气和用户报告工具的标准化注册与跨 Agent 复用。

### Phase 5：可观测与演示

目标：让项目更像生产级应用，而不是课堂 Demo。

- 记录每轮对话的工具调用链、耗时、检索命中文档、来源 URL 和最终回答。
- 增加 Hook 风格的工具调用审计：PreToolUse 负责权限、缓存、联网必要性判断；PostToolUse 负责结构化日志和失败归因。
- 增加简单管理页或报告页，展示工具调用次数、缓存命中率、平均延迟、RAG 命中率。
- 增加典型 demo case：售前选购、故障排查、天气适配、品牌对比、个人报告生成。

## 建议后的简历项目标题

智扫通智能客服系统：基于 MCP 与 Hybrid RAG 的扫地机器人智能客服 Agent

## 技术参考

- Anthropic Contextual Retrieval: https://www.anthropic.com/research/contextual-retrieval
- Anthropic Claude Code public repository: https://github.com/anthropics/claude-code
- Claude Code Plugins: https://code.claude.com/docs/en/plugins
- Claude Code Hooks: https://code.claude.com/docs/en/agent-sdk/hooks
- Claude Code Memory: https://code.claude.com/docs/zh-CN/memory
- Ragas Metrics: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
- Model Context Protocol: https://modelcontextprotocol.io/docs/learn/architecture
- Qwen3 Embedding and Reranker: https://arxiv.org/abs/2506.05176
