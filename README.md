# 智扫通智能客服系统

智扫通智能客服系统是一个面向扫地机器人和扫拖一体机器人场景的智能客服 Agent。用户可以通过聊天界面咨询产品知识、使用方法、故障排查、维护保养、天气适配、品牌对比，也可以生成个人设备使用报告。

系统基于 LangChain Agent 构建，结合本地知识库检索、动态联网检索、天气查询、用户设备数据查询和工具调用日志，提供更完整的客服问答体验。

## 功能

| 功能 | 说明 |
|---|---|
| 产品知识问答 | 回答扫地机器人、扫拖一体机器人的功能、使用、维护、故障和选购问题。 |
| 本地知识库检索 | 从项目内产品资料中检索相关内容，支持向量检索、BM25、RRF 融合和 Reranker 重排。 |
| 联网信息补充 | 当问题涉及最新产品、品牌对比或本地知识不足时，调用 WebSearch 获取外部网页信息并附带来源。 |
| 天气适配建议 | 根据指定城市天气、湿度、风力等信息，判断是否适合使用扫地或拖地功能。 |
| 使用报告生成 | 根据用户 ID 和月份查询设备使用记录，生成清洁效率、耗材状态和维护建议。 |
| 工具调用日志 | 记录工具调用输入、输出摘要、耗时、成功状态、异常和拦截原因，便于排查问题。 |
| RAG 评测 | 提供检索和生成评测脚本，用于比较不同检索策略的效果。 |

## 使用示例

可以在聊天界面中尝试以下问题：

```text
小户型用户应该怎么选扫地机器人？
扫地机器人主刷缠绕严重应该怎么处理？
深圳今天适合使用扫拖一体机器人的拖地功能吗？
帮我查询用户 1001 在 2025-01 的扫地机器人使用报告。
2026 年扫地机器人主流品牌对比有哪些变化？
```

## 项目结构

```text
.
├── app.py                         # Streamlit 聊天入口
├── agent/
│   ├── react_agent.py             # Agent 创建与流式输出
│   ├── hooks/                     # 工具调用日志、策略、脱敏和报表
│   └── tools/                     # LangChain 工具、核心工具层、SQLite 数据访问
├── rag/
│   ├── rag_service.py             # 本地 RAG 问答链路
│   ├── vector_store.py            # Chroma 入库和多策略检索
│   ├── retriever_evaluation.py    # 检索评测脚本
│   └── generation_evaluation.py   # 生成评测脚本
├── websearch/
│   ├── web_search_service.py      # 动态 Web RAG 主流程
│   ├── cache.py                   # WebSearch TTL 缓存
│   ├── source_filter.py           # URL 标准化、过滤和去重
│   └── fetch_web_content.py       # 网页抓取
├── data/                          # 产品知识库、评测数据、用户设备 seed 数据
├── config/                        # 项目配置
├── prompts/                       # Agent、RAG、报告生成提示词
└── docs/                          # 说明文档
```

## 安装

建议使用 Python 3.11 或以上版本。

```bash
pip install -r requirements.txt
```

## 配置

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

Windows PowerShell 可以使用：

```powershell
Copy-Item .env.example .env
```

按需填写以下配置：

| 配置项 | 用途 |
|---|---|
| `DASHSCOPE_API_KEY` | Qwen 生成模型、Embedding、Reranker |
| `DASHSCOPE_BASE_URL` | DashScope OpenAI 兼容接口地址 |
| `DASHSCOPE_RERANK_API_URL` | DashScope Reranker 接口地址 |
| `SERPER_API_KEY` | WebSearch 搜索接口 |
| `SERPER_API_URL` | Serper 搜索接口地址 |
| `AMAP_KEY` | 高德天气接口 |
| `AMAP_WEATHER_API_URL` | 高德天气 API 地址 |
| `DEEPSEEK_API_KEY` | 生成评测中的 Judge 模型 |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 |

`.env` 不会被提交到 Git。

## 启动

```bash
streamlit run app.py
```

启动后在浏览器中打开 Streamlit 提供的本地地址即可使用。

## 数据说明

- 产品知识库位于 `data/`，包括扫地机器人、扫拖一体机器人、故障排除、维护保养、选购指南等资料。
- 用户设备使用记录 seed 数据位于 `data/external/records.csv`。
- `fetch_external_data` 首次查询时会自动生成 SQLite 数据库 `data/external/user_usage.db`。
- `data/external/user_usage.db` 是运行时文件，不会提交到 Git。
- WebSearch 缓存、Hook 日志、本地向量库也不会提交到 Git。

## 常用命令

### 检索评测

```bash
python -m rag.retriever_evaluation --reset-index --top-k 1 --strategies vector,bm25,fusion,fusion_rerank
```

### 生成评测

```bash
python -m rag.generation_evaluation --strategies vector,fusion_rerank --skip-load-documents
```

### 工具调用日志报表

```bash
python -m agent.hooks.report
```

### 工具层检查

```bash
python -m py_compile agent/tools/core_tools.py agent/tools/user_data_store.py agent/tools/tool_result.py agent/tools/agent_tools.py
python -c "from agent.tools.core_tools import fetch_external_data_core; print(fetch_external_data_core('1001', '2025-01'))"
```

## 文档

- [项目总览](docs/project_overview.md)
- [Retriever 评测说明](docs/phase1_retriever_evaluation_v1.md)
- [Generator 评测说明](docs/phase1_generation_evaluation_v1.md)
- [Hook 生命周期管控](docs/phase2_hook_lifecycle_v1.md)
- [动态 Web RAG](docs/phase3_websearch_v1.md)
- [工具服务层标准化](docs/phase4_tool_service_v1.md)

## 注意事项

- 第一次运行本地 RAG 或评测时，系统会根据 `data/` 中的资料构建本地向量库。
- WebSearch、天气查询和生成评测会调用外部 API，需要提前配置对应 Key。
- 如果修改了知识库数据，建议重新构建本地索引，避免旧数据影响检索结果。
- 日志、缓存、向量库和 SQLite 运行时数据库均已加入 `.gitignore`。
