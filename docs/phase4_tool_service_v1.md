# Phase 4 Tool Service Layer V1

版本目标：将 Agent 工具层从“直接散落在 LangChain 工具函数中”整理为可维护、可测试、可审计的统一工具服务层，并将虚拟数据工具改造成更接近真实业务系统的实现。

## 一句话理解

本版本没有引入额外协议层，而是在现有 LangChain Agent 内部完成工具服务层标准化：统一输入校验、结构化返回、真实天气错误处理、SQLite 用户设备数据查询和 Hook 审计接入。

## 改造边界

对外仍然使用 LangChain `@tool` 注册工具：

```text
rag_summarize
web_search
get_weather
fetch_external_data
fill_context_for_report
```

其中 `fill_context_for_report` 只负责报告场景的 prompt switch，是 Agent 内部流程工具。

以下随机工具已移除：

| 工具 | 处理方式 | 原因 |
|---|---|---|
| `get_user_id` | 移除 | 随机用户 ID 不符合真实业务语义 |
| `get_current_month` | 移除 | 报告月份应由用户显式提供或由业务上下文传入 |
| `get_user_location` | 移除 | 天气工具要求显式 city 参数，缺少城市时由 Agent 追问 |

## 代码结构

| 文件 | 作用 |
|---|---|
| `agent/tools/core_tools.py` | 核心业务工具层，封装 RAG、WebSearch、天气查询和用户设备数据查询。 |
| `agent/tools/tool_result.py` | 统一结构化返回格式，包含 `ok`、`data`、`error`、`trace_id`、`meta`。 |
| `agent/tools/user_data_store.py` | SQLite 用户设备数据访问层，从 CSV seed 初始化本地数据库。 |
| `agent/tools/agent_tools.py` | LangChain `@tool` 包装层，将结构化结果转换为适合 LLM 使用的中文文本。 |
| `agent/tools/middleware.py` | 保留工具生命周期 Hook、WebSearch trace 记录和 report prompt switch。 |

## 核心设计

### 1. 工具业务逻辑下沉到 core_tools.py

`agent_tools.py` 不再直接写业务逻辑，而是只做 LangChain 包装。

例如：

```python
@tool(...)
def get_weather(city: str) -> str:
    return result_to_text(get_weather_core(city))
```

这样做的好处是：

- 工具核心逻辑可以独立测试。
- LangChain 包装层保持很薄。
- 未来如果要接 API、任务队列或其他 Agent runtime，不需要重写业务逻辑。

### 2. 统一结构化返回

所有核心工具都返回统一结构：

```json
{
  "ok": true,
  "tool_name": "fetch_external_data",
  "trace_id": "fetch_external_data_xxx",
  "data": {},
  "error": null,
  "meta": {}
}
```

失败时：

```json
{
  "ok": false,
  "tool_name": "get_weather",
  "trace_id": "get_weather_xxx",
  "data": null,
  "error": {
    "type": "invalid_input",
    "message": "city 不能为空，请提供标准城市名称",
    "details": {}
  },
  "meta": {}
}
```

LangChain Agent 不直接消费 JSON，而是通过 `result_to_text()` 转成中文文本，让 LLM 更容易整合到回答中。

### 3. 天气工具真实化

`get_weather(city)` 必须显式传入城市。

关键变化：

- 不再调用随机 `get_user_location`。
- 不再返回“虚拟晴天 26 度”。
- 未提供城市时返回结构化错误。
- 未配置高德 Key 或接口失败时返回结构化错误。
- 成功时返回真实天气字段，包括城市、天气、温度、湿度、风向、风力和发布时间。

这让工具输出更可信，也避免 Agent 把兜底假数据当事实回答用户。

### 4. 用户设备数据接 SQLite

`fetch_external_data(user_id, month)` 从本地 SQLite 查询用户设备使用记录。

数据库路径：

```text
data/external/user_usage.db
```

首次查询时自动从 CSV seed 初始化：

```text
data/external/records.csv
```

表结构：

```text
user_usage_records
- user_id
- month
- feature_profile
- cleaning_efficiency
- consumables_status
- comparison_summary
```

查询使用参数化 SQL：

```sql
WHERE user_id = ? AND month = ?
```

这比内存字典更接近真实业务系统，也便于后续替换成 CRM、IoT 设备平台、售后工单库或用户画像库。

### 5. 报告流程保留内部开关

`fill_context_for_report` 保留在 LangChain Agent 内部。

它的作用不是查询业务数据，而是通知 middleware：

```text
当前进入报告生成场景，需要切换 report prompt
```

所以它不应该被当成业务工具扩展，只保留在 Agent 内部流程中即可。

## LangChain 调用流程

以天气问题为例：

```text
用户：深圳今天适合拖地吗？
  ↓
Agent 判断需要天气信息
  ↓
调用 get_weather(city="深圳")
  ↓
agent_tools.py 调用 get_weather_core
  ↓
core_tools.py 校验 city 并请求高德天气 API
  ↓
tool_result.py 将结构化结果转成中文文本
  ↓
middleware 写入 Hook 日志
  ↓
Agent 结合天气和扫地机器人知识生成回答
```

以报告问题为例：

```text
用户：帮我生成用户 1001 在 2025-01 的使用报告
  ↓
Agent 确认 user_id 和 month 完整
  ↓
调用 fill_context_for_report
  ↓
middleware 切换 report prompt
  ↓
调用 fetch_external_data(user_id="1001", month="2025-01")
  ↓
SQLite 查询 user_usage_records
  ↓
返回结构化使用记录并转成中文文本
  ↓
Agent 生成 Markdown 报告
```

## 验证命令

编译检查：

```bash
python -m py_compile agent/tools/core_tools.py agent/tools/user_data_store.py agent/tools/tool_result.py agent/tools/agent_tools.py agent/tools/middleware.py agent/react_agent.py
```

验证 SQLite 查询：

```bash
python -c "from agent.tools.core_tools import fetch_external_data_core; print(fetch_external_data_core('1001', '2025-01'))"
```

验证天气输入校验：

```bash
python -c "from agent.tools.core_tools import get_weather_core; print(get_weather_core(''))"
```

## 可写入简历的一句话

构建可校验、可审计、可测试的 Agent 工具服务层，统一封装 RAG、动态 Web RAG、真实天气查询和 SQLite 用户设备数据查询，并通过结构化错误与 Hook 日志提升客服 Agent 的工程可信度和业务可复盘能力。

## 面试摘要

该项目的工具服务层由 `core_tools.py`、`tool_result.py`、`user_data_store.py` 和 LangChain `@tool` 包装层组成。`core_tools.py` 负责 RAG、动态 Web RAG、天气查询和用户设备数据查询的核心业务逻辑；LangChain `@tool` 负责工具注册和文本适配；核心工具统一返回 `ok/data/error/trace_id/meta`，便于测试、日志审计和错误处理。

天气工具以显式城市作为输入，通过高德天气 API 返回真实天气字段；报告工具以 `user_id` 和 `month` 作为输入，通过 SQLite 查询脱敏用户设备使用记录，并使用参数化 SQL 保证查询边界清晰。所有工具调用继续接入 Hook 生命周期日志，使工具输入、输出摘要、耗时、成功状态和异常信息都可以被审计和复盘。
