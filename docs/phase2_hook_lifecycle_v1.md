# Phase 2 Hook 生命周期管控 V1

## 目标

本阶段的目标是把 Agent 的工具调用从“直接执行工具”升级为“可拦截、可审计、可复盘、可统计”的生命周期管控。

一句话理解：

```text
原来：Agent 决定调用工具 -> 工具直接执行 -> 返回结果
现在：Agent 决定调用工具 -> Hook 先检查 -> 允许后执行 -> 成功/失败都记录结构化日志
```

这不是新增一个业务工具，而是在所有工具外面套了一层统一的治理层。

## 整体流程

```text
用户问题
  ↓
React Agent 推理
  ↓
模型决定调用某个工具
  ↓
LangChain middleware 拦截工具调用
  ↓
before_tool_call
  ├─ 生成 / 复用 session_id
  ├─ 生成 / 复用 tool_call_id
  ├─ 检测工具输入是否包含敏感信息
  ├─ 判断工具是否联网、是否高成本、是否访问用户数据
  └─ 写入 before_tool_call JSONL 日志
  ↓
策略判断
  ├─ allowed=True  -> 执行真实工具
  └─ allowed=False -> 直接返回安全拦截消息，不执行真实工具
  ↓
真实工具执行
  ↓
after_tool_call / on_tool_error
  ├─ 成功：记录输出摘要、耗时、状态
  └─ 失败：记录错误类型、错误信息、兜底建议
  ↓
结果返回给 Agent
```

## 代码文件职责

| 文件 | 职责 |
|---|---|
| `agent/tools/agent_tools.py` | 存放真实业务工具，例如 RAG、WebSearch、天气、用户数据、报告上下文工具。 |
| `agent/tools/middleware.py` | Hook 接入 LangChain 的入口，在工具调用前后插入生命周期逻辑。 |
| `agent/hooks/lifecycle.py` | 生命周期编排核心，提供 `before_tool_call`、`after_tool_call`、`on_tool_error`。 |
| `agent/hooks/policy.py` | 工具策略层，定义工具风险等级、是否联网、是否高成本、是否访问用户数据、是否允许敏感输入。 |
| `agent/hooks/sanitizer.py` | 敏感信息检测与日志脱敏，识别身份证号、手机号、银行卡、邮箱、API Key、token 等。 |
| `agent/hooks/recorder.py` | JSONL 审计日志写入层，把每个 Hook 事件追加到本地日志文件。 |
| `agent/hooks/report.py` | Hook 日志统计脚本，按工具聚合调用次数、成功数、失败数、拦截数和平均耗时。 |

## 核心入口：middleware.py

`monitor_tool` 是当前 Hook 系统真正接入 Agent 的地方。

它的逻辑可以简化为：

```python
context = hook_manager.before_tool_call(...)

if not context.decision.allowed:
    return ToolMessage(content=blocked_message)

try:
    result = handler(request)
    hook_manager.after_tool_call(context, result)
    return result
except Exception as error:
    fallback = hook_manager.on_tool_error(context, error)
    return ToolMessage(content=fallback)
```

也就是说，所有工具调用都会先经过 `monitor_tool`，再决定是否真正执行业务工具。

## 生命周期一：before_tool_call

位置：`agent/hooks/lifecycle.py`

工具执行前会先进入 `before_tool_call`。

它主要做四件事：

1. 生成或复用 `session_id`
2. 生成或复用 `tool_call_id`
3. 调用 `evaluate_tool_call` 做策略判断
4. 写入 `before_tool_call` 日志

`session_id` 用来串起一次会话中的多个工具调用。

例如用户一次问题触发了 RAG 和天气两个工具：

```text
session_id = sess_xxx
  ├─ tool_call_id = tool_1 -> rag_summarize
  └─ tool_call_id = tool_2 -> get_weather
```

`tool_call_id` 用来标识某一次具体工具调用，后续定位问题、统计耗时、分析失败原因时都会用到。

## 生命周期二：after_tool_call

位置：`agent/hooks/lifecycle.py`

工具成功执行后进入 `after_tool_call`。

它会记录：

```text
session_id
tool_call_id
tool_name
status=success
latency_ms
input_summary
output_summary
```

其中 `latency_ms` 是从 `before_tool_call` 记录的开始时间计算出来的。

这使得后续可以统计：

```text
哪个工具调用最多
哪个工具平均耗时最高
RAG 工具是否变慢
联网工具是否拖慢整体 Agent
```

## 生命周期三：on_tool_error

位置：`agent/hooks/lifecycle.py`

工具执行失败后进入 `on_tool_error`。

它会记录：

```text
session_id
tool_call_id
tool_name
status=error
error_type
error_message
latency_ms
fallback_message
```

然后返回一个安全兜底消息，而不是把异常堆栈直接暴露给用户。

这样可以避免：

```text
外部 API 报错污染最终回答
网络超时导致 Agent 中断
工具异常信息直接泄露给用户
```

## 策略判断流程

位置：`agent/hooks/policy.py` 和 `agent/hooks/sanitizer.py`

策略判断的核心流程是：

```text
工具入参
  ↓
sanitizer.to_text 转成文本
  ↓
detect_sensitive_types 检测敏感信息
  ↓
get_tool_policy 获取工具策略
  ↓
evaluate_tool_call 生成 HookDecision
  ↓
middleware 根据 allowed 决定是否执行工具
```

当前 `ToolPolicy` 默认：

```python
allow_sensitive_input: bool = False
```

因此所有工具默认都不允许携带敏感输入。只有未来某个工具明确声明 `allow_sensitive_input=True`，才会被允许处理敏感输入。

例如用户请求：

```text
帮我联网查询这个身份证号的信息：110101199001011234
```

如果 Agent 准备调用 `web_search`，Hook 会：

```text
1. 在工具参数中检测到身份证号
2. 读取 web_search 的 ToolPolicy
3. 发现 web_search 没有允许敏感输入
4. 返回 allowed=False
5. middleware 直接返回安全拦截消息，不执行 web_search
```

## JSONL 审计日志

日志位置：

```text
logs/hooks/tool_calls_YYYYMMDD.jsonl
```

一次成功工具调用通常会有两条事件：

```text
before_tool_call
after_tool_call
```

一次失败工具调用通常会有两条事件：

```text
before_tool_call
on_tool_error
```

一次被策略拦截的工具调用会记录：

```text
before_tool_call，status=blocked
```

示例：

```json
{
  "created_at": "2026-06-09T14:30:00.000",
  "stage": "after_tool_call",
  "session_id": "sess_xxx",
  "tool_call_id": "tool_xxx",
  "tool_name": "web_search",
  "status": "success",
  "latency_ms": 1234.5,
  "input_summary": "...",
  "output_summary": "..."
}
```

## 统计命令

```bash
python -m agent.hooks.report
python -m agent.hooks.report --date 20260609
```

统计字段：

| 字段 | 说明 |
|---|---|
| `tool_name` | 工具名 |
| `calls` | 成功和失败的调用总数 |
| `success` | 成功次数 |
| `errors` | 失败次数 |
| `blocked` | 被策略拦截次数 |
| `avg_latency_ms` | 平均耗时 |

## 当前 V1 边界

当前版本已经完成工具调用生命周期管控的核心闭环，但还不是最终生产形态。

当前边界：

```text
1. 敏感信息检测以规则和正则为主，还没有接入专门的 PII 识别模型。
2. 用户 ID、城市、外部数据仍是演示数据，还没有接入真实用户数据库。
3. 权限判断目前是工具级策略，还没有绑定真实登录态和用户角色。
4. 日志目前是 JSONL 文件，还没有接入可视化 dashboard。
5. 可靠性执行层已支持按工具策略配置超时、有限重试、结果校验和熔断；超出恢复边界后返回工具级安全兜底消息，并将重试、失败、熔断和恢复事件写入同一条审计链路。
```

## 可写入简历的一句话

构建策略驱动的 Agent 工具调用可靠性治理层，基于 LangChain middleware 统一编排调用前策略拦截、超时重试、结果校验、熔断降级、调用后结构化审计与敏感信息治理，提升客服 Agent 的异常可恢复性、可观测性和工程化可复盘能力。

## 面试摘要

我在项目中实现了一个工具调用生命周期 Hook 管控层，基于 LangChain middleware 对所有工具调用做统一拦截。

调用前通过 `before_tool_call` 生成 `session_id` 和 `tool_call_id`，并结合工具策略判断是否联网、是否高成本、是否访问用户数据、是否包含身份证号/API Key 等敏感信息。

调用成功后通过 `after_tool_call` 记录工具名、输入摘要、输出摘要、耗时和状态。

工具执行时由 `execute_tool` 按工具策略施加超时和有限重试，仅对可恢复的网络类错误重试；连续失败达到阈值后暂时熔断，避免故障工具被反复调用。工具返回后校验统一结果契约，空结果或失败状态不会继续传给模型。超出恢复边界时通过 `on_tool_error` 返回工具级安全兜底消息，并将每次重试、失败、熔断和降级事件写入结构化 JSONL 审计日志，避免工具异常污染最终回答。

同时我把策略层、脱敏层、日志层和生命周期编排层解耦，后续可以继续扩展权限控制、缓存命中统计、失败分布分析和工具调用报表。
