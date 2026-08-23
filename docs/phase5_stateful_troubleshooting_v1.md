# 多轮故障诊断与状态管理

## 目标

故障诊断不能把每一轮都当成独立问题。系统需要知道已经确认了什么、执行过哪些步骤、每一步结果如何，以及何时应停止自动排查并生成人工交接信息。

本阶段在现有单 Agent 架构内增加受控诊断工作流，不引入多个协作 Agent。

## 分层记忆

```text
短期对话窗口
  最近 8 条 user/assistant 消息；新诊断从其中的 user 消息补齐缺失槽位
        ↓
结构化诊断状态
  设备型号、故障现象、错误码、当前步骤、执行结果、风险项
        ↓
长期服务记录
  已解决案例和转人工工单，按 session/case 持久化到 SQLite
```

`ReactAgent` 将有界历史传给诊断引擎。新案例先处理当前消息，再从最近的用户消息中补齐缺失的设备型号、错误码或故障现象；助手消息不作为用户事实，历史风险词也不会被继承到新案例。活动案例后续依赖 SQLite 中的结构化状态继续推进。长期层不保存全部原始聊天，只保存完成服务所需的结构化事实和案例摘要。

## 状态模型

一次诊断案例包含：

- `case_id` 与稳定的 `session_id`；
- `device_model`、`symptom_code`、`symptom_text`、`error_code`；
- `attempts`：步骤 ID、操作说明和 `pending/failed/success` 结果；
- `risk_flags`、`evidence_ids` 和 `escalation_reason`；
- `turn_count`、创建时间和更新时间。

状态迁移如下：

```text
COLLECTING
  ├─ 识别故障现象 → WAITING_FEEDBACK
  ├─ 两轮仍无法确认 → ESCALATED
  └─ 高风险/主动人工 → ESCALATED

WAITING_FEEDBACK
  ├─ 已恢复 → RESOLVED
  ├─ 仍未解决且有下一步 → WAITING_FEEDBACK
  ├─ 步骤耗尽 → ESCALATED
  └─ 结束诊断 → CANCELLED
```

## 知识检索与动态排障路径

初始故障描述先进入已有的本地 Hybrid RAG：Chroma 向量检索与 BM25 通过 RRF 形成候选，再由 `qwen3-rerank` 对候选证据做二阶段重排。召回的 `evidence_id` 会写入诊断状态，便于审计和交接。

除已知错误码的精确映射外，未知错误码和无错误码场景统一进入 `LLMKnowledgeResolver`。知识卡片 metadata 中的 `symptom_code` 只作为候选提示，不能直接决定故障类型。模型只能在现有 Playbook 类型列表中选择，并输出 `MATCH/AMBIGUOUS/NO_MATCH`、置信度、证据 ID 和证据原文引用；引擎会校验类型白名单、置信度、证据 ID 是否属于本次召回，以及引用是否真实存在于对应 chunk。任一校验失败时继续追问，不根据 metadata 或召回文本中的偶然关键词直接误诊。

`config/troubleshooting.yml` 将已知错误码映射到审核过的排障路径。没有错误码时，故障类型由本地 RAG 证据和受约束语义归一确定，不再依赖 Playbook 中的字符信号列表直接分类。当前所有故障 Playbook 均使用 `action_groups`，每个动作组把低风险、相互依赖的动作合并发送，并通过 `on_success`、`on_failure` 决定结束、进入下一组或升级人工，避免重复建议和一次输出全部无关方案。

动作组的结构为：

- `actions`：本轮需要用户连续完成的相关动作；
- `verification`：动作组完成后的成功/失败观察标准；
- `on_success`：通常为 `resolved`；
- `on_failure`：下一个动作组或 `handoff`。

例如 E01 先合并“断电、拆下主刷、清理缠绕物”，只有该动作组失败时才进入轴承和安装位置检查。其他故障也按各自的风险和依赖关系定义动作组。动作组仍使用现有 `StepAttempt` 持久化字段保存执行结果，以兼容已有 SQLite 状态。

每个排障步骤同时维护一个诊断契约：

- `instruction`：本轮给用户执行的动作；
- `success_signals`：完成动作后可验证的正向观察；
- `failure_signals`：动作无效或问题仍存在的负向观察。

用户反馈先由 `ObservationExtractor` 按当前动作组契约检查；规则无法明确判断时，`LLMObservationParser` 使用独立配置的 `deepseek-v4-flash`，在携带当前动作组上下文的前提下输出 `SUCCESS/FAILURE/UNKNOWN` 受约束 JSON，并保留原文依据。模型或规则都不能直接写入 `resolved`、`escalated` 等状态，这些状态只能由策略层验证后的观察驱动。

当前版本的 RAG 负责提供故障证据，Playbook 负责提供安全、可复核的执行步骤；尚未把所有知识卡片自动编译成完全动态的排障流程。

当前覆盖无法回充、工作异响、原地打转、水箱漏水、拖地不出水、配网/离线、吸力下降和主刷 E01 等路径。

## 人工升级策略

升级不依赖人工设定的概率分数，而依赖可解释条件：

- 用户明确请求人工客服；
- 冒烟、起火、糊味、发烫、电池鼓包、漏电等高风险信号；
- 连续澄清后仍没有足够诊断信息；
- 当前故障的排障步骤全部执行仍未恢复。

高风险信号可以抢占普通路由。升级后系统停止自动排查，创建包含故障现象、设备信息、错误码、已尝试步骤、风险项、知识依据和升级原因的结构化工单。

当前边界止于升级决策、交接摘要和 SQLite 工单持久化，未实现人工坐席领取、人工消息通道或第三方客服平台接入，因此不将其描述为完整转人工闭环。

## 持久化与审计

`DiagnosisStore` 使用 SQLite 保存：

- `diagnosis_cases`：活动和历史诊断状态；
- `handoff_tickets`：人工交接工单。

每轮状态变更同时写入 Hook 审计事件，但不写入原始用户对话，降低敏感信息扩散风险。

## 评测

评测脚本按 `session_id` 顺序重放多轮会话，统计：

- `action_accuracy`：追问、给出步骤、解决、取消或升级是否正确；
- `state_accuracy`：每轮诊断状态是否正确；
- `symptom_accuracy`：故障类型是否正确归一；
- `escalation_accuracy`：是否在正确轮次触发人工升级；
- `handoff_completeness`：交接工单必填字段完整度；
- `avg_turns_to_terminal`：达到解决、取消或升级状态的平均轮次；
- `task_completion_rate`：最终应进入终态的会话中，预测终态正确的比例；
- `unnecessary_repeat_question_rate`：后续轮次中，不符合标准动作的症状或结果重复追问比例。

基础集覆盖标准表达和完整流程；困难集覆盖口语故障描述、反馈同义表达、隐含风险和人工请求。困难集优化前的动作准确率为 6.25%，语义映射增强后作为回归集达到 100%。该结果用于证明失败驱动的覆盖增强，不作为独立盲测指标。

`diagnosis_state_ablation_dataset.csv` 固定合并上述两套数据，共 18 个会话、41 个轮次。评测器在完全相同的数据上比较两种配置：

- Stateful：同一会话复用 `session_id`，保留结构化诊断状态；
- Stateless ablation：每个轮次使用独立 `session_id`，模拟不保留前序状态。

首次冻结回归结果中，任务完成率为 100.0% 对 33.3%，无效重复追问率为 0.0% 对 82.6%。由于数据来自开发和困难回归集，该结果只用于消融验证，不能直接作为独立测试成绩写入简历。

### 普通 RAG 对照

`agent/troubleshooting_rag_comparison.py` 提供更贴近原始客服方案的单一
Baseline：每轮直接调用 `LOCAL_RAG` 使用的 Hybrid RAG 工具，一次性生成
排障建议，不读取 `DiagnosisState`、动作组或 SQLite 诊断状态。为控制评测
耗时，两侧在本次对照中统一使用 Vector + BM25 RRF Top-3，不调用二阶段
reranker；生产链路仍保留 qwen3-rerank。状态化方案和 Baseline 共享本地知识库
与 RRF 检索基础设施，但状态化方案额外使用 Playbook、观察解析与受约束状态迁移。

两侧回复由同一个轻量 Judge 归类为 `ask_symptom`、`ask_feedback`、
`give_step`、`resolve`、`escalate`、`cancel` 或 `unknown`，并检查是否重复
已经明确执行或失败的建议。评测输出汇总 Markdown 和逐轮 CSV，便于审计 Judge 判断。该评测会将
合成诊断对话和召回知识片段发送到配置的 DeepSeek、DashScope 服务，运行前
需要确认数据外发边界。

```powershell
python -m agent.troubleshooting_rag_comparison `
  --dataset data/diagnosis_post_contract_acceptance.csv
```

契约改造前的独立验收基线记录在 [diagnosis_independent_acceptance.md](diagnosis_independent_acceptance.md)。该报告已经冻结，契约改造后的基础集和困难集回归均通过，但不能回写独立验收集来制造新的独立成绩；后续需要使用另一套全新数据重新验收。
