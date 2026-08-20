# 无路由全工具 Agent Baseline 报告

- dataset: `D:\hang\job\2_resume\1_zhisaotong\data\agent_route_eval_dataset.csv; D:\hang\job\2_resume\1_zhisaotong\data\agent_route_eval_hard_dataset.csv`
- total_cases: `59`

## 指标汇总

- tool_selection_accuracy: `0.7381`
- invalid_tool_rate: `0.0000`
- unnecessary_tool_rate: `0.1017`
- avg_latency_ms: `3802.4696`
- p95_latency_ms: `8188.5265`

## 单题结果

| query | expected_tool | predicted_tool | tool_correct | unnecessary_tool | latency_ms |
|---|---|---|---:|---:|---:|
| 谢谢你 |  |  | True | False | 2887.5358 |
| 谢谢 |  |  | True | False | 2889.6024 |
| 你能提供哪些服务？ |  |  | True | False | 8188.5265 |
| 你是谁？ |  |  | True | False | 3574.3941 |
| 你好 |  |  | True | False | 4498.8502 |
| 扫地机器人滤网多久清洗一次？ | rag_summarize | rag_summarize | True | False | 2118.3721 |
| 主刷应该如何清理？ | rag_summarize | rag_summarize | True | False | 2192.3647 |
| 小户型应该怎么选扫地机器人？ | rag_summarize | rag_summarize | True | False | 2103.3263 |
| 地毯区域需要设置禁拖区吗？ | rag_summarize | rag_summarize | True | False | 2356.6911 |
| 拖布和水箱平时怎么维护？ | rag_summarize | rag_summarize | True | False | 2156.4277 |
| 2026 年扫地机器人主流品牌对比有哪些变化？ | web_search | web_search | True | False | 2205.7270 |
| 今年有哪些新款扫拖一体机器人？ | web_search | web_search | True | False | 2083.1473 |
| 最新扫地机器人固件更新了什么？ | web_search | web_search | True | False | 5053.2801 |
| 当前市场上哪些品牌的避障能力更强？ | web_search | web_search | True | False | 2436.5280 |
| 最近扫地机器人行业趋势如何？ | web_search | web_search | True | False | 2156.2934 |
| 深圳今天适合使用扫拖一体机器人的拖地功能吗？ | get_weather | get_weather | True | False | 2406.9693 |
| 北京明天会下雨，适合拖地吗？ | get_weather | get_weather | True | False | 3450.9932 |
| 今天适合拖地吗？ | get_weather |  | False | False | 3481.4731 |
| 帮我查询用户 1001 在 2025-01 的扫地机器人使用报告。 | fetch_external_data | fill_context_for_report | False | False | 3872.1380 |
| 查询用户1002 2025-03的耗材状态。 | fetch_external_data | fetch_external_data | True | False | 14198.9277 |
| 我想生成个人使用报告 | fetch_external_data |  | False | False | 2485.2319 |
| 扫地机器人主刷缠绕并报错 E01，应该怎么处理？ | rag_summarize | rag_summarize | True | False | 2023.7683 |
| 扫地机器人无法回充怎么办？ | rag_summarize | rag_summarize | True | False | 1993.2753 |
| 机器漏水且不出水怎么排查？ | rag_summarize | rag_summarize | True | False | 2014.3070 |
| 扫地机器人工作时有异响。 | rag_summarize | rag_summarize | True | False | 1797.5510 |
| 设备总是卡住，怎么处理？ | rag_summarize | rag_summarize | True | False | 2187.5561 |
| 这个品牌怎么选？ | web_search |  | False | False | 4344.5752 |
| 最近想换一台扫地机器人，有没有值得买的？ | web_search | web_search | True | False | 5309.6711 |
| 我的机器人怎么样？ |  |  | True | False | 5609.0658 |
| 帮我看看用户 1001 最近的设备使用情况 | fetch_external_data |  | False | False | 3314.7628 |
| 用户1001的耗材状态 | fetch_external_data |  | False | False | 3364.9523 |
| 北京今天天气怎么样？ | get_weather | get_weather | True | False | 1736.4774 |
| 主刷最近总是缠绕，怎么办？ | rag_summarize | rag_summarize | True | False | 2283.1195 |
| 我的扫地机器人出现错误，但不知道错误码 | rag_summarize | rag_summarize | True | False | 2586.4269 |
| 最近推荐一款适合小户型的扫地机器人 | web_search | rag_summarize | False | False | 2655.3930 |
| 对比一下智扫通和其他品牌 | web_search | web_search | True | False | 3460.4142 |
| 主刷报错 E01，顺便查一下北京天气 |  | rag_summarize | False | True | 3190.4532 |
| 查询用户 1001 在 2025-01 的使用报告，再推荐一款新机器 |  | fill_context_for_report | False | True | 33794.3484 |
| 机器没有报错，只想知道滤网多久清洗一次 | rag_summarize | rag_summarize | True | False | 2326.2404 |
| 我不是要查天气，只想问滤网怎么清洗 | rag_summarize | rag_summarize | True | False | 1953.6057 |
| 查询用户 ABC 在二零二五年一月的使用报告 | fetch_external_data | fill_context_for_report | False | False | 5330.8415 |
| 查询用户 1001 在 2025-13 的使用报告 | fetch_external_data |  | False | False | 3886.1727 |
| 最新的滤网清洗标准是什么 |  | rag_summarize | False | True | 2949.2996 |
| 固件更新后一直报错 E01 |  | rag_summarize | False | True | 2334.0866 |
| 它这个怎么处理？ |  |  | True | False | 5079.4004 |
| 扫地机主刷卡主了咋弄 | rag_summarize | rag_summarize | True | False | 2204.4040 |
| 帮我写一份离职申请 |  |  | True | False | 4535.6958 |
| 帮我查一下 |  |  | True | False | 3853.2961 |
| 明天雨大吗 | get_weather |  | False | False | 1955.1633 |
| E01 是什么意思？ | rag_summarize | rag_summarize | True | False | 2481.1865 |
| 北京天气怎样？另外主刷缠绕 |  | get_weather | False | True | 2540.5853 |
| 用户1001的使用报告 | fetch_external_data |  | False | False | 2938.1457 |
| 最新固件更新后机器无法回充 |  | rag_summarize | False | True | 2486.5605 |
| 我不想看使用报告，只想问主刷怎么清理 | rag_summarize | rag_summarize | True | False | 2409.1535 |
| 小智，那个功能怎么开？ |  |  | True | False | 5675.4772 |
| 扫地机滤惘多九洗一次 | rag_summarize | rag_summarize | True | False | 2232.8033 |
| 帮我订一张明天去上海的机票 |  |  | True | False | 4277.3056 |
| 最近机器总是卡住 | rag_summarize | rag_summarize | True | False | 2467.3505 |
| 现在有什么新款适合地毯 | web_search | web_search | True | False | 1966.0174 |
