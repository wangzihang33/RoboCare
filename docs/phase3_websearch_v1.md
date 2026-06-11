# Phase 3 动态 Web RAG 强化 V1

## 目标

本阶段目标是构建具备缓存复用、URL 治理、来源追踪、引用输出和指标统计能力的动态 Web RAG 链路。

一句话理解：

```text
WebSearch 基于结构化搜索结果、网页来源 metadata、TTL 缓存、临时向量检索和 Hook trace，形成可追溯、低成本、可观测的联网知识补充链路。
```

## 整体流程

```text
Agent 调用 web_search(query)
  ↓
WebSearchService.search_summarize
  ↓
检查本地缓存 WebSearchCache
  ├─ 命中缓存：直接返回缓存 answer，并记录 cache_hit=true
  └─ 未命中：继续执行真实 Web RAG
       ↓
     SerperClient.serper
       ↓
     SerperClient.extract_components
       ├─ 转成 SearchResult
       ├─ URL 标准化
       ├─ 过滤 PDF/图片/压缩包等非网页资源
       ├─ 重复 URL 去重
       └─ 同域名限流
       ↓
     WebContentFetcher.fetch_pages
       ├─ 多线程抓取网页正文
       ├─ 生成 FetchedPage
       └─ 记录 fetched_at/status/content_length
       ↓
     EmbeddingRetriever.retrieve_page_embeddings
       ├─ 文本切分
       ├─ 临时 Chroma 向量库
       └─ 每个 chunk 写入 title/url/domain/rank/fetched_at metadata
       ↓
     LLM 基于来源化 context 生成回答
       ↓
     代码追加“参考来源”
       ↓
     缓存 answer + sources + trace
       ↓
     Hook 写入 websearch_trace
```

## 代码文件职责

| 文件 | 职责 |
|---|---|
| `websearch/models.py` | 定义 `SearchResult`、`FetchedPage`、`SourceFilterStats`，把搜索结果和抓取结果结构化。 |
| `websearch/source_filter.py` | URL 标准化、追踪参数清理、PDF/图片等资源过滤、重复 URL 去重、同域名限流。 |
| `websearch/cache.py` | WebSearch 本地 JSON 缓存，基于 `query + top_k + date_bucket + cache_version` 生成 `cache_key`。 |
| `websearch/serper_service.py` | 调用 Serper，并把原始 `organic` 结果转为结构化搜索结果。 |
| `websearch/fetch_web_content.py` | 多线程抓取网页正文，并返回带来源信息的 `FetchedPage`。 |
| `websearch/retrieval.py` | 构建临时 Chroma 向量库，并将来源 metadata 写入每个检索 chunk。 |
| `websearch/web_search_service.py` | WebSearch 主编排：缓存检查、抓取、检索、总结、引用来源追加、trace 记录。 |
| `agent/hooks/lifecycle.py` | 提供 `record_tool_trace`，支持写入 `websearch_trace` 事件。 |
| `agent/tools/middleware.py` | 在 `web_search` 成功后读取 `web.last_trace` 并写入 Hook 日志。 |
| `agent/hooks/report.py` | 统计 `cache_hit_rate`、平均来源数、过滤后 URL 数、抓取成功数和失败数。 |

## 搜索结果结构化

Serper 的 `organic` 搜索结果会被统一映射为 `SearchResult`，每条结果都承接标题、链接、摘要、排序、域名和来源类型等信息：

```text
title
url
snippet
rank
domain
original_url
source
```

后续缓存、抓取、检索、引用输出都依赖同一个结构，避免靠 list 下标维护标题、链接和摘要之间的对应关系。

## URL 过滤与去重

`source_filter.py` 做了几类处理：

```text
1. 只保留 http/https URL。
2. 去掉 utm_*、fbclid、gclid 等追踪参数。
3. 去掉 URL fragment。
4. 过滤 PDF、图片、视频、压缩包、Office 文档等非正文网页资源。
5. 重复 URL 只保留一条。
6. 每个域名默认最多保留 2 条，避免搜索结果被单一网站占满。
7. title 和 snippet 都为空的低质量结果会被过滤。
```

过滤结果会写入 `filter_stats`：

```text
original_count
kept_count
dropped_invalid_url
dropped_unsupported_url
dropped_duplicate_url
dropped_duplicate_domain
dropped_low_quality
```

## 网页抓取来源化

`WebContentFetcher.fetch_pages()` 返回的是 `FetchedPage` 列表。

每个页面包含：

```text
title
url
snippet
rank
domain
content
fetched_at
status
error_message
content_length
```

这样后续即使网页抓取失败，也能知道失败的是哪个 URL、哪个搜索排名、哪个 domain。

## 临时向量库来源 metadata

`EmbeddingRetriever.retrieve_page_embeddings()` 会把每个网页 chunk 的 metadata 写成：

```text
source_id
title
url
domain
snippet
search_rank
fetched_at
content_length
source
```

这些 metadata 会进入最终 context，也会用于生成参考来源列表。

## 缓存设计

缓存位置：

```text
cache/websearch/
```

该目录已加入 `.gitignore`，不会提交到 git。

缓存 key 由以下字段生成：

```text
cache_version
normalized_query
top_k
date_bucket
```

最终通过 sha256 生成 `cache_key`。

缓存内容包括：

```text
query
normalized_query
cache_key
created_at
expires_at
answer
sources
trace
```

TTL 策略：

| 查询类型 | TTL |
|---|---|
| 市场、价格、排行、品牌对比、最新产品 | 12 小时 |
| 默认查询 | 24 小时 |
| 故障、维护、保养、使用指导类通用知识 | 7 天 |

## 引用输出

LLM 生成正文后，代码会基于 `sources` 稳定追加参考来源。

输出形态：

```text
回答正文……

参考来源：
[S1] 标题 - URL | fetched_at=...
[S2] 标题 - URL | fetched_at=...
```

这里的来源不是模型自由生成的，而是来自检索 chunk 的 metadata，所以可追溯性更稳定。

## Hook 指标接入

WebSearch 执行成功后，`middleware.py` 会读取 `web.last_trace`，并通过 `record_tool_trace` 写入 Hook JSONL。

事件类型：

```text
stage = websearch_trace
```

典型字段：

```text
cache_hit
cache_key
source_count
original_count
urls_after_filter
filter_stats
fetch_stats
retrieved_docs
cached_created_at
cached_expires_at
```

## 统计命令

```bash
python -m agent.hooks.report
python -m agent.hooks.report --date 20260611
```

报表字段：

| 字段 | 说明 |
|---|---|
| `websearch_trace_events` | WebSearch trace 事件数 |
| `cache_hits` | 缓存命中次数 |
| `cache_hit_rate` | 缓存命中率 |
| `avg_source_count` | 平均来源数量 |
| `avg_urls_after_filter` | 过滤后平均 URL 数 |
| `avg_fetch_success` | 平均成功抓取页面数 |
| `avg_fetch_errors` | 平均抓取失败页面数 |

## 验证命令

语法检查：

```bash
python -m py_compile websearch/cache.py websearch/models.py websearch/source_filter.py websearch/serper_service.py websearch/fetch_web_content.py websearch/retrieval.py websearch/web_search_service.py
python -m py_compile agent/hooks/lifecycle.py agent/hooks/report.py agent/tools/middleware.py
```

缓存读写验证：

```bash
python -c "from websearch.cache import WebSearchCache; c=WebSearchCache('cache/test_websearch_cache'); p=c.set('2026 扫地机器人 品牌 对比', top_k=20, answer='answer', sources=[{'url':'u'}], trace={'cache_hit':False}); hit=c.get('2026 扫地机器人 品牌 对比', top_k=20); print(bool(hit), hit['answer'], hit['sources'][0]['url'])"
```

Hook 报表验证：

```bash
python -c "from agent.hooks.report import summarize_events; events=[{'stage':'after_tool_call','tool_name':'web_search','latency_ms':100},{'stage':'websearch_trace','tool_name':'web_search','cache_hit':False,'source_count':3,'urls_after_filter':5,'fetch_stats':{'success_count':4,'error_count':1}},{'stage':'after_tool_call','tool_name':'web_search','latency_ms':10},{'stage':'websearch_trace','tool_name':'web_search','cache_hit':True,'source_count':3}]; print(summarize_events(events))"
```

引用输出验证：

```bash
D:\app\conda\envs\zhisaotong\python.exe -c "from langchain_core.documents import Document; from websearch.web_search_service import WebSearchService; docs=[Document(page_content='content', metadata={'source_id':'S1','title':'Title','url':'https://example.com/a','domain':'example.com','search_rank':1,'fetched_at':'2026-06-11T00:00:00Z','snippet':'snippet'})]; sources=WebSearchService._extract_sources(docs); print(WebSearchService._append_references('answer', sources))"
```

## 当前 V1 边界

当前版本已经完成动态 Web RAG 的核心工程闭环，但仍有后续可优化点：

```text
1. URL 质量评分仍是规则级，后续可以加入来源可信度评分。
2. TTL 判断基于关键词，后续可以改成 LLM 或分类器判断查询时效性。
3. 当前缓存是本地 JSON 文件，生产环境可以迁移到 Redis 或 SQLite。
4. 当前引用来源是最终追加列表，后续可以进一步要求正文句子级引用。
5. 当前未跑真实联网回归，正式演示前建议用 1-2 个低成本 query 做真实 WebSearch 验证。
```

## 可写入简历的一句话

构建动态 Web RAG 证据链，基于 TTL 缓存、URL 去重过滤、网页来源 metadata 追踪、引用输出和 Hook 指标统计，实现可追溯、低成本、可观测的联网知识补充链路。

## 面试摘要

该项目的动态 Web RAG 链路由搜索结果治理、网页抓取、临时向量检索、引用输出、TTL 缓存和 Hook 指标统计组成。

Serper 返回结果会被结构化成 `SearchResult`，并经过 URL 标准化、PDF/图片过滤、重复 URL 去重和同域名限流，减少无效抓取。

网页抓取结果会保存为 `FetchedPage`，记录标题、URL、域名、搜索排名、抓取时间、正文长度和状态。进入临时 Chroma 检索时，这些来源信息会写入每个 chunk 的 metadata。

生成回答时，系统基于检索到的 sources 稳定追加参考来源列表，避免完全依赖模型自由生成引用。

缓存层基于 `query + top_k + date_bucket` 生成本地 JSON 缓存，并按查询时效性设置 TTL；WebSearch 的缓存命中率、来源数量、过滤后 URL 数、抓取成功/失败数接入 Hook JSONL 和统计报表。
