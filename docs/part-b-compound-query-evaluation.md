# Part B：跨页复合问题检索复测

## 范围与方法

2026-11-23 复用 [六份样本验收](part-b-six-sample-check.md)的本地索引，不重新入库。使用项目的 `DocumentRetrievalPipeline`，百炼 `text-embedding-v4`、2048 维，查询时只选中相应文件。比较原问题向量 Top-5、Top-7、Top-20，以及人工拆问后各取 Top-5 并按片段 ID 去重、用 RRF 合并成 Top-5。另对原问题运行项目原生混合检索 Top-7。没有接入语义 reranker 或 LLM 答案生成。

## UW PPTX 复合问题

问题：“原槽位被占后，二次探测的前四个偏移是多少？当表大小为素数且负载因子小于一半时，能保证什么？”需要第 24 张的 `1、4、9、16` 和第 26/27 张的“能找到空槽”。

| 策略 | 第 24 张 | 第 26/27 张 | 结论 |
|---|---:|---:|---|
| 原问题，向量 Top-5 | 第 1 | 未进入 | 只有前半问证据 |
| 原问题，向量 Top-7 | 第 1 | 第 27 张第 6、第 26 张第 7 | 两部分证据均进入候选 |
| 原问题，向量 Top-20 | 第 1 | 第 27 张第 6、第 26 张第 7 | 与 Top-7 相比没有改善这道题的目标排名 |
| 原问题，原生混合 Top-7 | 第 1 | 第 27 张第 6、第 26 张第 7 | 全文分支返回 0 条，与向量结果相同 |
| 人工拆为两问，每问向量 Top-5，RRF 合并 Top-5 | 第 1 | 第 27 张第 4 | 两部分证据均进入最终五条 |

拆问的两个子问题分别是“原槽位被占后，二次探测的前四个偏移是多少？”和“当表大小为素数且负载因子小于一半时，二次探测能保证什么？”。RRF 使用 `1 / (60 + rank)` 累加两个子问题的名次；相同片段 ID 只保留一次。结果记录 `score_type=rrf`、每个子问题的名次和原始相似度。RRF 分数不能与向量相似度直接比较。

## 六份资料回归

六道固定题目均以纯向量 Top-7 重跑。原 Top-5 的片段 ID 与顺序在六题中均未改变，Top-7 每题均无重复 ID。

| 资料 | 预期证据在 Top-7 中的位置 |
|---|---|
| UW PDF | 第 25 页第 2 |
| UW PPTX | 第 24 张第 1；第 27 张第 6、第 26 张第 7 |
| UTS Word | `Headings` 章节第 1 |
| 上海理工大学教学大纲 | 桥梁示例片段第 1 |
| UTS PowerPoint | 第 4 张第 1；第 17 张第 5 |
| 中科大 ClassIn PDF | 第 8 页第 1 |

这六题的证据覆盖从 Top-5 的 5/6 变为 Top-7 的 6/6，仅代表这六个手工验收问题。Top-7 仍包含无关片段；实际问答还需要控制传给生成模型的上下文。人工拆问的 RRF 只在 UW 复合问题上测过，不能据此宣称其他问题均会改善。

## 本地结果与调用

完整片段正文、ID、来源页、相似度和融合来源在 `ktem_app_data/part_b/project-pipeline-v3`：`results.json`、`results-six-top7.json`、`results-compound-top7.json`、`results-compound-top20.json`、`results-compound-hybrid-top7.json`、`results-compound-split-rrf-top5.json`。脚本允许 `--result-name` 将每次查询写入新文件，以免覆盖之前的结果。

```powershell
$env:PYTHONUTF8 = '1'
$env:ANONYMIZED_TELEMETRY = 'False'
$envFile = '你的本机百炼配置文件路径'
$indexDir = 'ktem_app_data/part_b/project-pipeline-v3'

# 原问题，扩大到 Top-7：
.\.venv\Scripts\python.exe scripts\part_b_project_pipeline.py --query-only `
  --env-file $envFile --index-dir $indexDir --samples-dir downloads\real-samples `
  --file 11-hash2.pptx --mode vector --top-k 7 `
  --question '原槽位被占后，二次探测的前四个偏移是多少？当表大小为素数且负载因子小于一半时，能保证什么？' `
  --result-name 'my-compound-top7.json'

# 人工提供两个子问题，按片段 ID 去重并用 RRF 融合：
.\.venv\Scripts\python.exe scripts\part_b_project_pipeline.py --query-only `
  --env-file $envFile --index-dir $indexDir --samples-dir downloads\real-samples `
  --file 11-hash2.pptx --mode vector --top-k 5 `
  --question '原槽位被占后，二次探测的前四个偏移是多少？当表大小为素数且负载因子小于一半时，能保证什么？' `
  --subquestion '原槽位被占后，二次探测的前四个偏移是多少？' `
  --subquestion '当表大小为素数且负载因子小于一半时，二次探测能保证什么？' `
  --result-name 'my-compound-split-rrf.json'
```

目前建议保留可配置的 Top-K，以 Top-7 作为这六题的候选数；遇到多证据复合问题时，可以显式提供子问题再融合。自动识别与拆分问题尚未实现。接入队友 A 的数据库仍取决于获取其索引配置和数据访问方式；本次只验证本机独立索引，未上传 GitHub。
