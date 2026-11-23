# 评测设计与接口检查（v1）

本文回答三件事：**现有代码的入口和字段是什么**、**指标怎么定义才公平**、**要跑完整实验还缺什么**。所有结论都来自静态阅读 + 离线单测，未运行任何真实模型调用。

## 1. 接口检查结论

### 1.1 检索链路（B）

| 层 | 入口 | 关键点 |
|---|---|---|
| UI | `libs/ktem/ktem/pages/chat/__init__.py:1258` | 取选中文件，调用 `index.get_retriever_pipelines(...)` |
| 索引层 | `libs/ktem/ktem/index/file/index.py:473` | `_selector_ui.get_selected_ids` 还原文件 ID |
| pipeline | `libs/ktem/ktem/index/file/pipelines.py:115` | `run(text, doc_ids, ...) -> list[RetrievedDocument]` |
| 检索 | `libs/kotaemon/kotaemon/indices/vectorindex.py:134` | `run(text, top_k, **kwargs)`，内部 `self.embedding(text)[0].embedding`（**问题向量化的唯一出口**） |
| 结果字段 | `RetrievedDocument`（`base/schema.py:121`） | `text`、`metadata.file_name/page_label/section/type`、`score`、`doc_id` |

- 检索模式默认 `hybrid`（`pipelines.py:250-255`），但混合是"文本命中优先 + 向量候选，各自截断"，**没有分数融合**（`vectorindex.py:224-233`）。
- RRF 只存在于 B 的脚本里（`scripts/part_b_retrieval.py:41`），库内没有。
- 重排默认绑定 Cohere（`flowsettings.py:309`），但空 key 时**原样返回**（`rankings/cohere.py:41`）；而 `rerankers` 非空会把首轮候选从 `top_k` 放大到 `top_k*10`（`pipelines.py:156`）。**"关掉重排"和"候选深度"是耦合的**，做消融必须显式固定。

### 1.2 回答与引用链路（C）

| 能力 | 入口 | 字段 |
|---|---|---|
| 证据编号 | `format_context.py:71-77` | `[n]` 前缀 + `evidence_index`（`index/doc_id/file_name/page_label/section/type`） |
| 问答 | `citation_qa.py:205-259` | `run/invoke/stream(question, evidence, evidence_mode)` → `Document(text, metadata)` |
| 引用状态 | `citation_qa.py:41-67` | `citation_status ∈ {ok, failed}`、`citation_validation`（8 键，见 `utils.py:36-59`） |
| 拒答 | `utils.py:28`、`citation_qa.py:70` | `is_abstention(text)` 文本判定 + `ABSTAIN_MARKER` 固定句 |
| UI 提示 | `libs/ktem/ktem/reasoning/simple.py:230-258` | 信息面板两条提示，**不改写答案文本** |

**产品层没有结构化的拒答字段**（无 `abstained` 布尔值），拒答只能从文本判定——评测因此复用同一个 `is_abstention`，而不是另写一套。

### 1.3 入库链路（A）

| 能力 | 入口 | 说明 |
|---|---|---|
| 解析 + 分块 | `libs/ktem/ktem/index/file/pipelines.py:764-789` | `chunk_size or 1024`、`chunk_overlap or 256`；每个索引可用 `chunk_size/chunk_overlap` 覆盖（`index.py:419-437`） |
| 建库 | `VectorIndexing.run`（`vectorindex.py:95`）→ Chroma / LanceDB / SQLite | 片段保留 `doc_id` + `page_label/section/type` |
| 来源版本 | `test-materials/download_samples.py` | 6 份资料的 SHA-256 |

**缺口**：没有独立的建库脚本，也没有第二套 512/128 索引；`ingestion-results.json` 未记录分块参数。

## 2. 指标定义

### 2.1 Hit@K

对**有答案**题集 `A`：

```
Hit@K = |{q ∈ A : Top-K(q) ∩ Gold(q) ≠ ∅}| / |A|
```

- 只统计 `question_type == answerable`；无答案题跳过（否则会把"检索不到"当成命中）。
- `Gold(q)` 是按 `file + locator` 匹配的证据单元，**不是 chunk id**。
- 分母不含缺证据的题（会被记入 `skipped_ids`，避免静默放水）。

### 2.2 MRR@K

```
MRR@K = (1/|A|) · Σ_q 1/rank_q(首个命中)      未命中记 0
```

### 2.3 证据组召回@K（本框架对"双证据题"的补充）

`sc-02`（偏移 + 保证）、`sc-05`（选项卡 + 选项名）这类题，命中一半不该算全对：

```
recall@K = 已覆盖组数 / 总组数
```

组语义：**同组内任一条命中即算覆盖**（"第 26 或 27 张"），**不同组必须各自命中**。

### 2.4 Answer Correctness

人工三档：`correct=1.0`、`partial=0.5`、`incorrect=0.0`；`not_judged` 不计入分母。

```
accuracy       = Σ score / 已判题数      （部分正确给 0.5）
strict_accuracy = 判为 correct / 已判题数  （严格口径，两者都报）
```

脚本**不做语义判断**；无标签就是 `None`，不会用 0 或 1 填充。

### 2.5 Citation Correctness

分两层，必须分开报告：

| 层 | 定义 | 判定方式 |
|---|---|---|
| 有效性 | 回答里所有 `【n】` 都落在本次 `evidence_index` 内 | 自动，调用 `kotaemon.indices.qa.utils.validate_citations` |
| 支持度 | 被引片段确实支持该句 | 人工逐条标注 `supported: true/false` |

```
citation_validity_rate = 有效回答数 / 参与评分的回答数
citation_support_rate  = 被支持引用数 / 被标注引用总数
```

**口径决策**：拒答样本不计入引用指标。原因是当前 `validate_citations` 在"无任何引用标记"时直接置 `citation_valid=False`（`utils.py:137-140`），纯拒答会被误判成"引用无效"（已知缺陷，见 §4）。若不剔除，正确拒答会被扣分。

### 2.6 Refusal Accuracy

双向报告，避免"全拒答"或"全作答"刷单一指标：

```
refusal_accuracy   = 无答案题中拒答数 / 无答案题总数      （越高越好）
false_refusal_rate = 有答案题中拒答数 / 有答案题总数      （越低越好）
balanced_accuracy  = (refusal_accuracy + (1 - false_refusal_rate)) / 2
```

拒答判定复用 C 的 `is_abstention`（`utils.py:28`）。注意它也是"文本包含匹配"，`"don't know"` 一类可能被正文误触发——所以 `refusal_accuracy` 只能作为**自动初筛**，最终拒答结论仍需人工抽样复核。

## 3. 数据格式

见 `datasets/schema.md`。要点：

- 证据锚定 `file + locator(kind, value)`；答案类题目要求 `quote` 逐字可复核（`evidence_status=quote_verified`）。
- 支持证据组：同组可选、跨组必中，用于双证据/多问句题。
- 无答案题必须带 `checked_scope`（检查了哪些文件、用什么方法、可机器复核的缺失证据）。

**正式数据集 `qa_cases_v2.jsonl`：24 题**，分布 `direct 12 / synthesis 6 / detail 3 / unanswerable 3`，共 34 条证据引文。

数据来源是 4 份校验过 SHA-256 的真实文档：`11-hash2.pdf`（37 页）、`11-hash2.pptx`（37 张）、`USTC-ClassIn.pdf`（16 页）、`USST-syllabus.docx`（模板）。

选题约束（避免"看起来合理但没依据"的题）：

1. 只使用 **文本层可提取** 的内容。PDF 里的公式排版、槽位示意表在文本层是乱序的，凡无法逐字核对的片段一律不出题。
2. `USST-syllabus.docx` 是空白模板，只有"桥梁工程的基本知识（示例）"一行有真实内容，因此只从该行出题（学生两项任务），另出一道无答案题（学分/学时字段为空）。
3. `unanswerable` 题必须能机器验证"缺失"：`absent_phrases` 在全库无命中、`max_mentions` 限制提及次数（如 cuckoo 仅出现 1 次且只有名称）、或 `blank_fields` 断言字段行不含数字。

复核命令：

```sh
.venv/Scripts/python.exe -m evaluation.scripts.verify_evidence
# -> 24 cases, 34 evidence quotes, 4 documents loaded, 0 error(s)
```

`datasets/qa_cases.jsonl`（原 6 道 search-cases）已废弃，仅作历史对照。

## 4. 已知风险与设计决策

| 风险 | 证据 | 对评测的影响 | 处理 |
|---|---|---|---|
| 纯拒答被判 `citation_valid=False` | `utils.py:137-140` + PR #3 的 `short-circuit abstention` 提交 | 引用有效率被拒答样本拉低 | 拒答样本不计入引用指标；报告注明基线 commit |
| `citation_status=failed` 依赖 5s 超时 | `citation_qa.py:35,345` | 引用状态可能偶发抖动 | 多次运行取多数，或只统计 `citation_validation` |
| 关掉引用样式仍会误报"引用未校验" | `simple.py:425-427` + `413-415` | 若做"无引用"配置，指标失真 | 实验矩阵固定 `highlight_citation=on` |
| B 的索引不是 A 的主库，且只支持 PDF | `docs/part-b-retrieval.md:3` | 无法直接比较 PPTX/DOCX 题目 | 适配器接口先就位；`part-b` 路径标注"实验性" |
| 重排未启用，且候选深度耦合 | `pipelines.py:156`、`cohere.py:41` | "有无重排"不是干净消融 | 显式固定 `rerankers=[]` 与 `use_reranking`，作为配置因子记录 |
| 混合检索无融合 | `vectorindex.py:224-233` | 混合分数不可比 | 只比较排名类指标，不比较分数 |

## 5. 实验矩阵（待 A 提供两套索引后执行）

| 因子 | 取值 | 说明 |
|---|---|---|
| 分块 | 1024/256、512/128 | A 的两套独立索引 |
| 检索模式 | vector、hybrid、rrf | rrf 目前只在 B 的脚本里 |
| Top-K | 5、7、20 | 5 为主口径，7/20 用于观察证据覆盖变化 |
| 重排 | off | 当前无可用 reranker，不纳入主表 |

每个组合报告：`Hit@K`、`MRR@K`、`evidence_recall@K`、`answer_accuracy`、`citation_validity_rate`、`citation_support_rate`、`refusal_accuracy`、`false_refusal_rate`。

## 6. 需要 A / B / C 配合的事项

| 对象 | 事项 | 原因 |
|---|---|---|
| A | 两套独立索引（1024/256、512/128）+ 分块参数说明 + 可重复建库脚本 | 跨分块对比的前置条件；当前只有一套结果且未记录参数 |
| A | 在 `ingestion-results.json` 里补 `chunk_size/chunk_overlap` | 否则实验报告无法说明"数字对应哪套参数" |
| B | 把 `part_b_retrieval.py query` 的索引换成/兼容 A 的主库，或提供 A 库的查询入口 | 现在 B 只查自建 PDF 索引，题目要求"从 A 建好的库检索" |
| B | 支持 PPTX/DOCX 的独立索引，或确认评测只用 PDF 题 | 否则 6 题里 4 题无法用 part-b 路径 |
| C | 确认拒答判定的权威口径（文本匹配 vs 结构化字段） | 决定 `refusal_accuracy` 是否可信 |
| C | 提供一次真实 pipeline 的输出 JSONL（`case_id/text/evidence_index`） | 评测脚本只吃录制结果，不直接调 DeepSeek |
| 全体 | 确认评测基线 commit（`2bed52f` 还是 PR #3 合并后） | C 的拒答/引用缺陷正在 PR #3 修 |

## 7. 本阶段交付物

| 交付物 | 位置 | 状态 |
|---|---|---|
| 统一数据格式 | `evaluation/datasets/schema.md` | ✅ |
| 24 题正式数据集 | `evaluation/datasets/qa_cases_v2.jsonl` | ✅ 12/6/3/3，34 条引文逐字复核 |
| 引文复核工具 | `evaluation/scripts/verify_evidence.py` | ✅ 0 error |
| 指标实现 | `evaluation/metrics/*.py` | ✅ |
| 离线单测 | `evaluation/tests/` | ✅ 全绿 |
| 检索适配器接口 | `evaluation/adapters/base.py`、`part_b.py` | ✅（fixture 已验证，part-b 未对真实索引运行） |
| 实验矩阵 | `evaluation/matrix.py` | ✅ 18 组，全部显式 blocked |
| 结果记录与报告 | `evaluation/reporting.py` | ✅ 12 字段/题 + 分题型统计 |
| Simulated 链路验证 | `evaluation/scripts/run_experiment.py` | ✅ 带 SIMULATED 标记 |
| 真实实验结果 | — | ❌ 未运行，符合本阶段约束 |

## 8. 实验矩阵与阻塞状态

### 8.1 矩阵定义

| 维度 | 取值 | 数量 |
|---|---|---:|
| Chunking | `1024-256`、`512-128` | 2 |
| Retrieval | `vector`、`hybrid`、`rrf` | 3 |
| Top-K | `3`、`5`、`10` | 3 |
| Reranking | `off`（默认）；`on` 为可选 | 1 / 1 |

主矩阵 = 2 × 3 × 3 = **18 组**，全部使用**同一份** `datasets/qa_cases_v2.jsonl`（24 题）。报告里记录该文件 SHA-256，避免"换了数据集却说同一组实验"。

Reranking 不进入主矩阵：`on` 状态是 **blocked**（Cohere key 为空时重排原样返回；qwen3-rerank 未接入），所以不存在"已实现重排"的实验。

### 8.2 各维度状态

| 维度取值 | 状态 | 原因 | 依据 |
|---|---|---|---|
| chunking `1024-256` | blocked | 本机没有 A 的主库索引；Part B 自建索引是**按页**抽文本，不是 1024/256 token 分块 | `docs/part-b-retrieval.md`、`libs/ktem/ktem/index/file/pipelines.py:788` |
| chunking `512-128` | blocked | A 未提供第二套索引，也没有可复现的分块参数记录 | 本文件 §6 |
| retrieval `vector` | blocked | 需要索引 + `DASHSCOPE_API_KEY`；本阶段不调真实 API | `scripts/part_b_retrieval.py:134` |
| retrieval `hybrid` | blocked | 同上；且混合检索无分数融合，只能比排名类指标 | `vectorindex.py:224` |
| retrieval `rrf` | blocked | RRF 只在 B 的脚本里，仍需索引与 key | `scripts/part_b_retrieval.py:41` |
| reranking `off` | ready | 不使用重排；需把候选深度固定为 `top_k`，避免 `do_extend` 隐式放大 | `pipelines.py:156` |
| reranking `on` | blocked | 无可用 reranker | `rankings/cohere.py:41` |

结果：**18 组全部 blocked，0 ready**。`run_experiment --matrix` 只输出状态，不产出任何指标。

### 8.3 每道题的结果记录

`per_case.jsonl` 每条 = 一道题 × 一个实验组，字段：

`case_id`、`category`、`chunking_strategy`、`retrieval_method`、`top_k`、`reranking`、`simulated`、`retrieved_evidence`、`hit_at_k`、`reciprocal_rank`、`recall_at_k`、`generated_answer`、`answer_correct`、`citation_valid`、`citation_supported`、`refusal_correct`、`expected_refusal`、`citation_support_supported`、`citation_support_total`。

- `citation_valid`：编号存在性（自动，复用 C 的 `validate_citations`）；
- `citation_supported`：`supported` / `unsupported` / `mixed`，来自人工逐条标注；
- 二者**分开存储、分开统计**，编号合法不代表内容支持。

### 8.4 报告统计

支持 7 项：`Hit@K`、`MRR`、`Recall@K`、`Answer Correctness`、`Citation Validity`、`Citation Support`、`Refusal Accuracy`，并额外给出 `false_refusal_rate`、`balanced_accuracy`。

每种指标同时给出**总体**与**分题型**（direct / synthesis / detail / unanswerable）。无样本或无标签时输出 `—`（`null`），不按 0 计算。

### 8.5 Simulated 政策

`synthetic fixture` 只用于验证"矩阵 → 记录 → 报告"链路：

- 生成器 `make_simulated_fixture.py` 的每个输出都带 `"simulated": true`；
- 报告顶部有 `SIMULATED — synthetic fixture, NOT experimental results` 横幅；
- 报告配置里的 chunking / retrieval 写成 `SIMULATED-*`，不可能被误读成真实配置；
- `run_experiment` 拒绝加载未标记 `simulated` 的 fixture。

`evaluation/results/` 里**没有**任何真实或模拟结果入库；验证产物写到仓库外。

### 8.6 当前结论（明确声明）

1. 正式测试集是 `datasets/qa_cases_v2.jsonl`，**24 题**（12 直接 / 6 综合 / 3 细节 / 3 无答案）。
2. 数据来源只有 **4 份可读取文档**：`11-hash2.pdf`、`11-hash2.pptx`、`USTC-ClassIn.pdf`、`USST-syllabus.docx`。
3. `UTS-accessible-word.docx` 与 `UTS-accessible-powerpoint.pptx` 因源站 **HTTP 403 未纳入**。
4. **尚未完成任何真实检索或问答实验**：18 组实验全部 blocked，`results/` 为空。
5. 本地代码基线为 `2bed52f`（加上 D 自己的 evaluation 提交），**尚未包含 PR #3**；因此 C 的"纯拒答被判 citation_valid=False"等缺陷仍存在，引用指标必须剔除拒答样本（已实现）。

