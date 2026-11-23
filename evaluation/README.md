# evaluation/

离线优先的评测框架，用于衡量本组 RAG 项目的**检索**（B）、**证据化回答 / 引用 / 拒答**（C），以及不同分块方案（A）对检索的影响。

- 不修改 `libs/kotaemon/`、`libs/ktem/`、`app.py` 等核心代码，只以库的方式调用。
- 不调用任何真实模型 API。指标全部是纯函数，可从录制的预测结果离线计算。
- **不编造数据**：每条引文都必须能在原文指定页/张里逐字匹配（`verify_evidence.py` 会失败，如果匹配不上）。

## 正式数据集：`datasets/qa_cases_v2.jsonl`

30 道题，全部基于**已校验的真实文档**，每题都有文档名、页码/幻灯片号、章节或表格行，以及原文片段。

| 题型 | 数量 | 说明 |
|---|---:|---|
| `direct` 直接事实题 | 12 | 单个证据点、答案在原文中直接给出 |
| `synthesis` 综合理解题 | 8 | 每题 ≥2 个证据组，需跨页/跨段整合（多证据整合，不宣称深度推理） |
| `detail` 细节定位题 | 5 | 定位到具体页/张的一行或一个符号（如 long 字段哈希写法、墓碑符号 🪦） |
| `unanswerable` 无答案题 | 5 | 知识库确实没有答案，且记录了检查范围与可机器复核的缺失证据 |

> `datasets/qa_cases.jsonl`（原 6 道 search-cases 转换）**已废弃**，仅保留作历史对照，不参与任何评测。按约定不再修改或补齐。

### 数据来源

| 文件 | 来源 | 校验 |
|---|---|---|
| `11-hash2.pdf` | UW CSE 332《Hash Tables II》讲义（37 页） | SHA-256 `7aa2d367…` |
| `11-hash2.pptx` | 同课程 PPTX 版（37 张） | SHA-256 `8d92e93c…` |
| `USTC-ClassIn.pdf` | 中科大《线上教学操作说明》（16 页，中文） | SHA-256 `ced8a509…` |
| `USST-syllabus.docx` | 上海理工大学课程教学大纲模板（中文，含桥梁示例行） | SHA-256 `09762a9c…` |

下载与校验：

```sh
python test-materials/download_samples.py   # 前两份
```

`download_samples.py` 对 UTS 的两份文档会返回 **HTTP 403**（源站拒绝访问），因此本数据集**只使用上述 4 份**。实测结论已记录在 `DESIGN.md` §3，不通过猜测补题。

## 快速开始

```sh
# 环境（仓库根目录）
uv sync --locked --package kotaemon --package ktem

# 1. 结构校验（30 题、题型分布、字段规则、无答案题的检查范围）
.venv/Scripts/python.exe -m evaluation.scripts.validate_dataset \
    evaluation/datasets/qa_cases_v2.jsonl --min-cases 24

# 2. 引文复核：重新抽取 4 份文档，逐条断言引文出现在指定页/张/章节
.venv/Scripts/python.exe -m evaluation.scripts.verify_evidence

# 3. 跑检索指标（离线 fixture，不是实验结论）
.venv/Scripts/python.exe -m evaluation.scripts.run_retrieval_eval \
    --adapter fixture --stub-fixture evaluation/tests/fixtures/rankings_stub.json \
    --k 5 --out evaluation/results/retrieval_fixture.json

# 4. 聚合回答 / 拒答 / 引用指标（离线 fixture）
.venv/Scripts/python.exe -m evaluation.scripts.run_answer_eval \
    --predictions evaluation/tests/fixtures/predictions_stub.jsonl \
    --labels evaluation/tests/fixtures/labels_stub.jsonl \
    --citation-labels evaluation/tests/fixtures/citation_labels_stub.jsonl \
    --out evaluation/results/answer_fixture.json

# 5. 单测（全部离线；引文用例在文档缺失时自动跳过）
.venv/Scripts/python.exe -m pytest evaluation -q

# 6. 实验矩阵状态（18 组，当前全部 blocked，不产出指标）
.venv/Scripts/python.exe -m evaluation.scripts.run_experiment --matrix

# 7. 用 synthetic fixture 验证"记录 → 报告"链路（输出带 SIMULATED 标记）
.venv/Scripts/python.exe -m evaluation.scripts.make_simulated_fixture \
    --out-dir .eval-tmp/fixture
.venv/Scripts/python.exe -m evaluation.scripts.run_experiment \
    --simulated-fixture .eval-tmp/fixture --out-dir .eval-tmp/out --k 5
```

> 第 3、4 步用的是 `tests/fixtures/*_stub.*`，**synthetic fixture，不是实验结果**，只证明"数据集 → 适配器 → 指标 → 报告"链路通。

## 目录

```
evaluation/
├── README.md              本文件
├── DESIGN.md              接口检查结论、指标定义、实验矩阵、协作事项
├── datasets/
│   ├── schema.md          JSONL 字段规范（含证据分组、无答案题的检查范围）
│   ├── qa_cases_v2.jsonl  ✅ 正式数据集：30 题
│   ├── qa_cases.jsonl     ⚠️ 已废弃：原 6 道 search-cases，仅作历史对照
│   └── candidates.md      v2 的选题依据与后续扩充方向
├── metrics/               retrieval / answer / citation 指标（纯函数）
├── matrix.py              实验矩阵与阻塞状态（18 组）
├── reporting.py           每题记录 + 分题型报告 + markdown 渲染
├── adapters/              base 协议、part_b 适配器、fixture 适配器
├── scripts/
│   ├── convert_search_cases.py   v1 转换器（历史用途，仍可用）
│   ├── validate_dataset.py       结构与规则校验
│   ├── verify_evidence.py        引文逐字复核 + 无答案题缺失校验
│   ├── make_simulated_fixture.py 生成 simulated fixture
│   ├── run_experiment.py         矩阵状态 / simulated 运行
│   ├── run_retrieval_eval.py
│   └── run_answer_eval.py
├── tests/                 单测 + fixtures
└── results/               运行产物（当前为空，报告需人工确认后再入库）
```

## 指标口径（详见 DESIGN.md §2）

| 指标 | 定义 | 自动化程度 |
|---|---|---|
| `Hit@K` | 有答案题中，Top-K 含**至少一个**正确证据的题目比例 | 全自动 |
| `MRR@K` | 首个正确证据排名的倒数均值 | 全自动 |
| `evidence_recall@K` | 已覆盖证据**组**数 / 总组数 | 全自动 |
| `answer_accuracy` | 人工标签 `correct=1 / partial=0.5 / incorrect=0` 的均值 | 标签由人工给 |
| `citation_validity_rate` | 非拒答回答中，全部 `【n】` 都能在本次证据里找到的比例 | 全自动（复用 C 的校验） |
| `citation_support_rate` | 被引片段确实支持该句的比例 | 标签由人工给 |
| `refusal_accuracy` | 无答案题中正确拒答的比例 | 全自动（复用 C 的判定） |
| `false_refusal_rate` | 有答案题中被误拒答的比例 | 全自动 |
| `balanced_accuracy` | 上述两者的平衡值，防止"全部拒答"刷分 | 全自动 |

## 证据与分块的关系

`gold_evidence` 锚定 **文件 + 页码/幻灯片/章节**，不用 chunk ID。原因是 1024/256 与 512/128 两种分块会生成不同的 chunk id 和边界，用 chunk id 做判据会让对比失真。`chunk_id` 字段保留但默认为 `null`。

多证据题用 `group` 表达语义：

- 同组任选（例如"第 26 或 27 张任一命中"）；
- 跨组必中（例如"选项卡"+"选项名"是两组，都必须命中）。

## 与仓库现有资产的关系

| 现有资产 | 关系 |
|---|---|
| `test-materials/download_samples.py` | 数据集来源与 SHA-256 校验；本框架不改动它 |
| `test-materials/search-cases.md` | 仅用于生成**已废弃**的 v1，不再参与评测 |
| `test-materials/ingestion-results.json` | A 的入库计数与锚点；本框架不做指标复用 |
| `scripts/part_b_*.py` | B 的检索实验脚本；`PartBRetrievalAdapter` 通过 CLI 调用并解析 JSON，不改其代码 |
| `kotaemon.indices.qa.utils` | C 的引用校验与拒答判定；`metrics/` **直接调用**，保证与产品行为一致 |

## 实验矩阵（18 组，当前全部 blocked）

| 维度 | 取值 |
|---|---|
| Chunking | `1024-256`、`512-128` |
| Retrieval | `vector`、`hybrid`、`rrf` |
| Top-K | `3`、`5`、`10` |
| Reranking | `off`（默认）；`on` 可选且 blocked |

2 × 3 × 3 = 18 组，**全部使用同一份 30 题数据集**（报告记录其 SHA-256）。

当前 **0 ready / 18 blocked**，原因是：两套 chunking 索引都不在本机可查、检索需要真实 embedding key、reranker 未接入。`run_experiment --matrix` 只输状态表，**不产出任何指标**；详见 [DESIGN.md §8](DESIGN.md)。

## 结果记录与报告

`per_case.jsonl` 每题一条，含 `case_id`、`chunking_strategy`、`retrieval_method`、`top_k`、`retrieved_evidence`、`hit_at_k`、`reciprocal_rank`、`recall_at_k`、`generated_answer`、`answer_correct`、`citation_valid`、`citation_supported`、`refusal_correct` 等字段。

报告给出 7 项指标（Hit@K、MRR、Recall@K、Answer Correctness、Citation Validity、Citation Support、Refusal Accuracy）的**总体 + 分题型**结果。`citation_valid`（编号是否存在）与 `citation_supported`（内容是否真的支持）分开统计。

## 当前状态（真实进度）

- 数据集：**30/30 题**已完成，题型分布 12/8/5/5，43 条引文全部通过逐字复核。
- 文档覆盖：hash 讲义（PDF 26 条引文、PPTX 1 条）、USTC 6 条、USST 1 条。**UTS 两份文档 403 不可得**，因此没有使用。
- 无答案题：5 条，均记录 `checked_scope`（检查文件、方法、可机器复核的缺失证据）。
- 适配器：`fixture` 可用；`part-b` 已实现但**尚未对真实索引运行**（需要 B 的 `--index-dir` 与本地 `.env`）。
- 实验矩阵：**18 组全部 blocked**（0 ready），原因见 [DESIGN.md §8](DESIGN.md)。
- 真实实验：**一条都没有跑**，`results/` 为空是符合预期的；未完成真实检索与问答实验。
- 代码基线：`2bed52f` + 本目录的本地提交，**尚未包含 PR #3**（C 的拒答/引用修复未合并）。

## 约定

- `results/` 下的报告在人工核对前不提交；仓库根 `.gitignore` 不修改，若要忽略产物请先确认。
- 提交信息遵循仓库 `AGENTS.md`：`<type>(<scope>): <subject>`，本目录用 `test(eval)` / `docs(eval)` / `feat(eval)`。
