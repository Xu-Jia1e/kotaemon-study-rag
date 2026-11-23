# evaluation/

离线优先的评测框架，用于衡量本组 RAG 项目的**检索**（B）、**证据化回答 / 引用 / 拒答**（C），以及不同分块方案（A）对检索的影响。

- 不修改 `libs/kotaemon/`、`libs/ktem/`、`app.py` 等核心代码，只以库的方式调用。
- 不调用任何真实模型 API。指标全部是纯函数，可从录制的预测结果离线计算。
- **不编造数据**：真实实验数字只来自实际运行；缺失的原文片段一律标 `pending`，留空等待补录。

## 快速开始

```sh
# 环境（仓库根目录）
uv sync --locked --package kotaemon --package ktem

# 1. 从 test-materials/search-cases.md 生成数据集（幂等，可重复运行）
.venv/Scripts/python.exe -m evaluation.scripts.convert_search_cases

# 2. 校验数据集结构
.venv/Scripts/python.exe -m evaluation.scripts.validate_dataset evaluation/datasets/qa_cases.jsonl

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

# 5. 单测（全部离线）
.venv/Scripts/python.exe -m pytest evaluation -q
```

> 上面第 3、4 步用的是 `tests/fixtures/*_stub.*`，**synthetic fixture，不是实验结果**，只用来证明"数据集 → 适配器 → 指标 → 报告"这条链路通。

## 目录

```
evaluation/
├── README.md              本文件
├── DESIGN.md              接口检查结论、指标定义、实验矩阵、协作事项
├── datasets/
│   ├── schema.md          JSONL 字段规范（含证据分组语义）
│   ├── qa_cases.jsonl     已转换的 6 道 search-cases（原文片段待补）
│   └── candidates.md      如何把 6 题扩充到 20+（只用真实资料）
├── metrics/
│   ├── retrieval.py       Hit@K / MRR@K / 证据组召回@K
│   ├── answer.py          答案正确率（人工标签聚合）/ 拒答指标
│   └── citation.py        引用有效性（自动）/ 引用支持度（人工标签）
├── adapters/
│   ├── base.py            RetrievalAdapter / AnswerAdapter 协议与数据形状
│   ├── part_b.py          B 独立索引的适配器（子进程调用其 CLI）
│   └── stub.py            fixture 适配器（离线链路验证）
├── scripts/
│   ├── convert_search_cases.py
│   ├── validate_dataset.py
│   ├── run_retrieval_eval.py
│   └── run_answer_eval.py
├── tests/                 34 项离线单测 + fixtures
└── results/               运行产物（当前为空，报告需人工确认后再入库）
```

## 与仓库现有资产的关系

| 现有资产 | 关系 |
|---|---|
| `test-materials/search-cases.md` | **数据来源**，由 `convert_search_cases.py` 校验后转换；markdown 改动会让转换失败，防止静默漂移 |
| `test-materials/ingestion-results.json` | A 的入库计数与 6 条锚点；本框架不复用它做指标，只在文档中引用 |
| `scripts/part_b_*.py` | B 的检索实验脚本；本框架不 import、不修改，`PartBRetrievalAdapter` 通过 CLI 调用并解析其 JSON |
| `kotaemon.indices.qa.utils` | C 的引用校验与拒答判定；`metrics/` **直接调用**，保证与产品行为一致 |
| `libs/kotaemon/tests/test_qa_citation.py` | C 的 22 项单测；本框架不重复实现其逻辑，只做指标层封装 |

## 指标口径（详见 DESIGN.md §2）

| 指标 | 定义 | 自动化程度 |
|---|---|---|
| `Hit@K` | 有答案题中，Top-K 含**至少一个**正确证据的题目比例 | 全自动 |
| `MRR@K` | 首个正确证据排名的倒数均值 | 全自动 |
| `evidence_recall@K` | 已覆盖证据**组**数 / 总组数（双证据题看部分覆盖） | 全自动 |
| `answer_accuracy` | 人工标签 `correct=1 / partial=0.5 / incorrect=0` 的均值 | 标签由人工给 |
| `citation_validity_rate` | 非拒答回答中，全部 `【n】` 都能在本次证据里找到的比例 | 全自动（复用 C 的校验） |
| `citation_support_rate` | 被引片段确实支持该句的比例 | 标签由人工给 |
| `refusal_accuracy` | 无答案题中正确拒答的比例 | 全自动（复用 C 的判定） |
| `false_refusal_rate` | 有答案题中被误拒答的比例 | 全自动 |
| `balanced_accuracy` | 上述两者的平衡值，防止"全部拒答"刷分 | 全自动 |

## 当前状态（真实进度）

- 数据集：**6/20** 题，全部为 `answerable`；9 个证据单元、8 个证据组；**9 处原文片段待补**（`quote=null`）。
- 无答案题：**0 条**，需要先确认"库中确实没有"再录入（见 `datasets/candidates.md`）。
- 适配器：`fixture` 可用；`part-b` 已实现但**尚未对真实索引运行**（需要 B 的 `--index-dir` 与本地 `.env` 里的 DashScope key）。
- 真实实验：**一条都没有跑**，`results/` 为空是符合预期的。

## 约定

- `results/` 下的报告在人工核对前不提交；仓库根 `.gitignore` 不修改，若要忽略产物请先确认。
- 提交信息遵循仓库 `AGENTS.md`：`<type>(<scope>): <subject>`，本目录用 `test(eval)` / `docs(eval)` / `feat(eval)`。
