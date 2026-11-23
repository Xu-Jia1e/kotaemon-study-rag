# QA 测试集格式（v1）

一行一个 JSON 对象（JSONL），UTF-8，`\n` 结尾。字段分三类：**问题**、**参考答案**、**证据**。

## 字段定义

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | ✅ | 稳定 ID，`sc-01`…`sc-06` 来自 `test-materials/search-cases.md`；新增题用 `qa-xxx` 并保证唯一 |
| `source` | string | ✅ | 出处，形如 `test-materials/search-cases.md#1`，用于追溯，**不可臆造** |
| `question` | string | ✅ | 用户问题原文 |
| `question_type` | string | ✅ | `answerable` / `unanswerable` / `partially_answerable`。Hit@K 只统计 `answerable`；拒答指标需要 `unanswerable` 与 `answerable` 两类 |
| `category` | string | ⬜ | 题型：`direct`（直接事实）/ `synthesis`（综合理解，需多组证据）/ `detail`（细节定位）/ `unanswerable`。用于报告题型分布 |
| `reference_answer` | string | ✅ | 参考答案（人类可读，允许中文长句） |
| `key_points` | string[] | ⬜ | 人工判分用的要点清单；用于把"部分正确"拆成可核对的小点 |
| `expected_files` | string[] | ✅ | 本题应命中的文件（用于给检索适配器加文件过滤，也用于一致性检查） |
| `gold_evidence` | object[] | ✅（`answerable`） | 支持答案的证据单元，见下；`unanswerable` 必须为 `[]` |
| `evidence_status` | string | ✅ | `locator_only`（只有位置）/ `quote_verified`（原文片段已核对）/ `pending`（待补）/ `not_applicable`（无答案题） |
| `checked_scope` | object | ✅（`unanswerable`） | 无答案题的检查记录，见下 |
| `notes` | string | ⬜ | 待补事项、已知歧义 |

## `gold_evidence` 单元

| 字段 | 类型 | 说明 |
|---|---|---|
| `file` | string | 文件名，取自 `test-materials/` 的正式文件名（如 `11-hash2.pdf`） |
| `locator` | object | `{"kind": "page"\|"slide"\|"section"\|"table_row", "value": 25}`；页码/幻灯片号是整数，章节用原文标题 |
| `quote` | string \| null | 原文片段。**未核对时必须是 `null`，禁止编造** |
| `quote_status` | string | `pending` / `verified` |
| `group` | string | 证据组。**同组内任一条命中即算该组满足**；不同组必须全部满足（多问句题、双证据题） |
| `chunk_id` | string \| null | 可选。只有确实记录过时才填；不同分块方案 chunk id 不同，**不能作为唯一的公平判据** |

## 分组语义（为什么需要）

`Hit@K` 只看"有没有命中任意证据"，而双证据题（如 `sc-02` 的"前四个偏移"+"保证能找到空槽"）需要区分**部分命中**：

- `sc-02`：组 `offsets`（第 24 张）+ 组 `guarantee`（第 26 张 **或** 第 27 张，二者同组）；
- `sc-05`：组 `tab`（第 4 张）+ 组 `banded`（第 17 张），两组都必须命中。

`evidence_recall@K = 已覆盖组数 / 总组数`，因此备选证据不会被算成"漏检"，双证据题也不会因为命中一半而拿满分。

## 为什么不用 chunk ID 做判据

两种分块方案（1024/256 与 512/128）会生成不同的 chunk id 和边界，同一段原文可能落在不同块里。**公平对比必须锚定「文件 + 页码/章节」**，chunk id 只作为可选快速路径。

## 最小示例

```json
{"id": "sc-01", "source": "test-materials/search-cases.md#1",
 "question": "按二次探测依次把 89、18、49、58、79 插入大小为 10 的空哈希表，哪些槽位有值？",
 "question_type": "answerable", "reference_answer": "第 25 页。0→49、2→58、3→79、8→18、9→89，其余为空。",
 "expected_files": ["11-hash2.pdf"],
 "gold_evidence": [{"file": "11-hash2.pdf", "locator": {"kind": "page", "value": 25},
                    "quote": null, "quote_status": "pending", "group": "slots"}],
 "evidence_status": "locator_only", "notes": "原文片段待补录"}
```

## 无答案题（`unanswerable`）

无答案题不能只写"查不到"，必须留下**可复核的检查记录**，否则无法区分"真的没有"和"没找"。

```json
{"id": "v2-u02", "question_type": "unanswerable", "gold_evidence": [],
 "evidence_status": "not_applicable",
 "checked_scope": {
   "files": ["11-hash2.pdf", "11-hash2.pptx", "USTC-ClassIn.pdf", "USST-syllabus.docx"],
   "method": "检索 cuckoo/Hopscotch，只在第 34 页命中一次名称列举；实现细节短语全部无命中",
   "max_mentions": {"cuckoo": 1},
   "absent_phrases": ["两个哈希函数", "三次哈希", "relocation", "搬迁", "踢出"]
 }}
```

`checked_scope` 三选一（至少一项），都会被 `verify_evidence.py` 机器复核：

| 字段 | 含义 |
|---|---|
| `absent_phrases` | 一旦出现就说明题目其实有答案的短语，必须全库无命中 |
| `max_mentions` | 某个词的提及次数上限（如只出现过一次、且只是名称） |
| `blank_fields` | 模板字段存在但未填写：`{file, row_keywords, must_not_contain_digits}` |

## 校验与复核

```sh
# 结构校验（字段、题型、证据规则、无答案题的检查范围）
.venv/Scripts/python.exe -m evaluation.scripts.validate_dataset \
    evaluation/datasets/qa_cases_v2.jsonl --min-cases 24

# 引文逐字复核：重新抽取 4 份文档，断言 quote 出现在指定页/张/章节
.venv/Scripts/python.exe -m evaluation.scripts.verify_evidence
```

`verify_evidence.py` 是防止编造引文的硬门禁：引文对不上、或 `absent_phrases` 意外命中，都会以非零退出。
