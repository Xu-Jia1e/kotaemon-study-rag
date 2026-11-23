# Part B：六份真实资料入库与检索验收

## 测试范围

2026-11-23 使用仓库 `test-materials/download_samples.py` 下载并按提供的 SHA-256 核对六份公开资料。用项目 `IndexDocumentPipeline` 的默认读取器、OCR、切块与入库流程，以及 `DocumentRetrievalPipeline` 查询；SQLite、Chroma、LanceDB 和文件副本均在独立的 `ktem_app_data/part_b/project-pipeline-v3`，不是队友设备上的数据库。embedding 为百炼 `text-embedding-v4`、2048 维，切块大小 1024、重叠 256，单次检索限定对应文件，纯向量 Top-5，无重排或 LLM 生成。

## 入库结果

六份资料均完成入库，数量与仓库 README 中这六份样本的记录一致。文档记录还包括 PDF 缩略图，缩略图不写入向量库。

| 资料 | 文档记录 | 向量记录 |
|---|---:|---:|
| UW Hash Tables II PDF | 74 | 37 |
| UW Hash Tables II PPTX | 37 | 37 |
| UTS Word 指南 | 14 | 14 |
| 上海理工大学教学大纲 DOCX | 15 | 15 |
| UTS PowerPoint 指南 | 25 | 25 |
| 中科大 ClassIn PDF | 33 | 17 |

## 检索结果

采用仓库 `test-materials/search-cases.md` 的六个问题，对 Top-5 返回的正文和位置做人工证据核对；这不是最终生成回答的准确率。

| 资料 | 核对结果 |
|---|---|
| UW PDF | 第 25 页排第 2，片段中的表格保留了槽位 0、2、3、8、9 与 49、58、79、18、89 的对应关系。 |
| UW PPTX | **部分命中**：第 24 张排第 1，包含 1、4、9、16；第 26 或 27 张的保证条件未进入复合问题的 Top-5。 |
| UTS Word | `Headings` 章节排第 1，包含 `Normal`、`No Spacing` 及 `Heading 1`–`Heading 3`。 |
| 上海理工大学 DOCX | 桥梁示例表格排第 1，包含课程目标 1、4、5，以及观看视频和撰写报告的任务。 |
| UTS PowerPoint | 第 4 张排第 1，包含 `Review`；第 17 张排第 5，OCR 文本包含 `Banded Columns`。 |
| 中科大 PDF | 第 8 页排第 1，包含课件格式以及截图 OCR 的“试卷资源”“授权资源”。 |

因此，五题的预期证据均在各自的 Top-5 内，UW PPTX 复合问题部分命中。六题的结果均未出现重复片段 ID，返回分数按降序排列，且每个结果 ID 均能对应所选源文件的向量记录。

将 UW PPTX 的复合问题拆成两问后，偏移问题的目标第 24 张排第 2；保证条件问题的目标第 27 张排第 3。拆问仅用于诊断，不代表原复合问题已经通过，也没有自动实现问题拆分。

## 本次发现和修复

1. PDF 关联预览图时，原代码用无序集合重组结果，且写入 `_id` 而实际 ID 字段为 `id_`；同一缩略图关联多个切块时还会覆盖切块。现按原始结果顺序处理预览图，保留每个切块 ID、正文和分数。
2. 缓存的 RapidOCR 引擎在单行识别后保留 `use_det=False` 等状态，影响后续截图识别；现按引擎配置恢复整图识别模式。修复前后分别重建索引，Word `No Spacing`、PPT `Banded Columns` 和中科大 PDF 两项截图文字均恢复。独立小样本证明了状态污染，六份资料的重新入库验证了实际结果。
3. 之前已修复文本与向量重叠导致的重复结果；新增与原有入库、切块、检索相关测试共 15 项通过，其中包括人工拆问后的 RRF 去重融合测试。

## 使用与复现

先按 README 安装项目依赖，下载并校验六份样本。建库命令会调用 embedding API，OCR 在本地执行；目录必须不存在：

```powershell
$env:PYTHONUTF8 = '1'
$env:ANONYMIZED_TELEMETRY = 'False'
$envFile = '你的本地配置文件路径'
$indexDir = 'ktem_app_data/part_b/project-pipeline-v3'

# 使用新目录重新建库并执行固定的六道验收题：
.\.venv\Scripts\python.exe scripts\part_b_project_pipeline.py `
  --env-file $envFile --index-dir 'ktem_app_data/part_b/new-index' `
  --samples-dir downloads\real-samples

# 复用已有索引，查询自定义问题，结果保存为索引目录下的 results-query.json：
.\.venv\Scripts\python.exe scripts\part_b_project_pipeline.py --query-only `
  --env-file $envFile --index-dir $indexDir --samples-dir downloads\real-samples `
  --file 11-hash2.pptx --mode vector --top-k 5 `
  --question '当表大小为素数且负载因子小于一半时，二次探测能保证什么？'
```

`--mode` 可选 `vector` 或原项目的 `hybrid`；`--query-only --diagnostic` 运行拆问诊断。完整六题结果存放在索引目录的 `results.json`，诊断结果为 `results-diagnostic.json`。跨页复合问题的 Top-7 与人工拆问 RRF 复测见 [复合问题检索复测](part-b-compound-query-evaluation.md)。截图 OCR 有误识别的可能，仍需与原文核对。接入队友实际数据库和自动拆解复合问题均未完成。
