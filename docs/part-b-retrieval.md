# Part B：本地检索

`scripts/part_b_retrieval.py` 有两个命令：`build` 从 PDF 创建持久化索引，`query` 查询已经创建的索引。建库时逐页提取文本，以文件 SHA-256 和页码生成片段 ID；查询只对问题调用 embedding，不会重新向量化文档。

这是独立的本地样本索引，向量使用 Chroma，正文使用 LanceDB，索引清单记录来源、模型和维度；它不是 Kotaemon 应用或成员 A 已有数据库的直接适配。当前只支持可提取文本的 PDF，扫描页、图文位置关系和其他格式仍待后续接入。检索结果是片段，不是生成式回答。

## 使用

按仓库 README 安装 Python 3.10 和项目依赖。准备一个本地 `.env` 文件，包含 `DASHSCOPE_API_KEY`，可包含 `EMBEDDING_MODEL_NAME` 和 `EMBEDDING_API_BASE`；当前固定使用百炼 `text-embedding-v4`、2048 维。不要将密钥写入仓库。

在项目根目录执行：

```powershell
$env:PYTHONUTF8 = '1'
$env:ANONYMIZED_TELEMETRY = 'False'
$envFile = '你的本地配置文件路径'
$indexDir = 'ktem_app_data/part_b/my-index'

# 每个 --pdf 添加一份资料；索引目录必须不存在。
.\.venv\Scripts\python.exe scripts\part_b_retrieval.py build `
  --env-file $envFile --index-dir $indexDir `
  --pdf downloads\real-samples\11-hash2.pdf

# 在现有索引中查询；--file 可省略，省略时搜索索引内所有资料。
.\.venv\Scripts\python.exe scripts\part_b_retrieval.py query `
  --env-file $envFile --index-dir $indexDir --file 11-hash2.pdf `
  --question '按二次探测依次把 89、18、49、58、79 插入大小为 10 的空哈希表，哪些槽位有值？' `
  --mode vector --top-k 5
```

`--mode` 可选 `vector`（默认）、`hybrid` 或 `rrf`。原混合模式并行查询文本与向量候选，先放只被文本命中的结果，再放向量候选，最后截取 Top-K；没有做排序融合，也没有启用重排。

`rrf` 是独立入口中的实验方案：分别查询两类候选（每路最多 20 条，若 Top-K 大于 20 则候选数等于 Top-K），按 ID 去重，按照每路排名计算 `1 / (60 + 排名)` 并相加，按融合分排序。两个分支当前顺序执行；并列时优先向量排名，再以片段 ID 确保顺序稳定。这个模式不改变应用原有混合检索，也没有调用重排模型。

查询结果以 JSON 输出到标准输出，包含问题、模型、检索方式及各结果的 `id`、`file_name`、`page`、`section`、`score`、`score_type` 和 `text`。原混合模式仅文本命中的占位分数 `-1` 输出为 `null`、类型为 `text_unscored`。RRF 模式的 `score` 是融合分，另提供 `vector_rank`、`text_rank` 和 `vector_score`；不同类型的分数不能直接比较。运行日志输出到标准错误，可用 `> result.json` 保存结果。

每次 `build` 需要新的目录，不会覆盖已有索引；如中途失败，目录中没有完整清单，查询会拒绝使用。重新构建时请选择另一个新目录。查询会检查模型配置、向量数量及取回的文档 ID；向量分数仅表示当前存储实现的相似度，不能直接和其他检索或重排分数比较。

## 本地验证

2026-11-23 建立了一个包含公开样本 PDF（37 页）和一页测试资料的索引，保存于 `ktem_app_data/part_b/sample-v1`。在独立进程重复查询该索引，哈希表测试题返回页码 17、25、30、16、31，预期页 25 排名第 2；按第二份文件筛选时只返回该文件的一页。输出样例保存在 `ktem_app_data/part_b/sample-query.json`。

离线防护测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p test_part_b_retrieval.py -v
```

测试覆盖页码与 ID、已有索引不覆盖、拒绝不完整索引及文档缺失时拒绝错配分数。LlamaIndex 的导入可能检查 NLTK 数据；这些测试本身不调用 embedding API。
