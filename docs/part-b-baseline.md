# Part B：纯向量检索基线

## 范围

在仓库提交 `b55043c65cbae34d87cabc4186cfc9ba63565282` 上，使用 `scripts/part_b_baseline.py` 验证 `VectorIndexing` 和 `VectorRetrieval` 的纯向量链路。脚本用 PyMuPDF 将公开的 `11-hash2.pdf` 按页抽取文本，写入临时 Chroma 向量库和内存文档库，检索后清理临时数据。

这是独立的冒烟测试，并非 A 阶段完整入库流水线或其持久化数据库的验收。PDF 的图形、OCR 和位置布局没有在此测试中验证。每次运行会重新调用 embedding API 对样本建库。

## 运行

先按项目 README 安装依赖，并用 `test-materials/download_samples.py` 或其他方式取得经过哈希验证的验收 PDF。准备一个本地环境文件，包含 `DASHSCOPE_API_KEY`；脚本还读取 `EMBEDDING_MODEL_NAME` 和 `EMBEDDING_API_BASE`，密钥不要放入仓库。

```powershell
$env:PYTHONUTF8 = '1'
$env:ANONYMIZED_TELEMETRY = 'False'
.\.venv\Scripts\python.exe scripts\part_b_baseline.py `
  --env-file '配置文件路径' `
  --pdf downloads\real-samples\11-hash2.pdf `
  --question '按二次探测依次把 89、18、49、58、79 插入大小为 10 的空哈希表，哪些槽位有值？' `
  --top-k 5
```

## 一次实际运行结果

2026-11-23，使用百炼 `text-embedding-v4`、2048 维，抽取并索引 37 页。对上面的测试问题，纯向量检索返回：

| 排名 | 页码 | 片段 ID | 检索分数（约） |
|---:|---:|---|---:|
| 1 | 17 | page-17 | 0.6230 |
| 2 | 25 | page-25 | 0.5901 |
| 3 | 30 | page-30 | 0.5200 |
| 4 | 16 | page-16 | 0.5090 |
| 5 | 31 | page-31 | 0.4999 |

仓库的 `test-materials/search-cases.md` 将第 25 页列为目标证据；该页进入 Top-5，排名第 2，表明本次查询召回了预期页面。第 17 页是另一个哈希表示例。本次只验证了一个问题的页面召回，不能据此判断整体准确率。

第 25 页的文本提取保留了数字，但其二维槽位对应关系不能仅靠纯文本顺序确认；需要结合原页面或 A 阶段的结构化处理核对。输出分数来自当前存储实现，不应直接与文本检索或重排分数比较。
