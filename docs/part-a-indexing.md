# 成员 A 入库复现

在仓库根目录运行。依赖安装见 [AGENTS.md](../AGENTS.md)。在当前终端设置
`DASHSCOPE_API_KEY`。如果 Key 属于百炼工作空间，同时将
`EMBEDDING_API_BASE` 设为该工作空间的 OpenAI 兼容地址，包含末尾的
`/compatible-mode/v1`。默认地址为
`https://dashscope.aliyuncs.com/compatible-mode/v1`。

```powershell
uv run --no-sync python test-materials/download_samples.py
$samples = (Get-ChildItem downloads/real-samples -File).FullName
uv run --no-sync python -m scripts.part_a_index --index-dir downloads/index-1024-256 --chunk-size 1024 --chunk-overlap 256 $samples
uv run --no-sync python -m scripts.part_a_index --index-dir downloads/index-512-128 --chunk-size 512 --chunk-overlap 128 $samples
```

命令也接受任意一组本地文件路径。切块参数以 token 计，要求
`0 < chunk-overlap < chunk-size`。每次指定新的索引目录。

入口使用项目默认解析和 OCR。正文按指定参数切分，独立表格沿用读取器的分块规则。
百炼 `text-embedding-v4` 以 2048 维、每请求最多 10 条生成向量。
Chroma 存向量，LanceDB 存切块和 PDF 预览，SQLite 记录源文件与切块及向量的关系。
运行结果写入索引目录的 `index.json`。有文件失败时，命令返回非零并保留现场，复跑时指定新目录。

## 六份真实资料的建库结果

两套索引使用同一批经 SHA-256 核验的样本、同一默认读取模式和同一百炼模型。
表内数字依次为正文块、表格块、向量数。

| 样本                            | 1024 / 256    | 512 / 128     |
| ------------------------------- | ------------- | ------------- |
| UW Hash Tables II PDF           | 37 / 0 / 37   | 37 / 0 / 37   |
| UW Hash Tables II PPTX          | 37 / 0 / 37   | 37 / 0 / 37   |
| UTS 无障碍 Word 指南 DOCX       | 14 / 0 / 14   | 14 / 0 / 14   |
| 上理工课程大纲 DOCX             | 8 / 7 / 15    | 8 / 7 / 15    |
| UTS 无障碍 PowerPoint 指南 PPTX | 25 / 0 / 25   | 29 / 0 / 29   |
| 中科大 ClassIn 指南 PDF         | 17 / 0 / 17   | 22 / 0 / 22   |
| 合计                            | 138 / 7 / 145 | 147 / 7 / 154 |

UTS PowerPoint 第 24 张从 2 块变为 6 块，最长正文块从 869 降到 369 tokens。
中科大 ClassIn PDF 第 8 页从 519 tokens 的 1 块变为 401 和 239 tokens 的 2 块，
两套都保留截图中的「试卷资源」和「授权资源」。UW PDF 第 25 页的哈希槽位对应关系可回查，
UTS Word 的截图文字 `No Spacing` 保留在 `Headings` 章节。

逐条核对了 SQLite 关联、LanceDB 切块与 Chroma 向量。两套分别完成 17 和 18 次百炼请求，
全部向量均为 2048 维且数值有限。每套保留 53 张 PDF 页面预览，预览向量数均为 0。
