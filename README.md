# 上市公司文档 RAG

本地运行的 **Advanced RAG**：归档感知的文档导入、BM25 + 向量混合检索、公司与披露日期筛选、中文问答、可核查的原文引用。方案依据和取舍见 [架构说明](docs/architecture.md)，参考论文为目录内 `2312.10997v5.pdf`。

## 启动

当前工作目录已配置 `.venv`。在本目录运行：

```bash
.venv/bin/company-rag serve
```

打开 <http://127.0.0.1:8000>。服务仅监听本机；端口可通过 `--port 8001` 修改。

新机器需要 Python 3.11 或更高版本：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/company-rag doctor
```

Chat 和 embedding 默认都使用 `http://127.0.0.1:18789/v1` 下的 `openclaw/llm-gpt55`。优先读取环境变量 `RAG_API_KEY`，否则仅对本机地址读取 `~/.openclaw/openclaw.json` 的 `gateway.auth.token`。密钥不会写入数据库、日志或 Git。

可用环境变量列在 [.env.example](.env.example)；需要自行 `export`，程序不执行 `.env` 文件。`RAG_DB_PATH` 默认 `data/rag.sqlite3`，CLI 全局 `--db` 可覆盖。切换 embedding endpoint/model 需要新数据库；不要混用向量空间。同名模型在服务端被替换时，程序无法自动识别，需重新构建数据库。

## 导入与更新

```bash
.venv/bin/company-rag ingest \
  /Users/zhenzhang/bigplan/IR_retrieval/CBRS_2024-09-17_2026-09-26 \
  /Users/zhenzhang/bigplan/IR_retrieval/NOC_2024-09-17_2026-09-26 \
  --ocr --report data/ingest-report.json
.venv/bin/company-rag embed --batch-size 256
.venv/bin/company-rag stats
```

导入和向量化分开，可以检查解析结果后再调用模型。更新索引时暂停问答，完成导入和 `embed` 后再启动服务；首版不提供在线索引版本切换。重复导入跳过未变更文件；内容相同的文本共享 embedding。向量批次保存后即可断点续跑。`embed` 自动按网关字符上限拆分请求，最多两个并发请求，并对网关暂时错误进行有限退避重试。

- 归档目录以根 `index.json` 和发布 `meta.json` 为依据，保留正文及附件，排除爬虫审计、索引副本、媒体链接和已知转换副本。
- 普通目录或单文件也可导入：`company-rag ingest /path/to/reports --company AAPL`。没有可靠元数据时日期为空，不根据文件名猜日期。
- 支持 HTML、TXT、Markdown、PDF、DOCX、PPTX、XLSX、XLS、CSV。Office 表格保留行列和表头；PDF 优先 Poppler 的版式文本提取，没有 Poppler 时使用 pypdf。
- `--ocr` 为没有文字层的 PDF 页面启用本地 Tesseract 英文识别，需要 `pdftoppm` 和 `tesseract`（本机已安装；新 macOS 可用 `brew install poppler tesseract`）。引用会标注 OCR，识别结果仍需对照原件。增量更新应保留同样的 `--ocr` 选项；切换解析模式会重新解析。
- `--prune` 同步删除已不在来源中的索引记录。发现元数据读取错误时保留未能确认的旧记录，只允许清理明确排除的副本。
- `--limit N` 是显式的部分导入，每个根目录最多 N 份文件。正常使用不要设置。
- 发现或解析来源失败时退出码为 2，成功的文档仍已保存；详细失败、跳过原因和警告位于指定 JSON 报告中。其他运行错误退出码为 1。

原始公司材料只读，不会拷贝进 Git。数据库、向量、运行日志、导入报告均在忽略范围内。

## 提问与检索

```bash
.venv/bin/company-rag ask 'CBRS 2026 年第一季度 GAAP 收入与 core 收入有什么区别？' --company CBRS
.venv/bin/company-rag ask 'NOC 2026 年第二季度销售额与上半年累计销售额分别是多少？' --company NOC
.venv/bin/company-rag search 'Cerebras Q1 2026 GAAP revenue' --company CBRS --mode lexical --no-rewrite
.venv/bin/company-rag ask '截至 2026-06-23 对第二季度的收入指引是多少？' --company CBRS --date-to 2026-06-23
```

默认使用有限英文检索改写，同时保留原问题。未显式筛选时会按问题中的已知公司名称或代码缩小范围；未识别到才搜索全部公司。跨公司问题应勾选全部目标公司，或重复传入 `--company`；跨指标或跨期间问题会尝试生成至多三条子查询，并将明确指向单一公司的子查询限制在该公司。普通检索默认返回十条证据，限制单份文件占比并去除相同文本。跨公司问答先召回最多 48 条候选，再用同一 Chat 模型选择包含明确单位与期间的证据；这会增加一次模型调用。

日期筛选指归档给出的**披露/活动/生效日期**，具体含义见每条证据的 `date_basis`，不是财务会计期间。未知日期在严格日期筛选下被排除；这也不等价于完整的历史“当时可得信息”数据库。对于归档未包含的公司，请明确选择公司范围；不要将未检索到解释为事实不存在。

回答区域自动渲染 Markdown 标题、加粗、列表、表格、引用块及代码块。答案附 `[S1]` 等引用、逐字原文、文档名、源 URL、页码/表格/幻灯片定位。程序验证引用编号和原文摘录，失败则重试一次，再失败则拒绝展示答案。**摘录存在不代表已经证明每项结论正确**，尤其是金额计算、复杂表格和冲突披露仍需核查。

API 示例：

```bash
curl http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"CBRS 第一季度 2026 GAAP 收入是多少？","companies":["CBRS"],"mode":"hybrid","rewrite":true,"top_k":10}'
```

另有 `GET /api/stats` 和 `GET /api/source/{chunk_id}`。界面和 API 为本机单用户使用设计，不是公网多用户服务。

## 测试与评测

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/evaluate.py --generation --output data/evaluation.json
```

离线测试覆盖真实 SQLite 检索、过滤、增量更新、embedding 校验、拒答、路径边界和本机 API。评测脚本在同一小型真实问题集上对比关键词、向量、混合检索的证据 Hit@10 和 MRR，并可运行中文问答与无答案问题。问题集取自两份业绩材料，**不是独立大规模 benchmark**，不能据此宣称财务问答普遍准确。

## 已知边界

- 不读取音视频内容、图片中的所有图表或独立 XML/XBRL；旧 DOC/PPT 和不支持的文件会在报告中列出。XBRL 对应的 HTML 主申报通常已导入。
- 表格的行列文本并不保证每个复杂合并单元格、跨页表头或脚注均被正确理解。XLSX 读取文件缓存值，不计算缺失的公式结果。
- 年份、GAAP/non-GAAP、季度/YTD、实际/指引、币种与数量级需要明确区分。模型生成的算术没有独立计算器验证。
- 向量采用分批精确扫描，内存受批大小限制；扩大公司数量后，应根据实际延迟决定是否迁移 ANN。
- 本地原件可能包含提示注入；系统把文档当作证据，回答通过禁用原始 HTML、图片及引用式链接的 Markdown 解析器渲染，不执行模型 HTML。原文证据仍按纯文本显示。API 保留原始 `answer`，另返回安全渲染后的 `answer_html`。
