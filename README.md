# 上市公司文档 RAG

本地运行的 **Advanced RAG**：归档感知的文档导入、BM25 + 向量混合检索、公司与披露日期筛选、英语/简体中文界面和问答、可核查的原文引用。默认语言为英语。方案依据和取舍见 [架构说明](docs/architecture.md)，参考论文为目录内 `2312.10997v5.pdf`。

## 启动

当前工作目录已配置 `.venv`。在本目录运行：

```bash
.venv/bin/company-rag serve
```

打开 <http://127.0.0.1:8000>。界面初次打开默认使用英语，可通过 Language 切换英语和简体中文，并在浏览器中记住选择；所选语言同时控制新回答的语言，原文引用保持来源语言。服务仅监听本机；端口可通过 `--port 8001` 修改。

新机器需要 Python 3.11 或更高版本：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/company-rag doctor
```

Chat 和 embedding 默认都使用 `http://127.0.0.1:18789/v1`，请求体的 `model` 为 `openclaw/llm-gpt55`。Embedding 额外发送 `x-openclaw-model: openai/text-embedding-3-large`，文档和问题均使用此模型；Chat 不发送该请求头。优先读取环境变量 `RAG_API_KEY`，否则仅对本机地址读取 `~/.openclaw/openclaw.json` 的 `gateway.auth.token`。密钥不会写入数据库、日志或 Git。

`RAG_EMBEDDING_OPENCLAW_MODEL` 控制上述 embedding 请求头，仅在 `RAG_EMBEDDING_MODEL` 以 `openclaw/` 开头时生效；显式设为空字符串时使用网关默认路由。实际路由模型也计入索引身份，切换后必须重新向量化，程序会拒绝混用旧向量。

可用环境变量列在 [.env.example](.env.example)；需要自行 `export`，程序不执行 `.env` 文件。`RAG_DB_PATH` 默认 `data/rag.sqlite3`，CLI 全局 `--db` 可覆盖。切换 embedding endpoint/model 需要新数据库；不要混用向量空间。同名模型在服务端被替换时，程序无法自动识别，需重新构建数据库。

### Embedding 和 Chat 都使用 OpenAI

下面以 `text-embedding-3-large` 和 `gpt-4.1-mini` 为例，配置会同时用于文档向量化、在线问题向量化、查询改写、证据筛选和回答生成，覆盖 CLI、Web 及评测脚本：

```bash
unset RAG_CHAT_BASE_URL RAG_CHAT_API_KEY RAG_CHAT_THINKING RAG_CHAT_REASONING_EFFORT
export RAG_BASE_URL=https://api.openai.com/v1
export RAG_API_KEY='your-openai-api-key'
export RAG_EMBEDDING_MODEL=text-embedding-3-large
export RAG_CHAT_MODEL=gpt-4.1-mini
export RAG_CHAT_TOKEN_LIMIT_FIELD=max_completion_tokens
export RAG_CHAT_TEMPERATURE=0
export RAG_DB_PATH=data/rag-openai.sqlite3
.venv/bin/company-rag doctor
```

如果已经设置 `OPENAI_API_KEY`，可用 `export RAG_API_KEY="$OPENAI_API_KEY"`；程序读取的是 `RAG_API_KEY`。开头的 `unset` 会清除先前 Z.AI 的 Chat 覆盖配置。密钥缺失时不会向 OpenAI 自动转发本机 OpenClaw 凭据。

**切换 embedding 服务需要重新向量化。** 使用上面的新数据库路径，将原来的 `ingest` 命令重新运行一次，再执行 `.venv/bin/company-rag embed`，完成后用 `.venv/bin/company-rag serve` 启动。原数据库保留；不要把旧 embedding 和 OpenAI embedding 混入同一索引。新数据库路径同样需用于后续 CLI、Web 和评测进程。

参数依据 OpenAI Docs 的 [Embeddings 接口](https://developers.openai.com/api/reference/resources/embeddings/methods/create) 和 [Chat Completions 接口](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。Chat 使用 `max_completion_tokens` 指定输出预算；示例模型为支持 Chat Completions 的 [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini)，不需要推理参数。OpenAI embedding 单条输入上限为 8192 tokens；当前分块按字符划分，不是精确 token 计数，特殊字符密集或过长输入可能被服务端拒绝，程序不会静默截断文本。

更换 Chat 模型时，以该模型支持的参数为准：`RAG_CHAT_TEMPERATURE=none` 可以完全省略温度；`RAG_CHAT_REASONING_EFFORT` 可选 `none`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max`，未设时不发送，但各模型不一定支持全部值。`max_completion_tokens` 包含推理 tokens；系统的改写/筛选预算为 500，`doctor` 为 16，回答为 4000，因此要求强制推理的模型不能保证适配这些短预算。使用 OpenAI 时应取消 `RAG_CHAT_THINKING`。

### 在线 LLM 使用 Z.AI

在启动服务的终端配置独立的 Chat 接口；下面以支持关闭 thinking 的 `glm-4.7` 为例：

```bash
unset RAG_CHAT_REASONING_EFFORT
export RAG_CHAT_BASE_URL=https://api.z.ai/api/paas/v4
export RAG_CHAT_API_KEY='your-zai-api-key'
export RAG_CHAT_MODEL=glm-4.7
export RAG_CHAT_TOKEN_LIMIT_FIELD=max_tokens
export RAG_CHAT_TEMPERATURE=0.1
export RAG_CHAT_THINKING=disabled
.venv/bin/company-rag doctor
.venv/bin/company-rag serve
```

CLI、Web 和评测脚本中的查询改写、跨公司证据筛选、回答生成及校验重试都会使用此 Chat 配置。文档向量化和在线问题向量化仍使用 `RAG_BASE_URL`、`RAG_API_KEY`、`RAG_EMBEDDING_MODEL`；仅切换 Chat 不需要重新建库。已有服务需在设置环境变量后重启。

如需 **OpenAI embedding + Z.AI Chat**，先完成上一节的 OpenAI 配置和向量化，再执行本节的 Chat 配置即可。embedding 继续使用 OpenAI 密钥，Chat 使用独立的 Z.AI 密钥；无需再重建 OpenAI 向量索引。

接口地址和参数依据 [Z.AI 官方接口文档](https://docs.z.ai/api-reference/llm/chat-completion)；[OpenAI 兼容接口说明](https://docs.z.ai/guides/develop/openai/python) 建议使用大于 0 的 temperature。这里关闭 thinking，适配改写/筛选的 500-token 上限和 `doctor` 的 16-token 检查。更换模型时需确认其支持这些参数；只支持 thinking 的模型（如 GLM-5.3）不能直接套用本例。`RAG_CHAT_TEMPERATURE` 默认 0，`RAG_CHAT_THINKING` 未设时不发送，设置时仅接受 `enabled` / `disabled`。

未指定 `RAG_CHAT_BASE_URL` 时沿用 `RAG_BASE_URL`；`RAG_CHAT_API_KEY` 优先用于 Chat，未设时仅在两个接口地址相同的情况下复用原密钥。独立 Chat 地址缺少密钥会报错，不会将本地网关凭据发送到远程服务。不要为切换在线 LLM 而修改 `RAG_BASE_URL`。

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
- 公司名称和别名从归档记录或 `meta.json` 的 `company` / `company_name`（字符串）及 `company_aliases`（字符串数组）导入，例如 `{"ticker":"ACME","company":"Acme Corporation","company_aliases":["Acme","艾克米"]}`。自动识别使用这些名称和已导入的 ticker，不内置公司名单，也不猜测简称；缺少别名时可直接输入 ticker 或选择公司。升级前的索引需重新运行原有 `ingest` 命令以载入名称和别名，相同文本仍复用向量缓存。
- 支持 HTML、TXT、Markdown、PDF、DOCX、PPTX、XLSX、XLS、CSV。Office 表格保留行列和表头；PDF 优先 Poppler 的版式文本提取，没有 Poppler 时使用 pypdf。
- `--ocr` 为没有文字层的 PDF 页面启用本地 Tesseract 英文识别，需要 `pdftoppm` 和 `tesseract`（本机已安装；新 macOS 可用 `brew install poppler tesseract`）。引用会标注 OCR，识别结果仍需对照原件。增量更新应保留同样的 `--ocr` 选项；切换解析模式会重新解析。
- `--prune` 同步删除已不在来源中的索引记录。发现元数据读取错误时保留未能确认的旧记录，只允许清理明确排除的副本。
- `--limit N` 是显式的部分导入，每个根目录最多 N 份文件。正常使用不要设置。
- 发现或解析来源失败时退出码为 2，成功的文档仍已保存；详细失败、跳过原因和警告位于指定 JSON 报告中。其他运行错误退出码为 1。

原始公司材料只读，不会拷贝进 Git。数据库、向量、运行日志、导入报告均在忽略范围内。

## 提问与检索

```bash
.venv/bin/company-rag ask 'How did CBRS Q1 2026 GAAP revenue differ from core revenue?' --company CBRS
.venv/bin/company-rag ask 'NOC 2026 年第二季度销售额与上半年累计销售额分别是多少？' --company NOC --language zh-CN
.venv/bin/company-rag search 'Cerebras Q1 2026 GAAP revenue' --company CBRS --mode lexical --no-rewrite
.venv/bin/company-rag ask '截至 2026-06-23 对第二季度的收入指引是多少？' --company CBRS --date-to 2026-06-23 --language zh-CN
```

`ask` 和 `search` 支持 `--language en`（默认）与 `--language zh-CN`。问题可以使用英语或中文；回答及检索提示使用指定语言，不根据问题语言自动切换。原文摘录、文档标题和来源元数据不翻译。

默认使用有限英文检索改写，同时保留原问题。未显式筛选时会按问题中的已知公司名称或代码缩小范围；未识别到才搜索全部公司。跨公司问题应勾选全部目标公司，或重复传入 `--company`；跨指标或跨期间问题会尝试生成至多三条子查询，并将明确指向单一公司的子查询限制在该公司。普通检索默认返回十条证据，限制单份文件占比并去除相同文本。跨公司问答先召回最多 48 条候选，再用同一 Chat 模型选择包含明确单位与期间的证据；这会增加一次模型调用。

日期筛选指归档给出的**披露/活动/生效日期**，具体含义见每条证据的 `date_basis`，不是财务会计期间。未知日期在严格日期筛选下被排除；这也不等价于完整的历史“当时可得信息”数据库。对于归档未包含的公司，请明确选择公司范围；不要将未检索到解释为事实不存在。

回答区域自动渲染 Markdown 标题、加粗、列表、表格、引用块及代码块。答案附 `[S1]` 等引用、逐字原文、文档名、源 URL、页码/表格/幻灯片定位。程序验证引用编号和原文摘录，失败则重试一次，再失败则拒绝展示答案。**摘录存在不代表已经证明每项结论正确**，尤其是金额计算、复杂表格和冲突披露仍需核查。

API 示例：

```bash
curl http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was CBRS Q1 2026 GAAP revenue?","companies":["CBRS"],"mode":"hybrid","rewrite":true,"top_k":10,"language":"en"}'
```

`language` 可省略，默认 `en`；设为 `zh-CN` 可获取中文回答与提示。不支持的语言值返回 HTTP 422。响应包含实际使用的 `language`。

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
