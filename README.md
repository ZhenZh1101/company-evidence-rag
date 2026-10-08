# Public Company Document RAG

A locally running **Advanced RAG** system with archive-aware document ingestion, hybrid BM25 + vector retrieval, company and disclosure-date filters, an English/Simplified Chinese interface and Q&A, and verifiable source quotations. Vector retrieval uses the free, open-source, self-hosted [Qdrant](https://github.com/qdrant/qdrant), with no cloud account required; SQLite stores documents, the full-text index, and the embedding cache. English is the default language. See the [architecture notes](docs/architecture.md) for design decisions and tradeoffs, and the [reference paper on arXiv](https://arxiv.org/abs/2312.10997v5).

For a hosted UI with all existing data, Z.AI GLM-5.3-Flash, OpenAI embeddings, IP quotas and a bounded queue, follow the [Hugging Face Spaces deployment guide](docs/huggingface.md). The Docker Space reads a private dataset mounted at `/app/space-data` and restores cached vectors; it does not ingest or embed new documents.

## Getting Started

This working directory already has a configured `.venv`. Start Docker, then run the following from this directory:

```bash
.venv/bin/python -m pip install -e '.[test]'
docker compose up -d
.venv/bin/company-rag sync-vectors
.venv/bin/company-rag serve
```

`sync-vectors` synchronizes the existing SQLite embedding cache to Qdrant without calling a model or incurring re-embedding costs. It can be interrupted and rerun; chunks without embeddings still require `embed`. When upgrading, stop the running Q&A service, complete the sync, and restart it. Qdrant uses a pinned image version, stores data in `data/qdrant/`, listens only on local `127.0.0.1:6333` by default, and has telemetry disabled. See the [official Qdrant documentation](https://qdrant.tech/documentation/quickstart/) for startup instructions. The database software is free; existing model API billing remains unchanged.

Open <http://127.0.0.1:8000>. The interface defaults to English on first use. Use the Language selector to switch between English and Simplified Chinese; the browser remembers your choice. The selected language also controls new answers, while source quotations retain their original language. The service listens only on the local machine; use `--port 8001` to change the port.

On a new machine, install Python 3.11 or later and have Docker running with Compose available:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
docker compose up -d
.venv/bin/company-rag doctor
```

`doctor` checks Qdrant, embeddings, and Chat separately, reporting the other results even if one endpoint fails. It makes model calls. To check only whether Qdrant is ready, run `curl --fail http://127.0.0.1:6333/readyz`.

Chat and embeddings both default to `http://127.0.0.1:18789/v1`, with `model` set to `openclaw/llm-gpt55` in the request body. Embedding requests also send `x-openclaw-model: openai/text-embedding-3-large`; this model is used for both documents and questions. Chat requests do not send this header. Credentials are read from `RAG_API_KEY` first; otherwise, `gateway.auth.token` in `~/.openclaw/openclaw.json` is used only for local endpoints. Keys are never written to the database, logs, or Git.

`RAG_EMBEDDING_OPENCLAW_MODEL` controls this embedding header and takes effect only when `RAG_EMBEDDING_MODEL` starts with `openclaw/`. Setting it explicitly to an empty string uses the gateway's default routing. The routed model is also part of the index identity, so changing it requires re-embedding; the application rejects mixing in old vectors.

Available environment variables are listed in [.env.example](.env.example). Export them yourself; the application does not execute `.env` files. `RAG_DB_PATH` defaults to `data/rag.sqlite3` and can be overridden with the global CLI option `--db`; `RAG_QDRANT_URL` defaults to `http://127.0.0.1:6333`. Each SQLite database gets a separate Qdrant collection derived from its absolute path. Run `sync-vectors` again after moving the database or switching to a new Qdrant instance. Keeping the SQLite cache lets you rebuild the vector index, and migration does not delete the original vectors. Changing the embedding endpoint or model requires a new database; do not mix vector spaces. If the server replaces a model under the same name, the application cannot detect it automatically, and the database must be rebuilt.

### Using OpenAI for Both Embeddings and Chat

The following example uses `text-embedding-3-large` and `gpt-4.1-mini`. This configuration applies to document embeddings, live question embeddings, query rewriting, evidence selection, and answer generation across the CLI, web interface, and evaluation script:

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

If `OPENAI_API_KEY` is already set, use `export RAG_API_KEY="$OPENAI_API_KEY"`; the application reads `RAG_API_KEY`. The initial `unset` clears any previous Z.AI Chat overrides. If the key is missing, local OpenClaw credentials are not automatically forwarded to OpenAI.

**Switching embedding providers requires re-embedding.** Use the new database path above, rerun your original `ingest` command, then run `.venv/bin/company-rag embed`. Once complete, start the service with `.venv/bin/company-rag serve`. Keep the original database; do not mix old embeddings and OpenAI embeddings in one index. Use the new database path for subsequent CLI, web, and evaluation processes as well.

The parameters follow the OpenAI documentation for [Embeddings](https://developers.openai.com/api/reference/resources/embeddings/methods/create) and [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create). Chat uses `max_completion_tokens` for the output budget. The example model, [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), supports Chat Completions and does not require reasoning parameters. OpenAI embeddings have an 8192-token limit per input. Current chunking is character-based rather than an exact token count, so unusually long inputs or inputs with many special characters may be rejected by the server; the application does not silently truncate text.

When changing Chat models, use the parameters that model supports. `RAG_CHAT_TEMPERATURE=none` omits temperature entirely. `RAG_CHAT_REASONING_EFFORT` accepts `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` and is omitted when unset, but individual models may not support every value. `max_completion_tokens` includes reasoning tokens. The system budgets 500 tokens for rewriting/selection, 16 for `doctor`, and 4000 for answers, so models that require reasoning may not work within these short budgets. Unset `RAG_CHAT_THINKING` when using OpenAI.

`RAG_CHAT_RESPONSE_FORMAT=json_object` requests JSON output from compatible Chat providers to reduce formatting failures that trigger another model call. Local configurations omit this parameter by default; the Docker Space enables it. Set it to an empty string to omit the parameter, or `text` to request ordinary text output. JSON mode keeps the configured model and existing answer-schema and verbatim-citation validation; invalid answers still receive at most one repair attempt. Embedding requests are unaffected.

### Using Z.AI for the Live LLM

Configure a separate Chat endpoint in the terminal where you start the service. This example uses `glm-4.7`, which supports disabling thinking:

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

Query rewriting, cross-company evidence selection, answer generation, and validation retries in the CLI, web interface, and evaluation script all use this Chat configuration. Document embeddings and live question embeddings still use `RAG_BASE_URL`, `RAG_API_KEY`, and `RAG_EMBEDDING_MODEL`. Changing only Chat does not require rebuilding the database. Restart any running service after setting these environment variables.

For **OpenAI embeddings + Z.AI Chat**, first complete the OpenAI configuration and embedding steps in the previous section, then apply the Chat configuration in this section. Embeddings continue to use the OpenAI key, while Chat uses a separate Z.AI key; the OpenAI vector index does not need to be rebuilt again.

The endpoint and parameters follow the [official Z.AI API documentation](https://docs.z.ai/api-reference/llm/chat-completion). Its [OpenAI-compatible API guide](https://docs.z.ai/guides/develop/openai/python) recommends a temperature greater than 0. Thinking is disabled here to fit the 500-token rewriting/selection limit and the 16-token `doctor` check. When changing models, confirm that they support these parameters; models that only support thinking, such as GLM-5.3, cannot use this example unchanged. `RAG_CHAT_TEMPERATURE` defaults to 0. `RAG_CHAT_THINKING` is omitted when unset and accepts only `enabled` / `disabled` when set.

If `RAG_CHAT_BASE_URL` is unset, Chat uses `RAG_BASE_URL`. Chat prefers `RAG_CHAT_API_KEY`; if unset, it reuses the original key only when both endpoint URLs are identical. A separate Chat endpoint without a key causes an error; local gateway credentials are not sent to remote services. Do not change `RAG_BASE_URL` just to switch the live LLM.

## Ingestion and Updates

```bash
.venv/bin/company-rag ingest \
  /Users/zhenzhang/bigplan/IR_retrieval/CBRS_2024-09-17_2026-09-26 \
  /Users/zhenzhang/bigplan/IR_retrieval/NOC_2024-09-17_2026-09-26 \
  --ocr --report data/ingest-report.json
.venv/bin/company-rag embed --batch-size 256
.venv/bin/company-rag stats
```

Ingestion and embedding are separate so you can inspect parsing results before making model calls. Pause Q&A while updating the index, then restart the service after ingestion and `embed` finish; the initial version does not support switching index versions online. Repeated ingestion skips unchanged files, and identical text shares embeddings. Embedding can resume from saved batches. `embed` automatically splits requests to fit the gateway's character limit, allows at most two concurrent requests, and uses bounded retries with backoff for temporary gateway errors. It automatically synchronizes Qdrant when complete. Pending changes from ingestion, updates, and deletions can also be processed separately with `sync-vectors --batch-size 128`; supported batch sizes are 1–256. If Qdrant is temporarily unavailable, keep the saved embedding cache and retry synchronization after the service recovers.

- Archive directories are interpreted using the root `index.json` and release `meta.json` files. Main text and attachments are retained; crawler audits, index copies, media links, and known converted duplicates are excluded.
- Ordinary directories and individual files can also be ingested: `company-rag ingest /path/to/reports --company AAPL`. Dates remain empty when reliable metadata is unavailable; they are not inferred from filenames.
- Company names and aliases are imported from archive records or the `company` / `company_name` strings and `company_aliases` string array in `meta.json`, for example `{"ticker":"ACME","company":"Acme Corporation","company_aliases":["Acme","Acme Corp"]}`. Automatic recognition uses these names and imported tickers, with no built-in company list or guessed abbreviations. If aliases are missing, enter a ticker directly or select the company. For indexes created before the upgrade, rerun the original `ingest` command to load names and aliases; identical text still reuses the vector cache.
- Supported formats are HTML, TXT, Markdown, PDF, DOCX, PPTX, XLSX, XLS, and CSV. Office tables retain rows, columns, and headers. PDF parsing prefers Poppler's layout-preserving text extraction and falls back to pypdf when Poppler is unavailable.
- `--ocr` enables local Tesseract English recognition for PDF pages without a text layer. It requires `pdftoppm` and `tesseract`, which are already installed on this machine; on a new macOS machine, use `brew install poppler tesseract`. Citations are marked as OCR, and recognized text should still be checked against the original. Keep the same `--ocr` option for incremental updates; changing parsing modes triggers reparsing.
- `--prune` removes index records that no longer exist in the source. If metadata cannot be read, unverified existing records are retained, and only explicitly excluded duplicates may be removed.
- `--limit N` explicitly requests partial ingestion, with at most N files per root directory. Leave it unset for normal use.
- Source discovery or parsing failures produce exit code 2; successfully processed documents are still saved. Detailed failures, skip reasons, and warnings are recorded in the specified JSON report. Other runtime errors produce exit code 1.

Original company materials are read-only and are not copied into Git. Databases, vectors, runtime logs, and ingestion reports are all ignored by Git.

## Questions and Search

```bash
.venv/bin/company-rag ask 'How did CBRS Q1 2026 GAAP revenue differ from core revenue?' --company CBRS
.venv/bin/company-rag ask 'What were NOC sales for Q2 2026 and the first half of 2026, respectively?' --company NOC --language zh-CN
.venv/bin/company-rag search 'Cerebras Q1 2026 GAAP revenue' --company CBRS --mode lexical --no-rewrite
.venv/bin/company-rag ask 'As of 2026-06-23, what was the revenue guidance for Q2?' --company CBRS --date-to 2026-06-23 --language zh-CN
```

`ask` and `search` support `--language en` (the default) and `--language zh-CN`. Questions can be in English or Chinese; answers and search messages use the selected language rather than switching automatically based on the question. Source excerpts, document titles, and source metadata are not translated.

By default, retrieval uses limited English query rewriting while retaining the original question. Without explicit filters, known company names or tickers in the question narrow the scope; all companies are searched only when none are recognized. For cross-company questions, select all target companies or pass `--company` multiple times. For questions spanning metrics or periods, the system attempts to generate up to three subqueries, with subqueries that explicitly refer to one company restricted to that company. Standard retrieval returns ten evidence items by default, limits the share from any one document, and removes identical text. Cross-company Q&A first retrieves up to 48 candidates, then uses the same Chat model to select evidence with explicit units and periods; this adds one model call.

Date filters refer to the archive's **disclosure/event/effective date**, as indicated by each evidence item's `date_basis`, rather than the financial reporting period. Unknown dates are excluded under strict date filtering. This is not equivalent to a complete historical database of information available at a given point in time. For companies absent from the archive, select the company scope explicitly; a retrieval miss does not establish that a fact does not exist.

The answer area automatically renders Markdown headings, bold text, lists, tables, blockquotes, and code blocks. Answers include citations such as `[S1]`, verbatim excerpts, document names, source URLs, and page/table/slide locations. The application validates citation identifiers and source excerpts, retries once on failure, and refuses to display the answer if validation fails again. **The presence of an excerpt does not prove every conclusion is correct.** Financial calculations, complex tables, and conflicting disclosures still require verification.

API example:

```bash
curl http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was CBRS Q1 2026 GAAP revenue?","companies":["CBRS"],"mode":"hybrid","rewrite":true,"top_k":10,"language":"en"}'
```

`language` is optional and defaults to `en`; set it to `zh-CN` for Chinese answers and messages. Unsupported language values return HTTP 422. The response includes the actual `language` used.

Completed query responses also return stage `timings`, `diagnostics`, and, through the web API, a `request_id`. `retrieval_seconds` includes rewriting, index checks, embedding, keyword/vector search, and comparison selection; it is not just database time. `answer_seconds` measures the first answer call and `repair_seconds` the optional second call, including any gateway retries. `total_seconds` covers `RAG.ask`; the web-only `request_seconds` also covers admission waiting, thread scheduling, opening the store, and Markdown rendering. See the [timing field reference](docs/huggingface.md#query-timings) for the remaining fields and log behavior. Dense/hybrid search validates the index once per search; query counts, result ranking, and request concurrency remain unchanged.

`GET /api/stats` and `GET /api/source/{chunk_id}` are also available. Local serving accepts only local hosts; the Docker Space allows its configured public origin. Question admission allows one executing request plus five waiting, with a rolling limit of five questions per minute and fifty per 24 hours per IP. Run exactly one web worker and one replica; see the deployment guide for proxy configuration and rate-counter storage limits.

## Testing and Evaluation

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/evaluate.py --generation --output data/evaluation.json
```

Offline tests use real SQLite and the Qdrant client's in-memory mode, covering retrieval, filtering, incremental updates, embedding validation, answer refusal, path boundaries, and the local API. Normal operation and evaluation use the Qdrant service. The evaluation script compares evidence Hit@10 and MRR for keyword, vector, and hybrid retrieval on the same small set of real questions, and can also run Chinese Q&A and unanswerable questions. The question set comes from two earnings documents and is **not an independent, large-scale benchmark**; it does not support claims of general accuracy in financial Q&A.

## Known Limitations

- The system does not read audio/video content or standalone XML/XBRL, and does not extract every chart embedded in images. Legacy DOC/PPT and unsupported files are listed in the report. The main HTML filings associated with XBRL are usually already ingested.
- Extracting table rows and columns does not guarantee correct interpretation of every complex merged cell, header spanning pages, or footnote. XLSX parsing reads cached values from the file and does not calculate missing formula results.
- Years, GAAP/non-GAAP, quarter/YTD, actuals/guidance, currencies, and scales must be distinguished explicitly. Model-generated arithmetic is not verified by an independent calculator.
- Qdrant performs vector retrieval using cosine distance and company, category, and disclosure-date filters. HNSW provides approximate retrieval, so rankings may differ from the previous exact scan; rerun local evaluations after migration. Dense/hybrid retrieval is unavailable when Qdrant is stopped; full-text search and source text remain in SQLite.
- Local source files may contain prompt injection. The system treats documents as evidence and renders answers through a Markdown parser with raw HTML, images, and reference-style links disabled; model-generated HTML is not executed. Source evidence is still displayed as plain text. The API retains the original `answer` and also returns the safely rendered `answer_html`.
