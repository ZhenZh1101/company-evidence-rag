# Design Choices and Boundaries

**Advanced RAG** was selected based on the [reference paper on arXiv](https://arxiv.org/abs/2312.10997v5) (2024-03-27). The paper surveys methods; it does not provide experimental evidence that this is the best approach for these two datasets. Performance must be assessed through local evaluation.

| Decision | Paper reference | Implementation in this project |
|---|---|---|
| Improve processing before and after retrieval | II.B, pp. 3–4, Fig. 3 | Parsing, metadata, hybrid retrieval, evidence selection |
| Preserve context and metadata | III.B, p. 8 | Company, disclosure date, date basis, reporting period, source, page number; chunking adjacent paragraphs together |
| Combine sparse and vector retrieval | III.D.1, p. 9 | SQLite FTS5 BM25 + Qdrant cosine vector retrieval + RRF (RRF is an engineering choice) |
| Control duplication and context length | IV.A, p. 10 | Content deduplication, limits on the share of evidence from any one file, bounded evidence budgets |
| Preserve table integrity | III.A.1, p. 7 | Preserve HTML table rows and columns, PDF layout and page numbers, and repeated headers for long tables. No claim is made that complex charts and tables are fully handled |
| Evaluate retrieval and generation separately | VI.B–D, pp. 12–15 | Retrieval hits/MRR, evidence citation validation, answer refusal; smoke evaluation on real data |

## Dataset Facts

The CBRS root index contains 429 release records; NOC contains 1,560. Ingestion uses only the final `index.json` and each release's `meta.json`. It does not treat `audit/`, crawler scripts, streaming links, or index pages as company document content. Attachments are retained because an earnings page may only list links, with the actual results in a PDF. Full SEC submission TXT files contain encoded binary content and duplicate materials, so the actual HTML filings and attachments take priority.

The same file may appear under multiple local paths, release packages, or formats. When formats are known to be mirrors, one is preferred. Otherwise, separate source records are retained, identical text shares an embedding, and retrieval results remove duplicate text from the same company. Attachments also retain the URL of their parent release page; not all source aliases for duplicate documents have necessarily been merged. A report's accounting period, collection timestamp, or directory date must not automatically be treated as its publication date. Materials with unknown dates are excluded when strict date filters apply.

## Processing Flow

Ingestion → Text and location information → SQLite documents/chunks/FTS → Batch embedding and caching → Synchronization with the Qdrant vector index.

Qdrant is a free, open-source, dedicated vector database running as a local service. SQLite retains source text, metadata, FTS5, and an embedding cache deduplicated by text hash. Qdrant stores vectors per chunk along with company, category, and disclosure-date filter fields, and retrieves them using cosine distance. Each SQLite database gets a separate collection derived from its absolute path to avoid mixing embedding spaces across databases. Additions, replacements, and deletions create pending changes for synchronization. `embed` synchronizes automatically; `sync-vectors` can separately migrate an existing cache and retry interrupted work without calling the embedding API or deleting the old cache. Q&A must still be paused during index updates, and there are no distributed transactions between the two databases.

Deployment uses `docker compose up -d`, exposes port 6333 only on the local machine, disables telemetry, and persists data in `data/qdrant/`. Run `company-rag sync-vectors` after upgrading an old index, moving the SQLite database, or rebuilding Qdrant. Qdrant's index and query planner handle vector searches. HNSW approximate retrieval may change rankings from the previous exact scan, so evaluation should be repeated. Source text and full-text search remain available when Qdrant is offline.

Question → Optional, bounded English query rewriting (retaining the original question and using company names from the corpus) → Company/disclosure-date filtering and company routing for subqueries → Fusion of keyword and vector retrieval → Evidence selection across sources → Answer grounded in evidence → Validation of citation IDs and verbatim source excerpts.

Tests of cross-company comparisons showed rounded summaries displacing exact totals from the retrieved evidence. These questions therefore use the same Chat model to select evidence from up to 48 complete candidate chunks before generating an answer. Single-company Q&A keeps a shorter flow. The selector can return only source IDs already provided and cannot add evidence; generated answers still require verbatim citation validation. Table units must accompany their chunks and must not be supplied from the model's memory.

Models use the local OpenClaw Chat and embedding HTTP endpoints by default. Embedding requests specify `openclaw/llm-gpt55` as the model in the request body and use `x-openclaw-model: openai/text-embedding-3-large` to select the actual model. Both endpoints can also use OpenAI, or OpenAI embeddings can be paired with Z.AI Chat. The live LLM is switched through a separate Chat endpoint, model, key, and request parameters. Changing only Chat does not change the index identity. Changing the embedding endpoint, request-body model, or actual routed model requires rebuilding vectors; old vectors must not be reused. Authentication is read from the environment or local OpenClaw configuration. Local gateway credentials are not automatically forwarded to remote endpoints or a separate Chat endpoint, and keys are not committed to Git. Self-hosted Qdrant incurs no database service fees; model API charges remain the same as under the original configuration.

## Quality Boundaries

Matching a cited excerpt proves only that the excerpt exists, not that the entire answer is correct. Answers must distinguish amounts, currencies, millions/billions, quarterly/YTD periods, GAAP/non-GAAP measures, actuals/guidance, and restated versions. When evidence is insufficient, the system must explicitly decline to answer. Documents are untrusted evidence, not instructions. Unsupported scanned images, audio/video, and extraction failures must be visible in the ingestion report.

Knowledge graphs, fine-tuning, HyDE, unbounded autonomous retrieval, and a separate reranking model are currently out of scope. Additional complexity will be introduced only in response to specific evaluation failures.

Implementation references: [Qdrant local deployment](https://qdrant.tech/documentation/quickstart/), [Qdrant configuration](https://qdrant.tech/documentation/operations/configuration/), [official SQLite FTS5 documentation](https://www.sqlite.org/fts5.html), and [pypdf text extraction documentation](https://pypdf.readthedocs.io/en/stable/user/extract-text.html). pypdf does not perform OCR, and locating text in a PDF does not guarantee reliable reconstruction of every table's structure.
