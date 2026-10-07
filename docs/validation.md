# Local Acceptance Records

## 2026-10-07: Qdrant Integration

The **42,896 chunk vectors** in the default database, `data/rag.sqlite3`, have been synchronized from the SQLite cache to local Qdrant `v1.19.2` without calling the embedding API again. Qdrant reported `status=green` and `points_count=indexed_vectors_count=42896`. Payload indexes for company, category, and disclosure date have been created, and there are 0 pending changes to synchronize; running `sync-vectors` again synchronized 0 records. The original database was backed up to `data/rag-before-qdrant.sqlite3.bak`, and the original embedding cache is still retained. The expansion database was not synchronized as part of this default-database migration; run `sync-vectors` with its path before using it.

**85 offline tests passed**, covering migration of existing databases, resumable retries, deletion and chunk ID reuse, pre-filtering, isolation between databases, and recovery after vector-store loss, partially missing vectors, or reference hash mismatches. Deployment configuration validation passed.

Using the same 9-question retrieval smoke set with query rewriting disabled: BM25 hit 8/9, vector-only retrieval 9/9, and hybrid retrieval 8/9; mean reciprocal ranks were 0.586, 0.737, and 0.844, respectively, with 0 company/date filter violations. These metrics match the pre-migration results in `evaluation-large-20261007.json`; the report is saved in `data/evaluation-qdrant.json`, and migration counts are in `data/qdrant-migration.json`. These results cover only this small test set and do not establish general accuracy or guarantee ANN recall across all queries.

The web service has switched to Qdrant, as confirmed through `/api/stats`. An actual NOC question answered with default query rewriting enabled returned sales of 10,876 million USD for the second quarter and 20,757 million USD for the first half, and passed source-quotation validation; the response is saved in `data/web-smoke-qdrant.json`. A different English phrasing with rewriting disabled did not retrieve direct evidence for the first half and produced a partial refusal, saved in `data/web-smoke-qdrant-no-rewrite.json`; this database migration did not eliminate the effects of question phrasing and retrieval strategy on evidence recall.

## 2026-10-07: Switch to text-embedding-3-large

Embeddings are now requested through local OpenClaw: the request body uses `model=openclaw/llm-gpt55`, and the request header is `x-openclaw-model: openai/text-embedding-3-large`. The measured dimension is 3072, and the actual routed model has been added to index identity validation. Chat requests do not send this header.

Both databases were rebuilt in copies and moved back to their original paths. Snapshots of the old databases and the original files from before the switch are retained in `data/backups/`; paths, counts, and validation summaries are recorded in `data/large-rebuild-20261007.json`.

| Database | Documents | Chunks | Valid unique vectors | Dimensions |
|---|---:|---:|---:|---:|
| `data/rag.sqlite3` | 3,549 | 42,896 | 34,786 | 3,072 |
| `data/expansion-20261007.sqlite3` | 9,605 | 210,703 | 135,270 | 3,072 |

Every chunk in both databases has a vector, and both databases returned `ok` from `quick_check`. SHA-256 hashes of all document and chunk contents match the old database backups, and vector retrieval checks using source text passed. The expansion database reused large-model vectors already generated for the main database, without mixing in vectors from the old model. The web service has been restarted, and the active index identity has been confirmed to include `openai/text-embedding-3-large`.

**71 offline tests passed**, with new coverage for the actual request body and routing header, isolation of Chat and direct API calls, environment variable overrides, and rejection of old-index reuse. Results for the existing 9-question retrieval smoke set are below, with query rewriting disabled and no company or date filter violations:

| Method | All targets hit @10 | Mean reciprocal rank |
|---|---:|---:|
| Keyword BM25 | 8/9 | 0.586 |
| Vector-only | 9/9 | 0.737 |
| Hybrid retrieval | 8/9 | 0.844 |

The report is in `data/evaluation-large-20261007.json`. An actual web question also passed after the restart: NOC sales for the second quarter and first half of 2026 were returned as 10,876 and 20,757 million USD, respectively, with source quotations and no warnings; the response is saved in `data/web-smoke-large-20261007.json`. These remain small, manually selected retrieval and question-answering checks and cannot establish general financial question-answering accuracy. The sections below retain the historical acceptance results from before the migration, using 1536-dimensional embeddings.

## Actual Data and APIs

The two user-provided CBRS and NOC archives have been processed without modifying their source directories. The specified local Chat / embedding APIs and `openclaw/llm-gpt55` were used; embeddings were measured at 1536 dimensions and checked for finite values, nonzero vectors, and dimensional consistency with the index.

| Company | Imported files | Text chunks |
|---|---:|---:|
| CBRS | 674 | 14,861 |
| NOC | 2,875 | 28,035 |
| Total | **3,549** | **42,896** |

All 42,896 chunks have vectors, with identical text sharing 34,786 valid vectors. SQLite `quick_check` returned `ok`, and the FTS record count matches the chunk count. The file count is not the count of distinct releases: multiple attachments or formats can belong to the same release.

Text became extractable from 49 PDFs without a text layer after local OCR, with English OCR applied to 180 pages in total. OCR warnings and other warnings about partially blank or image-only pages are retained in `data/ingest-report-ocr.json`; this does not establish that all image content has been recognized. The final ingestion report, `data/ingest-report-final.json`, records 0 parsing failures, but there is **1 missing RTF attachment listed in the original archive**: `0001959173-26-001026.rtf` for NOC's 2026-02-17 Form 144. This discovery error caused the ingestion command to return 2, while the successfully built index was still saved.

Historical manifests, copies of directory listings, IR homepages incorrectly assigned report dates, and hidden iXBRL technical fields have been excluded. Notes specifying monetary units outside tables are retained in table chunks, and entirely empty HTML layout columns have been removed; units were not filled in from model memory.

## Offline Tests

`python -m unittest discover -s tests -q`: **38 passed**.

Coverage includes archive manifests, path boundaries, HTML/PDF/OCR and table structure, hidden iXBRL, unit notes and table boundaries, incremental updates and FTS cleanup, safe preservation of the index when discovery errors occur, vector batching/order/dimensions, company/date filters, cross-company subquery routing, evidence selection, verbatim quotation validation, refusal to answer, credential isolation, and local API access protection.

## Retrieval Comparison

The 9 manually written questions in `tests/golden.json` were used with query rewriting disabled; each retrieval method returned 10 chunks. A hit requires the number for every specified evidence target to appear in text from the corresponding company. For questions with multiple targets, reciprocal rank is averaged across targets.

| Method | All targets hit @10 | Mean reciprocal rank | Company/date filter violations |
|---|---:|---:|---:|
| Keyword BM25 | 8/9 | 0.586 | 0 |
| Vector-only | 7/9 | 0.402 | 0 |
| Hybrid retrieval | 8/9 | 0.602 | 0 |

This sample does not demonstrate that hybrid retrieval generally outperforms keyword retrieval. With rewriting disabled, the cross-company question still failed to retrieve all the exact evidence; this failure is retained in the report. Actual question answering additionally enables subquery routing and cross-company candidate evidence selection.

## Review of Actual Chinese-Language Questions and Answers

Four answers from the final complete workflow were checked against the local source documents, and all quotations passed programmatic verbatim matching:

| Question | Final result | Local elapsed time |
|---|---|---:|
| CBRS Q1 2026 GAAP and Core revenue | 193,406 / 191,348 thousand USD; correctly distinguished non-GAAP figures and original units | 12.39 seconds |
| NOC Q2 and first-half sales | 10,876 / 20,757 million USD; correctly distinguished three months from six months | 12.00 seconds |
| Cross-company comparison of CBRS Q1 and NOC Q2 | 193.406 / 10,876 million USD; noted the different reporting periods | 18.46 seconds |
| CBRS Q4 2028 actual audited revenue | Explicitly stated that evidence was insufficient; did not invent future-quarter revenue | 15.43 seconds |

The 193.4 / 191.3 million USD figures in the CBRS press release are rounded, while the reconciliation table provides the more precise figures of 193,406 / 191,348 thousand USD; these do not conflict.

Full model outputs, evidence, queries, and timings are in the Git-ignored `data/acceptance.json`. The web application has been started locally and displays "Local index ready"; interactions, narrow-screen layout, safe text rendering, and API request validation have also been checked.

## Limitations

This was a small smoke set derived from two earnings materials, previously used to identify and fix issues, rather than an independently held-out test set. The four successful runs above do not guarantee correctness on repeated calls or all new questions. Quotation matching only proves that the quoted text exists in the source; it cannot automatically establish that every claim is supported. OCR, complex tables spanning multiple pages, differing financial definitions, and model calculations still require human review. Date filtering uses the archive's disclosure/event/effective dates and is not equivalent to full verification of the information available at a historical point in time. Pause question answering while updating the index; the current version does not support switching index versions while the service is online.
