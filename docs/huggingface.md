# Deploying to Hugging Face Spaces

The target is [zhenzh/company-evidence-rag](https://huggingface.co/spaces/zhenzh/company-evidence-rag), configured as a **Protected Docker Space on CPU Basic**. The application retains question answering, company/date filters, retrieval modes, Chinese/English interfaces, and the source evidence viewer. It has no document upload, parsing, OCR, or document-embedding endpoints.

Code belongs in the Space repository. The complete data snapshot belongs in the **private** dataset [zhenzh/company-evidence-rag-data](https://huggingface.co/datasets/zhenzh/company-evidence-rag-data), mounted read-only at `/app/space-data` and pinned to an exact commit. The Space rejected the approximately **15.22 GB** data upload with `Storage limit reached for this space (Max: 1 GB)`, so the Docker image does not copy or contain the dataset.

HF supports native read-only mounts of private dataset repositories. Free accounts include 100 GB of private storage, subject to remaining account quota; datasets have no separate per-repository size limit. Mount setup uses the authenticated local client; the application needs no `HF_TOKEN` secret or download code. [Volume documentation](https://huggingface.co/docs/hub/spaces-storage), [storage limits](https://huggingface.co/docs/hub/storage-limits).

Protected visibility hides the Space's repository while exposing its application URL. Application-level username/password authentication now protects the corpus and APIs. The private dataset is not publicly downloadable; signed-in users can read evidence returned by the application. [Visibility documentation](https://huggingface.co/docs/hub/spaces-overview#space-visibility).

## Friend accounts

Share <https://zhenzh-company-evidence-rag.hf.space> with friends. They need an account issued by the owner, not a Hugging Face account. The HF wrapper shows an "Open app" link because third-party iframe cookies are not reliable. The direct app URL supports the login form and a Sign out button.

Docker sets `RAG_AUTH_REQUIRED=1`. The **Secret** `RAG_AUTH_USERS_JSON` maps usernames to salted PBKDF2-SHA256 password hashes (600,000 iterations). Missing or invalid required configuration prevents startup; it never silently makes the app public. No password hashes are sent to the browser. Only the login page and login/logout endpoints are public; statistics, source passages, questions and session information require authentication.

Add an account or change its password locally (password prompts are hidden):

```bash
.venv/bin/python scripts/manage_users.py USERNAME
```

Remove a user's access while retaining at least one owner account:

```bash
.venv/bin/python scripts/manage_users.py USERNAME --remove
```

The owner-only, Git-ignored file `data/access-users.json` is the source of truth for account hashes. Do not upload it to either repository. Publish its contents only as the Space Secret:

```bash
tmp/hf-upload-venv/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi
HfApi().add_space_secret(
    'zhenzh/company-evidence-rag', key='RAG_AUTH_USERS_JSON',
    value=Path('data/access-users.json').read_text(),
    description='Individual application accounts; salted password hashes only',
)
PY
```

Secret changes restart the Space and revoke all existing sessions. Sessions use opaque random cookies (`HttpOnly`, `Secure` on HTTPS, `SameSite=Lax`), expire after 12 hours, and are revoked on logout. Each IP can make 10 login attempts per rolling minute; successful and failed attempts both count. Login hashing is limited to two concurrent checks. These limits are separate from question quotas. Sessions and login throttles live in the single application process, with at most 1,000 sessions. The service still requires exactly one worker/replica.

## Runtime and snapshot

FastAPI listens on `0.0.0.0:7860`; Qdrant listens only inside the container. Run exactly **1 application worker and 1 Space replica**. The README specifies `sdk: docker`, `app_port: 7860`, and `startup_duration_timeout: 1h` to allow restoration of the full vector index.

| Path | Purpose |
| --- | --- |
| `/app/space-data/rag.sqlite3` | Read-only dataset snapshot: text, full-text index, embedding cache, metadata |
| `/app/space-data/archive/*.tar.gz` | Five compressed archives of original documents and metadata |
| `/app/space-data/manifest.json` | Counts, file sizes, SHA-256 hashes, and packaging exclusions |
| `/app/runtime/rag.sqlite3` | Writable database copied from the mounted snapshot at startup |
| `/app/runtime/rate-limits.sqlite3` | Question quota records on ephemeral runtime storage |

Startup copies SQLite to local runtime storage and restores Qdrant from the existing vector cache. It makes no model calls, generates no embeddings, and does not extract or parse the archives. Evidence text comes from SQLite; source links point to the original public sources. Live SQLite databases are not placed on the dataset mount.

The 2026-10-07 snapshot uses `data/expansion-20261007.sqlite3`: **9,605 documents, 210,703 chunks, 135,270 deduplicated vectors, and 3072 dimensions**. Every chunk has a cached vector.

| Company | Documents |
| --- | ---: |
| AMKR | 1,054 |
| CBRS | 674 |
| NOC | 2,875 |
| NOK | 4,361 |
| VST | 641 |

The package includes 40,304 original document/metadata files in five archives. Collector logs, audit output, caches, code, and hidden files are excluded and recorded in the manifest. Runtime disk must also accommodate the working SQLite database and Qdrant; the raw chunk vectors alone occupy approximately 2.59 GB.

## 1. Prepare or verify the data

The existing `space-data` package is ready. To create a fresh package, run from the repository root, using a new output directory if `space-data` already exists:

```bash
.venv/bin/python scripts/prepare_space.py \
  --db data/expansion-20261007.sqlite3 \
  --output space-data
```

Packaging leaves the original database unchanged. It checks integrity, cache coverage, embedding identity, and inclusion of every indexed original file. It converts the verified OpenClaw route for `openai/text-embedding-3-large` to the direct OpenAI identity in the deployment copy, preserving the original identity as metadata. It does not accept arbitrary models or dimensions as compatible. Check the counts, archive list, and hashes in `space-data/manifest.json` before uploading.

## 2. Configure provider secrets

The two provider keys have already been saved in the target Space's **Settings → Variables and secrets → Secrets**:

| Secret | Purpose |
| --- | --- |
| `RAG_API_KEY` | OpenAI embeddings for incoming questions and rewritten queries |
| `RAG_CHAT_API_KEY` | Z.AI query rewriting, comparison selection, and answers |

Keep keys out of source files, manifests, upload folders, and Git. Z.AI uses the standard API endpoint and its API quota; a Coding Plan subscription does not establish entitlement for this RAG application. [Z.AI FAQ](https://docs.z.ai/devpack/faq).

Docker supplies these non-secret defaults; Space Variables can override them:

| Variable | Default |
| --- | --- |
| `RAG_BASE_URL` | `https://api.openai.com/v1` |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-large` |
| `RAG_CHAT_BASE_URL` | `https://api.z.ai/api/paas/v4` |
| `RAG_CHAT_MODEL` | `glm-5.3-flash` |
| `RAG_CHAT_THINKING` | `enabled` |
| `RAG_CHAT_REASONING_EFFORT` | `low` |
| `RAG_CHAT_TEMPERATURE` | `1` |
| `RAG_CHAT_TOKEN_LIMIT_FIELD` | `max_tokens` |
| `RAG_CHAT_MIN_TOKENS` | `8192` |
| `RAG_CHAT_RESPONSE_FORMAT` | `json_object` |

GLM-5.3-Flash requires thinking and supports `low`, `high`, or `max` effort. The 8192-token floor applies to doctor, rewriting, selection, answer, and repair calls. Real calls with this model and reasoning budget passed the checks recorded below. [Model documentation](https://docs.z.ai/guides/vlm/glm-5.3-flash), [API parameters](https://docs.z.ai/api-reference/llm/chat-completion).

Docker requests JSON object output to reduce format failures and their extra repair call. This keeps the existing Chat model and all answer-schema and verbatim-citation checks. Set `RAG_CHAT_RESPONSE_FORMAT` to an empty string to omit the parameter, or `text` to request ordinary text output; local configurations omit it by default. `doctor` uses a JSON prompt when JSON mode is enabled. Verify provider compatibility during deployment acceptance; JSON output alone does not establish valid citations.

## 3. Upload the private dataset and attach its revision

Use the isolated deployment client, **huggingface_hub 2.1.1**. The application's older client does not have `set_space_volumes`; there is no need to upgrade the application environment. The local HF CLI login has already succeeded. For a fresh machine, run `hf auth login` with the same client before proceeding.

```bash
.venv/bin/python -m venv tmp/hf-upload-venv
tmp/hf-upload-venv/bin/python -m pip install 'huggingface_hub==2.1.1'
tmp/hf-upload-venv/bin/hf auth whoami
```

From the repository root, upload the **contents** of `space-data` to the **dataset root**, then attach the uploaded commit. The following preserves unrelated mounts and replaces only the mount at `/app/space-data`:

```bash
tmp/hf-upload-venv/bin/python - <<'PY'
from huggingface_hub import HfApi, Volume

api = HfApi()
space_id = 'zhenzh/company-evidence-rag'
dataset_id = 'zhenzh/company-evidence-rag-data'
api.create_repo(dataset_id, repo_type='dataset', private=True, exist_ok=True)
if not api.dataset_info(dataset_id).private:
    raise RuntimeError('The snapshot dataset must be private before uploading.')

commit = api.upload_folder(
    repo_id=dataset_id,
    repo_type='dataset',
    folder_path='space-data',
    path_in_repo='',
    allow_patterns=['rag.sqlite3', 'manifest.json', 'archive/*.tar.gz'],
    commit_message='Upload complete corpus and existing embedding cache',
)
mounts = [volume for volume in (api.get_space_runtime(space_id).volumes or [])
          if volume.mount_path != '/app/space-data']
mounts.append(Volume(
    type='dataset', source=dataset_id, revision=commit.oid,
    mount_path='/app/space-data', read_only=True,
))
api.set_space_volumes(space_id, volumes=mounts)
print('Dataset revision:', commit.oid)
PY
```

Record the printed commit SHA. Confirm the dataset revision contains `rag.sqlite3`, `manifest.json`, and all five archives, and that the Space runtime shows that revision mounted read-only. `set_space_volumes` replaces the complete mount list, which is why existing unrelated mounts are retained. If uploading is interrupted, rerun the upload. [Upload API](https://huggingface.co/docs/huggingface_hub/guides/upload), [mount API](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#set_space_volumes).

## 4. Upload code separately

After attaching the dataset, upload only the application's explicit source allowlist. Do not upload `space-data` to the Space or add it to the Docker build context:

```bash
tmp/hf-upload-venv/bin/python - <<'PY'
from huggingface_hub import HfApi

HfApi().upload_folder(
    repo_id='zhenzh/company-evidence-rag',
    repo_type='space',
    folder_path='.',
    path_in_repo='',
    allow_patterns=[
        'README.md', 'Dockerfile', '.dockerignore', 'pyproject.toml',
        'rag/*.py', 'rag/static/*', 'scripts/prepare_space.py', 'docs/*.md',
    ],
    commit_message='Deploy application with mounted private dataset',
)
PY
```

This code commit triggers the Docker build. Check Build/Container logs until snapshot copying, Qdrant synchronization, and web startup finish. A missing `/app/space-data/rag.sqlite3` means the data mount or its repository layout needs correction; embedding the dataset into the image is not required.

## 5. Accept the hosted deployment

The hosted build, serving state, and actual HF proxy behavior still require verification. Open both the embedded Space UI and its direct `.hf.space` address, then check:

1. All five companies and the manifest's document/chunk counts appear. Language switching, company/date filters, keyword/vector/hybrid modes, and evidence expansion work.
2. A Chinese single-company question and a two-company comparison complete with verified citations and without rewrite/selection fallback warnings. An unsupported future-period question returns insufficient evidence. `doctor` calls both paid model APIs; do not use it as a liveness probe.
3. The sixth question from one IP in a rolling 60-second window and the 51st in a rolling 24-hour window return **429** with `Retry-After`. These are rolling windows, not midnight resets.
4. There is **one running request and at most five FIFO waiters**; overflow returns **503** with `Retry-After`. Use multiple test IPs to avoid hitting the per-IP limit before testing queue capacity.

Limits apply to `POST /api/ask`; browsing does not consume quota. Queue overflow does not consume quota. Queued disconnects release their slot and refund quota. Started requests consume quota and retain their execution slot until model work finishes, including after a client disconnect or upstream failure.

### Query timings

Completed `/api/ask` responses include a `request_id`, `timings` in seconds, and `diagnostics`. The same fields appear in a `Query metrics` log entry. These metrics are emitted only when the answer handler returns successfully, including an evidence refusal; failed HTTP requests do not receive a complete stage report. Metrics logs contain request IDs, mode, counters, timings, and warning codes, without question text, source text, usernames, or credentials.

| Timing | Meaning |
| --- | --- |
| `rewrite_seconds` | Query planning, its Chat call, and parsing the returned queries. |
| `index_check_seconds` | Embedding identity and vector-index readiness checks, once per dense/hybrid search. |
| `embedding_seconds` | Embedding the original and rewritten queries in a batch. |
| `lexical_seconds`, `vector_seconds` | Accumulated keyword and vector searches across query/company groups. |
| `selection_seconds` | Optional cross-company evidence selection and response parsing. |
| `retrieval_seconds` | The whole retrieval pipeline, including all the stages above plus filtering, fusion, and source selection. |
| `answer_seconds` | First answer-generation call, including any gateway HTTP retries and backoff. |
| `repair_seconds` | Optional second answer-generation call after failed validation, including its gateway retries and backoff. |
| `validation_seconds` | Local answer JSON, schema, and citation checks across attempts; normally a small fraction of a second. |
| `total_seconds` | Time inside `RAG.ask`, excluding the web admission queue and store setup. |
| `queue_seconds` | Admission work/waiting and thread scheduling before the answer handler starts. |
| `request_seconds` | Time from API admission through queueing, store setup, RAG execution, and Markdown rendering; excludes response delivery to the browser. |

Parent totals already include their component stages; do not add them again. Small uninstrumented overhead and rounding can prevent exact equality. A skipped stage reports zero. `diagnostics` records `query_count`, `lexical_searches`, `vector_searches`, `answer_attempts` (including any repair), and `repair_attempts`. `validation_failures` contains the one-based answer attempt and stage (`json` or `answer_schema_or_citation`), without the failed output or source excerpt. The answer/repair counters count model-call attempts, not HTTP retries.

Index readiness is checked once per dense/hybrid search instead of before every vector subquery. This change preserves query counts, retrieval rankings, and the one-running-request concurrency limit. Use the stage measurements to choose further changes; JSON mode and fewer index checks do not imply a fixed speedup.

### Proxy, origin, and restart caveats

HF's `SPACE_HOST` sets the exact public origin; use `RAG_PUBLIC_ORIGIN=https://your-domain` for a custom domain. The home page permits HF iframe embedding while the API requires same-origin requests. Do not use a wildcard origin. [HF environment variables](https://huggingface.co/docs/hub/spaces-overview#built-in-environment-variables), [embedding documentation](https://huggingface.co/docs/hub/spaces-embed).

`RAG_TRUSTED_PROXY_IPS` defaults to private/loopback CIDRs. The app preserves the socket peer and examines `X-Forwarded-For` from right to left, stopping at the first untrusted address. On the real HF route, verify that a forged XFF prefix cannot bypass the quota and that two public IPs receive independent quotas. Local proxy tests do not establish production behavior; verify any additional proxy CIDRs before trusting them. [Proxy trust documentation](https://fastapi.tiangolo.com/advanced/behind-a-proxy/).

Quota records survive a process restart only while the same runtime disk remains. **HF container rebuilds, migrations, or cleared ephemeral storage can reset quotas; the current deployment cannot guarantee the 50-question limit across these events.** Strict enforcement across container replacements or multiple replicas requires an external store with atomic updates. Do not assume an object-backed bucket supports SQLite locking. The read-only dataset keeps the corpus snapshot available independently of runtime storage. [Space storage documentation](https://huggingface.co/docs/hub/spaces-storage).

## Updating the corpus

Parse and embed new documents locally, create and verify a new complete package, upload it to the private dataset, and change the mount to the new commit SHA. Do not run `ingest` or `embed` inside the Space.

Startup copies the snapshot only when the working database is absent. If runtime storage survives the update, changing the mount alone leaves the old working corpus in use. Stop the service, back up and replace the working database, or select a fresh `RAG_DB_PATH`; then restart to restore Qdrant and verify counts against the new manifest. Never replace a database while the service is running.

## Validation record — 2026-10-07

- Username/password access control: **110 automated tests passed** after adding authentication. Hosted checks verified anonymous API denial, bad-password rejection, successful login, secure cookie attributes, authenticated corpus/question access, disabled API schema, and invalidation of old cookies after logout. Chrome login/logout and the HF iframe's standalone-login link also passed.
- **99 automated tests passed**, covering quotas, queue/disconnect behavior, proxy parsing, model settings, packaging, and citation repair feedback.
- The local Docker runtime restored all 210,703 chunks into a real Qdrant instance. Retrieval/filtering passed for all five companies; all 40,304 archive members and 9,605 indexed original paths were verified.
- Real OpenAI calls returned 3072-dimensional vectors. Five cached-vector comparisons each had cosine similarity above 0.9995. Real Z.AI GLM-5.3-Flash calls passed with the deployment settings.
- Live hybrid Chinese single-company answers and insufficient-evidence handling passed. A comparison initially joined a heading and row into a noncontiguous quote and was correctly rejected. Repair feedback now identifies the offending citation index/source; repeated live hybrid comparisons subsequently passed with verified quotations.
- The hosted Space is **RUNNING on CPU Basic** at <https://zhenzh-company-evidence-rag.hf.space>. All six large artifacts match the manifest's sizes and SHA-256 hashes. The read-only dataset mount is pinned to `352c3b2e3a74d57d030c05c35616845a053b416b`.
- Hosted API checks passed for the complete corpus, same-origin protection, iframe headers, missing ingestion routes, real single-company and cross-company hybrid answers, and source retrieval. Chrome verified Chinese UI, company/date filters, a real answer and expanded source evidence.
- On the real HF route, five requests succeeded and the sixth returned HTTP 429 despite changing forged XFF prefixes. Independent-IP isolation and the full global queue boundary are covered by automated tests; the attempted two-family cloud burst could not test independent clients because the IPv6 connection to this host used an IPv4-mapped address. It correctly encountered the per-IP limit before filling all five waiting slots.
