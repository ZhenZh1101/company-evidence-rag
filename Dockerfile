FROM qdrant/qdrant:v1.19.2 AS qdrant
FROM python:3.11-slim-trixie AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends libunwind8 tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /app/runtime /app/space-data && chown app:app /app /app/runtime /app/space-data
COPY --from=qdrant /qdrant/qdrant /usr/local/bin/qdrant
COPY --from=qdrant /qdrant/config /app/config
WORKDIR /app
COPY pyproject.toml ./
COPY rag ./rag
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1 \
    RAG_AUTH_REQUIRED=1 \
    RAG_DB_PATH=/app/runtime/rag.sqlite3 \
    RAG_RATE_LIMIT_DB_PATH=/app/runtime/rate-limits.sqlite3 \
    RAG_QDRANT_URL=http://127.0.0.1:6333 \
    RAG_BASE_URL=https://api.openai.com/v1 \
    RAG_EMBEDDING_MODEL=text-embedding-3-large \
    RAG_CHAT_BASE_URL=https://api.z.ai/api/paas/v4 \
    RAG_CHAT_MODEL=glm-5.3-flash \
    RAG_CHAT_THINKING=enabled \
    RAG_CHAT_REASONING_EFFORT=low \
    RAG_CHAT_TEMPERATURE=1 \
    RAG_CHAT_MIN_TOKENS=8192 \
    RAG_CHAT_TOKEN_LIMIT_FIELD=max_tokens \
    RAG_CHAT_RESPONSE_FORMAT=json_object \
    RAG_TRUSTED_PROXY_IPS=127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,::1/128,fc00::/7 \
    QDRANT__SERVICE__HOST=127.0.0.1 \
    QDRANT__STORAGE__STORAGE_PATH=/app/runtime/qdrant \
    QDRANT__STORAGE__SNAPSHOTS_PATH=/app/runtime/snapshots \
    QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=1 \
    QDRANT__TELEMETRY_DISABLED=true
USER app
EXPOSE 7860
ENTRYPOINT ["/usr/bin/tini", "--", "python", "-m", "rag.space"]

# Hugging Face mounts the private dataset read-only at /app/space-data.
# Local runs use: --mount type=bind,source=.../space-data,target=/app/space-data,readonly
