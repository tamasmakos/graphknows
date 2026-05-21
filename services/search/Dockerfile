# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps before copying source for layer-cache efficiency
COPY pyproject.toml uv.lock* ./
COPY services/search/pyproject.toml ./services/search/pyproject.toml

RUN uv sync --package graphknows-search --no-install-project --no-cache 2>/dev/null \
    || uv pip install --system \
        fastapi "uvicorn[standard]" httpx "pydantic>=2" pyyaml "mcp[cli]" \
        markdownify pymupdf langdetect numpy neo4j aiosqlite

# Copy full repo context so scrapers/ and services/ are both importable
COPY . .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 10001 app && useradd -u 10001 -g app -s /bin/false -M app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    EP_DEBATES_OUT_DIR=/data/raw/ep_debates \
    EURLEX_OUT_DIR=/data/raw/eurlex \
    UK_LEGISLATION_OUT_DIR=/data/raw/uk_legislation \
    JOB_DB_PATH=/data/jobs.db

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["tini", "--"]
# SEARCH_MODE=mcp overrides to stdio MCP server; default is HTTP
CMD ["uvicorn", "services.search.main:app", "--host", "0.0.0.0", "--port", "8000"]

