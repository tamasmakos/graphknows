# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps before copying source for layer-cache efficiency
COPY pyproject.toml uv.lock* ./
COPY services/search/pyproject.toml ./services/search/pyproject.toml

# Always install into .venv so runtime PATH is predictable
RUN uv sync --package graphknows-search --no-install-project --no-cache

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

USER app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    EP_DEBATES_OUT_DIR=/data/raw/ep_debates \
    EURLEX_OUT_DIR=/data/raw/eurlex \
    UK_LEGISLATION_OUT_DIR=/data/raw/uk_legislation \
    JOB_DB_PATH=/data/jobs.db

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "services.search.main:app", "--host", "0.0.0.0", "--port", "8000"]
