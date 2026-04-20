# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY services/graphgen/pyproject.toml ./services/graphgen/pyproject.toml
COPY services/graphrag/pyproject.toml ./services/graphrag/pyproject.toml

# Install graphgen deps into /app/.venv
RUN uv sync --package graphgen --no-install-project --no-cache

# Pre-download NLP assets
RUN uv run python -m spacy download en_core_web_lg
RUN uv run python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords');"

COPY services/graphgen/src ./src

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11.10-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -g 10001 app && useradd -u 10001 -g app -s /bin/false -M app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
