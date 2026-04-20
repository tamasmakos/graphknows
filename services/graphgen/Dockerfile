FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# System deps required for C extensions: leidenalg (igraph), psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all workspace manifests first for layer-cached dependency resolution.
# Both members must be present for UV to understand the workspace structure.
COPY pyproject.toml uv.lock ./
COPY services/graphgen/pyproject.toml ./services/graphgen/pyproject.toml
COPY services/graphrag/pyproject.toml ./services/graphrag/pyproject.toml

# Install graphgen dependencies only (not the project package itself).
# Source is provided at runtime via volume mount (dev) or COPY below (prod).
RUN uv sync --package graphgen --no-install-project --no-cache

# Download NLP models into the venv
RUN uv run python -m spacy download en_core_web_lg
RUN uv run python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords');"

# Copy application source (overridden by volume mount in docker-compose dev mode)
COPY services/graphgen/src ./src

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
