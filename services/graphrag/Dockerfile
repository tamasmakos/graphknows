FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# System deps required for C extensions: psycopg2-binary, torch build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all workspace manifests first for layer-cached dependency resolution.
# Both members must be present for UV to understand the workspace structure.
COPY pyproject.toml ./
COPY services/graphgen/pyproject.toml ./services/graphgen/pyproject.toml
COPY services/graphrag/pyproject.toml ./services/graphrag/pyproject.toml

# Install graphrag dependencies only (not the project package itself).
# Source is provided at runtime via volume mount (dev) or COPY below (prod).
RUN uv sync --package graphrag --no-install-project --no-cache

# Copy application source (overridden by volume mount in docker-compose dev mode)
COPY services/graphrag/src ./src

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
