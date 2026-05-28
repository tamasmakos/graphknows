# ---------------------------------------------------------------------------
# Graphknows Agent Service
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Security: non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project manifest first for layer caching
COPY services/agent/pyproject.toml ./services/agent/pyproject.toml

# Install Python dependencies (editable install against the repo layout)
COPY services/__init__.py ./services/__init__.py
COPY services/agent/__init__.py ./services/agent/__init__.py
COPY services/agent/session_graph.py ./services/agent/session_graph.py
COPY services/agent/main.py ./services/agent/main.py

RUN pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        httpx \
        "neo4j>=5.19.0" \
        "pydantic>=2.7.0"

# Drop to non-root
USER appuser

# Expose agent port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')"

# Run the service
CMD ["uvicorn", "services.agent.main:app", "--host", "0.0.0.0", "--port", "8002"]
