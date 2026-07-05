# ─── builder stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# uv for fast, reproducible installs
RUN pip install --no-cache-dir uv

# Copy workspace manifests first — maximise layer caching
COPY pyproject.toml uv.lock ./
COPY graphknows ./graphknows
COPY frontend ./frontend

# Install all production deps + dev/eval groups into an isolated venv
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache \
        .[assisted] \
        --group dev \
        --group eval \
        jupyterlab

# ─── runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Install uv in the runtime so it's available in the workspace
RUN pip install --no-cache-dir uv

# Ensure `git` is available inside runtime containers (workspace, mcp, etc.)
# Keep apt lists cleaned to reduce image size.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# User requested spacy installation and model download at the compose/docker layer
RUN pip install spacy && \
    python -m spacy download en_core_web_lg

# Bring in only the venv and the application source
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/graphknows ./graphknows
COPY --from=builder /build/frontend ./frontend

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_CONTROL=manual

# Non-root user for container security
RUN useradd -m -u 1000 graphknows
USER graphknows

CMD ["graphknows-mcp"]
