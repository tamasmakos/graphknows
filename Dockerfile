# ─── builder stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# uv for fast, reproducible installs
RUN pip install --no-cache-dir uv

# Copy workspace manifests first — maximise layer caching
COPY pyproject.toml uv.lock ./
COPY graphknows ./graphknows
COPY frontend ./frontend

# Install the batteries-included package (all extras) + dev/eval groups.
# Core deps are lean; the ML/NLP/transport stack lives behind extras (see
# pyproject [project.optional-dependencies]); [all] pulls them for the server.
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache \
        .[all] \
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

# Bring in only the venv and the application source
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/graphknows ./graphknows
COPY --from=builder /build/frontend ./frontend

# spaCy model for GLiNER2 schema mining — installed INTO the venv the app uses
# (a plain `spacy download` targets the system python, which the app never runs).
RUN uv pip install --python /opt/venv/bin/python \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_CONTROL=manual

# Non-root user for container security
RUN useradd -m -u 1000 graphknows
USER graphknows

CMD ["graphknows-mcp"]
