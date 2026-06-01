# ─── builder stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# uv for fast, reproducible installs
RUN pip install --no-cache-dir uv

# Copy workspace manifests first — maximise layer caching
COPY pyproject.toml uv.lock ./
COPY graphknows ./graphknows
COPY frontend ./frontend

# Install all production deps into an isolated venv
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache \
        .

# ─── runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Bring in only the venv and the application source
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/graphknows ./graphknows
COPY --from=builder /build/frontend ./frontend

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user for container security
RUN useradd -m -u 1000 graphknows
USER graphknows

CMD ["graphknows-mcp"]
