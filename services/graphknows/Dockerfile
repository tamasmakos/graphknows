# ─── builder stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# uv for fast, reproducible installs
RUN pip install --no-cache-dir uv

# Copy workspace manifests first — maximise layer caching
COPY pyproject.toml uv.lock ./
COPY services/graphknows/pyproject.toml ./services/graphknows/pyproject.toml

# Install all production deps into an isolated venv
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache \
        -e services/graphknows

# Copy source after deps so code changes don't bust the dependency layer
COPY services/graphknows ./services/graphknows

# ─── runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Bring in only the venv and the application source
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/services/graphknows ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user for container security
RUN useradd -m -u 1000 graphknows
USER graphknows

EXPOSE 8025

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8025/health')" || exit 1

CMD ["python", "-m", "graphknows.run"]
