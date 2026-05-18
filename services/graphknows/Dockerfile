FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy workspace manifests first (for layer caching)
COPY pyproject.toml uv.lock ./
COPY libs/python/graphknows-shared ./libs/python/graphknows-shared
COPY services/graphknows/pyproject.toml ./services/graphknows/pyproject.toml

# Install all deps from graphknows's own pyproject.toml
RUN uv pip install --system --no-cache \
    -e services/graphknows \
    -e libs/python/graphknows-shared

# Copy source (after deps for better cache hits)
COPY services/graphknows ./services/graphknows
COPY libs/python/graphknows-shared ./libs/python/graphknows-shared

# Non-root user for security
RUN useradd -m -u 1000 graphknows
USER graphknows

CMD ["python", "-m", "graphknows.run"]
