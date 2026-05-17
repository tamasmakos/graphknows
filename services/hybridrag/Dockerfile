FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy workspace and service manifests
COPY pyproject.toml uv.lock ./
COPY libs/python/graphknows-shared ./libs/python/graphknows-shared
COPY services/hybridrag/pyproject.toml ./services/hybridrag/pyproject.toml

# Install dependencies
RUN uv pip install --system --no-cache -e services/hybridrag -e libs/python/graphknows-shared

# Copy service source
COPY services/hybridrag ./services/hybridrag

ENV PYTHONPATH=/app/services/hybridrag

CMD ["python", "-m", "hybridrag.run"]
