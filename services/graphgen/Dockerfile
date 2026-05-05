FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy shared package first (layer cache)
COPY packages/shared-py /app/packages/shared-py
RUN pip install --no-cache-dir -e /app/packages/shared-py

# Copy graphgen service
COPY services/graphgen /app/services/graphgen
COPY services/graphgen/pyproject.toml /app/services/graphgen/
RUN pip install --no-cache-dir -e /app/services/graphgen

WORKDIR /app/services/graphgen

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "graphgen.api:app", "--host", "0.0.0.0", "--port", "8003"]
