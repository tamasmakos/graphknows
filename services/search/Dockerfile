FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for layer caching
# Build context is the repo root (see docker-compose.yaml build.context)
COPY services/search/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx "pydantic>=2" requests

# Copy full repo context so scrapers/ and services/ are both importable
COPY . .

# Repo root on PYTHONPATH → `import scrapers.ep_debates` and `import services.search` both work
ENV PYTHONPATH=/app

# Output directories for scraped raw documents (override via env var per adapter)
ENV EP_DEBATES_OUT_DIR=/data/raw/ep_debates \
    EURLEX_OUT_DIR=/data/raw/eurlex \
    UK_LEGISLATION_OUT_DIR=/data/raw/uk_legislation

EXPOSE 8000

CMD ["uvicorn", "services.search.main:app", "--host", "0.0.0.0", "--port", "8000"]
