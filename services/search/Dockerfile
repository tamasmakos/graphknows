FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY services/search/pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx "pydantic>=2" requests

# Copy full repo context
COPY . .

# Ensure repo root is on PYTHONPATH so both `services` and `scrapers` are importable
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "services.search.main:app", "--host", "0.0.0.0", "--port", "8000"]
