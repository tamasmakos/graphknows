# Knowledge Graph Project

This repository implements a decoupled Knowledge Graph system consisting of two primary services: **GraphGen** (Generation/ETL) and **GraphRAG** (Retrieval/API).

## Architecture Overview

- **`services/graphgen`**: The ETL pipeline. Responsible for parsing raw data, extracting entities, and building the graph in FalkorDB. Uses Pydantic Settings for configuration and Dependency Injection for core logic.
- **`services/graphrag`**: The Retrieval API. A FastAPI-based agentic system using LlamaIndex to explore the graph and answer user queries.

## Prerequisites

- **Docker** and **Docker Compose**
- **Python 3.11** (if running locally)
- API Keys for LLM providers (Groq, OpenAI, etc.) in a `.env` file.

## Quick Start with Docker Compose

The easiest way to run the entire stack is via Docker Compose:

```bash
# Build and start all services (FalkorDB, Postgres, GraphGen, GraphRAG)
docker-compose up --build -d
```

### Supporting Services
- **FalkorDB**: Primary graph database (port 6379)
- **PostgreSQL**: Vector store for hybrid search (port 5435)
- **Langfuse**: Tracing and observability (port 3000)

## Service Usage

### 1. GraphGen (Pipeline)
To run the ingestion pipeline manually or via Docker:

```bash
# Via Docker
docker-compose run graphgen

# Locally (from services/graphgen)
cd services/graphgen/src
python -m main
```

### 2. GraphRAG (API & UI)
The API server starts automatically with `docker-compose`. 

- **Web UI**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

To run locally:
```bash
cd services/graphrag/src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Development Environment (Dev Container)

This repository includes a VS Code Dev Container configuration. Opening the project in a container will automatically set up the Python environment and all system dependencies.

1. Open the folder in VS Code.
2. Click **"Reopen in Container"** when prompted.
3. Supporting databases will still need to be started via `docker-compose up -d`.

## Configuration

Configuration is managed via **Environment Variables** (loaded from `.env`) using Pydantic Settings:

- `FALKORDB_HOST`: Host for the graph database.
- `POSTGRES_HOST`: Host for pgvector.
- `OPENAI_API_KEY` / `GROQ_API_KEY`: LLM credentials.
- `INPUT_DIR`: Directory for raw data (GraphGen).