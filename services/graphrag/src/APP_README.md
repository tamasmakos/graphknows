# GraphRAG: Agentic Retrieval Service

The `graphrag` service is a FastAPI-based retrieval system that uses a **LlamaIndex** agent to navigate and query the Knowledge Graph.

## Architecture Overview

- **Agentic Exploration**: Uses a `FunctionAgent` to proactively browse entities, relationships, and topics before answering.
- **Service Isolation**: Contains its own local drivers for FalkorDB and PostgreSQL, ensuring zero dependency on the ingestion pipeline.
- **Pydantic Settings**: Configuration is strictly typed and managed via `AppSettings`.

## Core Components

### 1. The Agent (`src/agent/`)
A LlamaIndex agent equipped with tools to query the graph.
- `llamaindex_agent.py`: Agent definition and tool orchestration.
- `tracing.py`: Langfuse instrumentation and reasoning chain logging.

### 2. Infrastructure (`src/infrastructure/`)
- `graph_db.py`: Database client with hybrid search logic.
- `postgres_store.py`: Read-only driver for pgvector embedding retrieval.
- `llm.py`: Provider-agnostic LLM factory (Groq/OpenAI).
- `config.py`: Configuration factory using `AppSettings`.

### 3. Llama Adapters (`src/llama/`)
Adapts the local graph infrastructure to LlamaIndex's `GraphStore` and `Embedding` interfaces.

## API Endpoints

- `POST /agent/chat`: Proactive agent-based chat.
- `POST /chat`: Direct retrieval pipeline.
- `GET /schema`: Inspect the current graph schema.
- `GET /health`: Service health check.

## Running the Service

### 1. Configuration
Settings are loaded from environment variables:
```env
FALKORDB_HOST=localhost
GROQ_API_KEY=gsk_...
```

### 2. Start the API
```bash
# From services/graphrag/src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Monitoring
This service is instrumented with **Langfuse**. If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are provided, all agent reasoning steps and tool calls will be traced automatically.