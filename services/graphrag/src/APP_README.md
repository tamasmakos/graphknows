# GraphRAG: Agentic Retrieval Service

The `graphrag` service is a FastAPI-based retrieval system that uses a **LlamaIndex** agent to navigate and query the Knowledge Graph.

## Architecture Overview

- **Agentic Exploration**: Uses a `ReActAgent` with 4 Neo4j tools to proactively browse entities, relationships, and document chunks before answering.
- **Service Isolation**: Contains its own Neo4j driver; zero dependency on the ingestion pipeline.
- **Pydantic Settings**: Configuration is strictly typed and managed via `AppSettings`.

## Core Components

### 1. The Agent (`src/agent/`)
A LlamaIndex `ReActAgent` equipped with Neo4j tools.
- `workflow.py`: `run_agent()` and `stream_agent()` entry points.
- `tools.py`: `search_chunks`, `get_entity_neighbours`, `get_document_context`, `search_entities`.

### 2. Infrastructure (`src/infrastructure/`)
- `neo4j_driver.py`: Async Neo4j driver factory.
- `llm.py`: Provider-agnostic LLM factory (Groq/OpenAI).
- `config.py`: Configuration factory using `AppSettings`.

### 3. MCP Server (`src/mcp/`)
An MCP server exposing `kg_chat` and `kg_schema` tools for use with MCP-compatible clients (internal dev tool).

## API Endpoints

- `POST /chat`: Stream or non-stream agent chat.
- `GET /schema`: Inspect the current graph schema from Neo4j.
- `GET /health`: Service health check.

## Running the Service

### 1. Configuration
Settings are loaded from environment variables:
```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme
GROQ_API_KEY=gsk_...
```

### 2. Start the API
```bash
# From services/graphrag/
uvicorn src.main:app --reload --host 0.0.0.0 --port 8010
```

## Monitoring
This service is instrumented with **Langfuse**. If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are provided, all agent reasoning steps and tool calls will be traced automatically.