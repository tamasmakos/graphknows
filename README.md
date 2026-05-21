# Knowledge Graph Project

This repository implements a decoupled, agentic Knowledge Graph system designed to transform unstructured documents into a queryable semantic memory.

It consists of two primary microservices:
- **GraphGen** (Generation/ETL): A pipeline that parses documents, extracts entities using GLiNER + LLMs, resolves coreferences, detects communities, and stores everything in Neo4j.
- **GraphRAG** (Retrieval/API): An agentic retrieval system powered by **LlamaIndex** `ReActAgent` that navigates the graph to answer complex user queries.

---

## Developer Quickstart

> One command to start the full local stack and verify everything is healthy.

### Prerequisites

- Docker + Docker Compose (for infrastructure: Neo4j, pgvector, Memgraph)
- Python venvs already installed under `services/agent/.venv`
- `.env` file populated (copy from `.env.example`)

### Start everything

```bash
cd /root/graphknows
make dev-local
```

This single command:
1. Checks Neo4j, pgvector, and Memgraph are reachable — offers to `docker compose up -d` them if not
2. Kills any zombie processes on app ports (8025, 8002, 8040)
3. Starts all 4 application service watchdogs in the background
4. Waits up to 90s for each to pass its health check
5. Prints the full status board

To skip the Cloudflare tunnel:

```bash
make dev-local-no-tunnel
```

### Check health at any time

```bash
make status
```

Output covers all 8 services:

```
=== Graphknows Health Check ===

Infrastructure
  OK    Neo4j HTTP     (http://172.18.0.1:7474)
  OK    Neo4j Bolt     (172.18.0.1:7687)
  WARN  pgvector       (172.18.0.1:5432 — TCP open, pg_isready not available)
  OK    Memgraph Bolt  (172.18.0.1:17687)

Application Services
  OK    graphknows     (http://localhost:8025/health)
  OK    agent          (http://localhost:8002/health)
  OK    search         (http://localhost:8040/api/v1/health)
  OK    frontend       (http://localhost:8080/)

All services healthy.
```

### Start infrastructure only (if app services are already running)

```bash
make infra-up
```

### Service ports reference

| Service | Port | Layer |
|---|---|---|
| Neo4j HTTP | 7474 | Infrastructure (Docker) |
| Neo4j Bolt | 7687 | Infrastructure (Docker) |
| pgvector | 5432 | Infrastructure (Docker) |
| Memgraph Bolt | 17687 | Infrastructure (Docker) |
| graphknows (HybridRAG) | 8025 | Application (bare-metal) |
| agent (LangGraph) | 8002 | Application (bare-metal) |
| search | 8040 | Application (bare-metal) |
| frontend | 8080 | Application (bare-metal) |

### Log files (bare-metal services)

```
/tmp/graphknows.log
/tmp/agent.log
/tmp/search.log
/tmp/frontend.log
/tmp/cf.log          # Cloudflare tunnel
```

### Kill zombie processes manually

```bash
fuser -k 8025/tcp 8002/tcp 8040/tcp 2>/dev/null || true
pkill -f graphgen_worker 2>/dev/null || true
```

---

## System Architecture

The system follows a strict producer-consumer model, decoupled by Neo4j as the shared storage layer.

```mermaid
graph TD
    Raw[Documents .txt/.pdf/.docx/...] -->|Input| Gen[GraphGen Service :8020]

    subgraph "GraphGen Pipeline"
        Gen --> Parse[Parser Registry]
        Parse --> Upload[Neo4j Upload]
        Upload --> Ext[Entity Extraction GLiNER + LLM]
        Ext --> Enrich[Embeddings + Resolution]
        Enrich --> Comm[Community Detection]
    end

    Upload -->|Document + Chunks| Neo4j[(Neo4j)]
    Ext -->|Entity + Relations| Neo4j

    User -->|Query| RAG[GraphRAG Service :8010]

    subgraph "GraphRAG Agent"
        RAG -->|Vector + Graph Search| Neo4j
        RAG -->|Synthesis| LLM[LLM Response]
    end
```

### Shared Infrastructure
- **Neo4j**: Stores graph topology, properties, and 384-dim vector embeddings (Community Edition 5.11+).

---

## Service Details

### 1. GraphGen (The Builder)
Located in `services/graphgen`.

**Key Pipeline Stages:**
1. **Document Parsing**: Auto-discovers files in `input/` using a parser registry (TXT, MD, PDF, DOCX, PPTX, XLSX, HTML, Images via pytesseract).
2. **Neo4j Upload**: Stores `Document → [:CONTAINS] → Chunk` nodes immediately.
3. **Entity & Relation Extraction**: Uses **GLiNER** (NER) + **LLMs** (relation extraction).
4. **Semantic Enrichment**: Generates embeddings (`BAAI/bge-small-en-v1.5`), resolves duplicate entities.
5. **Community Detection**: Applies the **Leiden Algorithm** to cluster entities.
6. **Summarization**: Generates LLM-based summaries for every community.

**Key Tech**: `LangChain`, `GLiNER`, `NetworkX`, `Leidenalg`, `SentenceTransformers`, `Neo4j`.

### 2. GraphRAG (The Agent)
Located in `services/graphrag`.

**Key Tech**: `LlamaIndex ReActAgent`, `FastAPI`, `Langfuse` (Tracing), `Pydantic`, `Neo4j`.

---

## Current Local Agent + Frontend Workflow

> **Quickstart:** Use `make dev-local` — see the [Developer Quickstart](#developer-quickstart) section above.

The active local chat stack is:
- `services/agent` on port `8002`
- `apps/web` on port `8080`

The web app proxies both HTTP and WebSocket traffic to the agent via `AGENT_URL`, which defaults to `http://localhost:8002` in [apps/web/server.py](apps/web/server.py).

If you need to start individual services manually (e.g. for debugging a single service):

### Start the Agent Backend manually

```bash
cd /root/graphknows
set -a && source .env && set +a
PYTHONPATH="$PWD/services/agent:$PWD/services:$PWD/services/graphgen:$PWD/services/scrapers" \
python3 -B -m uvicorn services.agent.main:app --host 0.0.0.0 --port 8002 --log-level info
```

### 2. Start the Frontend

In a second terminal, from the repo root:

```bash
cd /root/graphknows
AGENT_URL=http://localhost:8002 \
uvicorn apps.web.server:app --host 0.0.0.0 --port 8080
```

Open:

```text
http://localhost:8080
```

Optional helper script:

```bash
cd /root/graphknows
./apps/web/start.sh
```

That script also opens a tunnel; the plain `uvicorn` command above is the simplest local-only frontend start path.

### 3. Try the Backend Without the Frontend

Health check:

```bash
curl -s http://127.0.0.1:8002/health
```

Blocking HTTP request against `/chat`:

```bash
curl -s \
  -H 'content-type: application/json' \
  -d '{"message":"health validation"}' \
  http://127.0.0.1:8002/chat
```

Direct websocket test against the agent, without the frontend proxy:

```bash
cd /root/graphknows
python - <<'PY'
import asyncio
import json

import websockets


async def main():
    uri = 'ws://127.0.0.1:8002/ws/chat/readme-smoke-test'
    async with websockets.connect(uri, open_timeout=10, close_timeout=10) as ws:
        await ws.send(json.dumps({
            'message': 'Build a short investigation plan for the EU AI Act',
            'session_id': 'readme-smoke-test',
        }))

        for _ in range(20):
            msg = await asyncio.wait_for(ws.recv(), timeout=120)
            data = json.loads(msg)
            print(json.dumps({
                'type': data.get('type'),
                'event_type': (data.get('payload') or {}).get('event_type'),
                'error': data.get('error'),
                'briefing_len': len(data.get('briefing') or ''),
            }, ensure_ascii=True))
            if data.get('type') in {'done', 'answer', 'error'}:
                break


asyncio.run(main())
PY
```

Expected behavior:
- First message: `{"type":"status","status":"thinking"}`
- Then a sequence of `ws_event` messages such as `plan_ready`, `search_start`, `docs_fetched`, `graph_update`, and `brief_done`
- Final message: `done`

For larger prompts, the investigative workflow can take a few minutes to finish.

---

## Data Schema

```
(:Document {doc_id, title, source_path, created_at})
    -[:CONTAINS]->
(:Chunk {chunk_id, text, position, doc_id, embedding[384]})
    -[:MENTIONS]->
(:Entity {entity_id, name, type, embedding[384]})
    -[:RELATED_TO {relation}]->
(:Entity)
```

---

## Getting Started

### 1. Prerequisites
- **Docker** and **Docker Compose**.
- API Keys for **Groq** (or OpenAI).

### 2. Configuration
```bash
cp .env.example .env
# Fill in: NEO4J_PASSWORD, GROQ_API_KEY
```

### 3. Run the Stack
```bash
docker-compose up --build -d
```
This starts:
- **GraphGen** (Port 8020)
- **GraphRAG** (Port 8010)
- **Neo4j** (Port 7474 / 7687)

### 4. Ingest Data
```bash
# Place files in input/
curl -X POST http://localhost:8020/run \
  -H "Content-Type: application/json" \
  -d '{"clean_database": true}'
```

Or upload individual documents:
```bash
curl -X POST http://localhost:8020/documents -F "file=@myfile.pdf"
```

### 5. Chat
- **Web UI**: [http://localhost:3333](http://localhost:3333)
- **API Docs**: [http://localhost:8010/docs](http://localhost:8010/docs)

## Development

```bash
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up
```

---

## ️ System Architecture

The system follows a strict producer-consumer model, decoupled by the storage layer.

```mermaid
graph TD
    Raw[Raw Data .txt/.csv] -->|Input| Gen[GraphGen Service]
    
    subgraph "GraphGen Pipeline"
        Gen --> Lex[Lexical Graph]
        Lex --> Ext[Entity Extraction]
        Ext --> Enrich[Semantic Enrichment]
        Enrich --> Comm[Community Detection]
    end
    
    Comm -->|Write Topology| Falkor[(FalkorDB Graph)]
    Comm -->|Write Vectors| PG[(Postgres pgvector)]
    
    User -->|Query| RAG[GraphRAG Service]
    
    subgraph "GraphRAG Agent"
        RAG -->|Vector Search| PG
        RAG -->|Graph Traversal| Falkor
        RAG -->|Synthesis| LLM[LLM Response]
    end
```

### Shared Infrastructure
- **FalkorDB**: Stores the graph topology (Nodes, Edges, Properties).
- **PostgreSQL (pgvector)**: Stores 384-dimensional vector embeddings for hybrid retrieval.

---

## ️ Service Details

### 1. GraphGen (The Builder)
Located in `services/graphgen`.

This service runs a multi-stage ETL pipeline to convert unstructured text into a structured knowledge graph.

**Key Pipeline Stages:**
1. **Lexical Construction**: Parses documents into a temporal hierarchy (`DAY` → `SEGMENT` → `EPISODE` → `CHUNK`).
2. **Entity & Relation Extraction**: Uses a hybrid of **GLiNER** (for high-precision NER) and **LLMs** (for semantic relation extraction).
3. **Semantic Enrichment**: Generates embeddings and resolves duplicate entities using vector similarity.
4. **Community Detection**: Applies the **Leiden Algorithm** to cluster entities into high-level `TOPIC` and `SUBTOPIC` nodes.
5. **Summarization**: Generates LLM-based summaries for every community and topic.
6. **Centrality**: Calculates Degree Centrality and Z-Scores for influential entities.
7. **Hybrid Upload**: Persists topology to FalkorDB and vectors to Postgres.

**Key Tech**: `LangChain`, `Spacy`, `GLiNER`, `NetworkX`, `Leidenalg`, `SentenceTransformers`.

### 2. GraphRAG (The Agent)
Located in `services/graphrag`.

This service provides a `FastAPI` interface for querying the graph. Unlike simple RAG, it uses an **Agentic Workflow** to "walk" the graph.

**Retrieval Workflow:**
1. **Keyword Extraction**: Analyzes the user query to find key entities.
2. **Seed Identification**: Uses **Hybrid Search** (Vector Similarity + Exact Match) to find entry points in the graph.
3. **Subgraph Expansion**: Expands from seeds to find relevant neighbors (Chunks, Topics, Related Entities) using Cypher queries.
4. **Context Building**: Formats the subgraph into a structured XML context for the LLM.
5. **Synthesis**: Generates a personalized answer based on the traversed path.

**Key Tech**: `LlamaIndex`, `FastAPI`, `Langfuse` (Tracing), `Pydantic`.

---

##  Data Schema

The graph uses a specific schema to represent both time and knowledge:

- **Temporal Nodes**: `DAY`, `SEGMENT` (Morning/Afternoon), `EPISODE` (Events).
- **Content Nodes**: `CHUNK` (Raw text).
- **Semantic Nodes**: `ENTITY_CONCEPT` (People, Places, Concepts).
- **Organizational Nodes**: `TOPIC`, `SUBTOPIC` (Leiden communities).

---

##  Getting Started

### 1. Prerequisites
- **Docker** and **Docker Compose**.
- API Keys for **Groq** and **OpenAI**.

### 2. Configuration
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Fill in your credentials in `.env`:
   ```env
   GROQ_API_KEY=gsk_...
   OPENAI_API_KEY=sk-...
   ```

### 3. Run the Stack
Build and start all services:
```bash
docker-compose up --build -d
```
This starts:
- **GraphGen** (Port 8020)
- **GraphRAG** (Port 8010)
- **FalkorDB** & **Postgres**

### 4. Ingest Data
1. Place text files (`.txt`, `.csv`) in the `input/` directory.
2. Trigger the pipeline via the API:
   ```bash
   curl -X POST http://localhost:8020/run \
     -H "Content-Type: application/json" \
     -d '{"clean_database": true}'
   ```
   *(Use `clean_database: false` for incremental updates)*

### 5. Chat
Access the retrieval interface:
- **Web UI**: [http://localhost:8010](http://localhost:8010)
- **API Docs**: [http://localhost:8010/docs](http://localhost:8010/docs)

## Development

To run with hot-reloading for local development:
```bash
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up
```