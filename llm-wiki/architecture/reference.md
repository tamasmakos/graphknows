---
title: "Service Topology & Reference — Ports, URLs, Env Vars"
created: 2026-05-04
updated: 2026-05-20
type: architecture
tags: [architecture, deployment, docker, reference, admin]
confidence: high
---

# Service Topology & Reference

Full service map, ports, environment variables, and local dev setup.

---

## Service Map

```
┌─────────────┐     WebSocket/HTTP      ┌──────────────┐
│  apps/web   │ ──────────────────────► │ services/    │
│  :8080      │                         │ agent :8030  │
└─────────────┘                         └──────┬───────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          │                    │                    │
                          ▼                    ▼                    ▼
               ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐
               │ services/search  │  │services/graphknows│  │   Memgraph    │
               │ :8040            │  │ :8025            │  │ bolt :7688    │
               └──────────────────┘  └─────────┬────────┘  └───────────────┘
                                               │
                          ┌────────────────────┼────────────┐
                          ▼                    ▼            ▼
                   ┌──────────┐        ┌───────────┐  ┌──────────┐
                   │  Neo4j   │        │ pgvector  │  │Memgraph  │
                   │ :7687    │        │ :5432     │  │ :7688    │
                   └──────────┘        └───────────┘  └──────────┘
```

**Rule:** The frontend NEVER calls search or graphknows directly. All requests go through `services/agent`.

---

## Ports

| Service | Host port | Internal port | Notes |
|---|---|---|---|
| `neo4j` (HTTP) | 7474 | 7474 | Browser UI |
| `neo4j` (Bolt) | 7687 | 7687 | Driver connections |
| `pgvector` | 5432 | 5432 | PostgreSQL with vector extension |
| `memgraph` (Bolt) | 7688 | 7687 | Host 7688 avoids collision with Neo4j |
| `memgraph` (HTTP) | 7444 | 7444 | Memgraph Lab web UI |
| `graphknows` | 8025 | 8025 | Consolidated graph/retrieval service |
| `agent` | 8030 | 8000 | Main agent API |
| `search` | 8040 | 8000 | EU source connector service |
| `frontend` | 8080 | 8080 | Web app |
| `langfuse` | 3000 | 3000 | Observability (optional profile) |

---

## Memgraph — from Hermes container

From inside the Hermes container (172.18.0.1 is the Docker bridge):

```
bolt://172.18.0.1:17687   ← confirmed working (no auth)
```

From inside Docker network:
```
bolt://memgraph:7687
```

From host machine:
```
bolt://localhost:7688
```

---

## Environment Variables

### services/agent

| Variable | Default | Purpose |
|---|---|---|
| `GRAPHKNOWS_URL` | `http://graphknows:8025` | graphknows service base URL |
| `GRAPHKNOWS_MODE` | `http` | `http` or `inprocess` |
| `SEARCH_URL` | `http://search:8000` | search service base URL |
| `MEMGRAPH_URI` | `bolt://memgraph:7687` | Memgraph bolt URI |
| `PGVECTOR_DSN` | `postgresql://postgres:***@pgvector:5432/vectordb` | pgvector DSN |
| `SESSION_GRAPH_BACKEND` | `memgraph` | `memgraph` or `neo4j_gds` |
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j bolt URI (for writeback) |
| `NEO4J_USER` | `neo4j` | Neo4j auth |
| `NEO4J_PASSWORD` | `changeme` | Neo4j auth |
| `LLM_PROVIDER` | `groq` | LLM provider |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | LLM model |
| `GROQ_API_KEY` | — | Required for Groq |
| `GRAPHKNOWS_ENV` | `development` | Environment tag |
| `EXPORT_GRAPH_HTML` | `false` | Write HTML graph snapshots to disk |
| `GRAPH_HTML_OUTPUT_DIR` | `/root/graphknows/output` | Output dir for HTML snapshots |

### services/graphknows

| Variable | Default | Purpose |
|---|---|---|
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j corpus KG |
| `PGVECTOR_DSN` | `postgresql://postgres:***@pgvector:5432/vectordb` | pgvector |
| `MEMGRAPH_URI` | `bolt://memgraph:7687` | Memgraph session graph |
| `GROQ_API_KEY` | — | For DSPy relation extraction |
| `LLM_PROVIDER` | `groq` | |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `INPUT_DIR` | `/data/input` | Graphgen pipeline input |
| `OUTPUT_DIR` | `/data/output` | Graphgen pipeline output |
| `GRAPHKNOWS_PORT` | `8025` | Service listen port |

### services/search

| Variable | Default | Purpose |
|---|---|---|
| `GRAPHGEN_URL` | `http://graphknows:8025` | graphknows service (legacy alias) |
| `RAW_DATA_DIR` | `/data/raw` | Raw document cache mount |
| `GRAPHKNOWS_ENV` | `development` | |
| `SEARCH_PORT` | `8040` | Host port |

---

## Local Dev Setup

```bash
# 1. Start infrastructure + services
cd /root/graphknows
docker compose up -d

# 2. Check health
curl http://localhost:8025/health   # graphknows
curl http://localhost:8030/health   # agent  (NOTE: maps to container :8000)
curl http://localhost:8040/health   # search (NOTE: maps to container :8000)
curl http://localhost:8080/         # frontend

# 3. Environment (for running services outside Docker)
set -a && source .env && set +a

# 4. Kill zombie graphgen workers if needed
pkill -f graphgen_worker
```

---

## API Contracts

### agent: POST /chat

```json
{
  "session_id": "string",
  "message": "string",
  "stream": true
}
```

Response: streams WS events or returns JSON brief on HTTP.

### agent: WS /ws/{session_id}

Bidirectional. Client sends `{"message": "..."}`, server pushes event objects (see [[architecture/service-agent]] for event schema).

### search: POST /search

```json
{
  "query": "string",
  "sources": ["ep_debates", "eurlex", "eprs_briefings"],
  "max_results": 10
}
```

Response: `list[ParsedDocument]`

### graphknows: POST /extract

```json
{
  "text": "string",
  "session_id": "string",
  "ontology": "politics"
}
```

Response: `{entities: [...], relations: [...]}`

---

## Notes

- `HYBRIDRAG_URL` and `GRAPHGEN_URL` env vars are legacy aliases for `GRAPHKNOWS_URL` — all three point to `http://graphknows:8025`.
- The `agent` service Docker port is `8030:8000` (host:container). Client code should use `http://localhost:8030` from host, `http://agent:8000` from Docker network.
- `search` service Docker port is `${SEARCH_PORT:-8040}:8000`. Use `http://localhost:8040` from host.
