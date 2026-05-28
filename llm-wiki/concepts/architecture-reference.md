# Architecture Reference — Graphknows v2

> **Purpose:** Authoritative technical reference for the Graphknows political intelligence platform. Covers service topology, data flows, API contracts, environment variables, and local development setup.
> Last updated: 2026-05-04

---

## 1. Service Map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          USER BROWSER / CLIENT                               │
│                         (D3.js graph + chat UI)                              │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │ HTTP :8080
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   FRONTEND  apps/web/server.py  :8080                       │
│   GET /          → serves index.html (SPA)                                  │
│   /api/*         → reverse-proxied to Agent :8002                           │
│   /static/*      → StaticFiles (CSS, JS, D3 bundle)                        │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │ HTTP :8002
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   AGENT  services/agent/main.py  :8002                      │
│   POST /chat                                                                │
│   GET  /session/{id}/graph                                                  │
│   DELETE /session/{id}                                                      │
│   GET  /health                                                              │
│                                                                             │
│   ┌─────────────────────────────────┐                                       │
│   │   SessionGraphManager           │──── bolt://127.0.0.1:17687 ──────────►│
│   │   (services/agent/session_graph)│                  MEMGRAPH :17687      │
│   └─────────────────────────────────┘                 (ephemeral sessions)  │
│                    │                                                        │
│                    │ HTTP :8001                                             │
│                    ▼                                                        │
│   ┌─────────────────────────────────┐                                       │
│   │  GRAPHRAG  services/graphrag/   │──── bolt://172.18.0.1:7687 ──────────►│
│   │  api.py  :8001                  │                   NEO4J  :7687        │
│   │  POST /retrieve                 │             (persistent KG)           │
│   │  GET  /health                   │                                       │
│   │                                 │──── postgresql://172.18.0.1:5432 ────►│
│   └─────────────────────────────────┘                PGVECTOR :5432         │
│                                                      (vectordb, embeddings)  │
└─────────────────────────────────────────────────────────────────────────────┘

Infrastructure (Docker / host-network)
──────────────────────────────────────
  Neo4j       bolt://172.18.0.1:7687     Persistent political KG
  pgvector    postgresql://172.18.0.1:5432/vectordb    Chunk embeddings
  Memgraph    bolt://127.0.0.1:17687     Session-scoped ephemeral graphs

Search Service (ingest path, not in live query cycle)
──────────────────────────────────────────────────────
  services/search/  :8003   (EP debates, UK legislation, EurLex adapters)
```

---

## 2. Data Flow — Full Query Cycle

### Step-by-step

```
1. USER TYPES a question in the chat UI (browser).

2. FRONTEND (SPA, :8080)
   • JS fetch → POST /api/chat   (proxied by apps/web/server.py)
   • Forwards body unchanged to Agent :8002

3. AGENT  POST /chat  (services/agent/main.py)
   a. Resolve or create session_id (UUID4 if new).
   b. If new session → SessionGraphManager.create_session(session_id)
      Creates isolated named graph in Memgraph.
   c. POST http://localhost:8001/retrieve  {query, session_id, max_steps:8}

4. GRAPHRAG  POST /retrieve  (services/graphrag/api.py)
   a. GraphRAGAgent.run(query, max_steps) in thread pool.
   b. ReAct loop (up to 8 steps):
      i.  LLM (Groq llama-3.3-70b) → emit Thought + Action.
      ii. Tool: neo4j_search  → Cypher traversal on Neo4j :7687
          Returns entities, relationships, communities.
      iii.Tool: pgvector_search → embedding similarity on pgvector :5432
          Returns top-k semantically similar chunks.
      iv. Observation appended → next LLM call.
   c. Final Answer assembled; citations list built from retrieved chunks.
   d. Returns RetrieveResponse JSON (answer, citations, reasoning_steps).

5. AGENT  (back in /chat handler)
   a. Parses entities_mentioned + reasoning_steps from graphrag response.
   b. asyncio.to_thread → SessionGraphManager.seed_from_neo4j(session_id, names)
      • Looks up each entity name in Neo4j for canonical properties.
      • MERGEs nodes + RELATED_TO edges into Memgraph session graph.
   c. asyncio.to_thread → SessionGraphManager.get_session_graph(session_id)
      • Reads Memgraph: MATCH (n)-[r]->(m) IN SESSION → returns {nodes, edges}.
   d. Returns ChatResponse {answer, session_id, graph:{nodes,edges}}.

6. FRONTEND receives ChatResponse.
   a. Renders answer text in chat pane.
   b. D3.js force-simulation receives graph.nodes + graph.edges.
   c. Graph panel redraws: nodes are entities (colour-coded by label),
      edges are political relationships, zoom/pan enabled.

7. SUBSEQUENT TURNS
   • Browser sends same session_id → Agent skips create_session.
   • Graph accumulates entities across the conversation.
   • DELETE /session/{id} or tab close → Memgraph session graph destroyed.
```

---

## 3. API Contracts

### 3.1 GraphRAG Service (:8001)

#### `GET /health`

```
Response 200
{
  "status": "ok",
  "service": "graphknows-graphrag"
}
```

#### `POST /retrieve`

**Request**

```json
{
  "query": "Which MEPs voted against the AI Act?",
  "max_steps": 8,
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | ✓ | — | Natural-language question |
| `max_steps` | int | ✗ | 8 | Max ReAct reasoning steps |
| `session_id` | string | ✗ | null | Reserved for Memgraph session |

**Response 200**

```json
{
  "query": "Which MEPs voted against the AI Act?",
  "answer": "Based on retrieved records, MEPs from the Identity & Democracy group...",
  "citations": [
    {
      "chunk_id": "ep_2024_03_12_chunk_042",
      "entity": "Maximilian Krah",
      "text": "Mr Krah raised concerns about surveillance...",
      "source_url": "https://europarl.europa.eu/..."
    }
  ],
  "reasoning_steps": [
    {
      "step": 1,
      "thought": "I need to find MEPs who opposed the AI Act vote.",
      "action": "neo4j_search",
      "action_input": "MATCH (m:MEP)-[:VOTED_AGAINST]->(l:Legislation {name:'AI Act'}) RETURN m.name",
      "observation": [{"name": "Maximilian Krah"}, {"name": "..."}]
    }
  ],
  "n_steps": 3,
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Error responses**

| Code | Condition |
|---|---|
| 503 | Agent not yet initialised (startup in progress) |
| 500 | Unhandled exception in ReAct loop |

---

### 3.2 Agent Service (:8002)

#### `GET /health`

```
Response 200
{
  "status": "ok",
  "service": "graphknows-agent"
}
```

#### `POST /chat`

**Request**

```json
{
  "message": "Who lobbied hardest for fossil fuel exemptions in the EU ETS?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `message` | string | ✓ | — | User question; must be non-empty |
| `session_id` | string | ✗ | auto-generated UUID | Omit to start a new session |

**Response 200**

```json
{
  "answer": "The most prominent lobbying came from BASF and the European Chemical Industry Council (Cefic)...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "graph": {
    "nodes": [
      {"id": "BASF", "label": "Organisation", "group": "industry"},
      {"id": "Cefic", "label": "Organisation", "group": "lobbying"},
      {"id": "EU ETS", "label": "Legislation", "group": "policy"}
    ],
    "edges": [
      {"source": "BASF", "target": "EU ETS", "type": "LOBBIED_FOR"},
      {"source": "Cefic", "target": "EU ETS", "type": "LOBBIED_FOR"}
    ]
  }
}
```

**Error responses**

| Code | Condition |
|---|---|
| 422 | Empty message field |
| 502 | GraphRAG service unreachable or returned HTTP error |

---

#### `GET /session/{session_id}/graph`

Returns the current accumulated entity graph for a session without sending a new message.

**Response 200** — same `graph` shape as `/chat` (`{nodes, edges}`)

**Response 404** — session not found

```json
{"detail": "Session '550e8400...' not found"}
```

---

#### `DELETE /session/{session_id}`

Destroys the Memgraph session graph and removes the session from the active registry.

**Response 200**

```json
{
  "status": "deleted",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response 404** — session not found

---

### 3.3 Frontend / Proxy (:8080)

| Route | Method | Behaviour |
|---|---|---|
| `/` | GET | Serves `apps/web/index.html` (SPA shell) |
| `/static/*` | GET | Serves static assets (JS, CSS, D3 bundle) |
| `/api/*` | ANY | Transparent reverse-proxy → Agent :8002 (strips `/api` prefix) |

> The frontend adds no auth headers today. All `/api/*` calls are forwarded verbatim.

---

## 4. Environment Variables

### Agent Service (`services/agent/`)

| Variable | Default | Description |
|---|---|---|
| `GRAPHRAG_URL` | `http://localhost:8001` | Base URL of the GraphRAG service |
| `MEMGRAPH_URI` | `bolt://172.17.0.1:17687` | Bolt URI for Memgraph (session graphs) |
| `NEO4J_URI` | `bolt://172.18.0.1:7687` | Bolt URI for persistent Neo4j KG |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `testpassword123` | Neo4j password |
| `HTTPX_TIMEOUT` | `60` | Seconds before graphrag HTTP call times out |

### GraphRAG Service (`services/graphrag/`)

Read from `.env` file at repo root (via `python-dotenv`).

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://172.18.0.1:7687` | Persistent KG bolt address |
| `NEO4J_HOST` | `172.18.0.1` | Fallback if `NEO4J_URI` not set |
| `NEO4J_PORT` | `7687` | Fallback port |
| `NEO4J_USER` | `neo4j` | Neo4j auth user |
| `NEO4J_PASSWORD` | `testpassword123` | Neo4j auth password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `PGVECTOR_DSN` | `postgresql://postgres:pgvectorpass123@172.18.0.1:5432/vectordb` | Full DSN (takes priority) |
| `PGVECTOR_USER` | `postgres` | pgvector user (used if DSN not set) |
| `PGVECTOR_PASSWORD` | `pgvectorpass123` | pgvector password |
| `PGVECTOR_HOST` | `172.18.0.1` | pgvector host |
| `PGVECTOR_PORT` | `5432` | pgvector port |
| `PGVECTOR_DB` | `vectordb` | pgvector database |
| `EMBED_MODEL` | *(from graph_store.EMBED_MODEL)* | Sentence-transformers model for embeddings |
| `GROQ_API_KEY` | *(required)* | Groq API key for LLM calls |
| `LLM_GROQ_API_KEY` | *(alias)* | Alternative env name for Groq key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model identifier |

### Frontend (`apps/web/`)

| Variable | Default | Description |
|---|---|---|
| `AGENT_URL` | `http://localhost:8002` | Agent service base URL for proxy |
| `PORT` | `8080` | Port to bind the frontend server |

---

## 5. Starting All Services Locally

### Prerequisites

```bash
# From repo root
cd /root/graphknows

# Install Python deps (recommend a venv)
pip install -e ".[dev]"

# Ensure infrastructure is up (Neo4j + pgvector via Docker, Memgraph via systemd/Docker)
docker-compose up -d   # or equivalent for your infra setup
```

### Start GraphRAG Service (port 8001)

```bash
cd /root/graphknows
PYTHONPATH=/root/graphknows/services \
  uvicorn graphrag.api:app \
  --host 0.0.0.0 \
  --port 8001 \
  --reload
```

### Start Agent Service (port 8002)

```bash
cd /root/graphknows
PYTHONPATH=/root/graphknows \
  uvicorn services.agent.main:app \
  --host 0.0.0.0 \
  --port 8002 \
  --reload
```

### Start Frontend (port 8080)

```bash
cd /root/graphknows
PYTHONPATH=/root/graphknows \
  uvicorn apps.web.server:app \
  --host 0.0.0.0 \
  --port 8080 \
  --reload
```

### Start All Three (tmux or separate terminals)

```bash
# Terminal 1 — GraphRAG
cd /root/graphknows && PYTHONPATH=/root/graphknows/services uvicorn graphrag.api:app --host 0.0.0.0 --port 8001

# Terminal 2 — Agent
cd /root/graphknows && PYTHONPATH=/root/graphknows uvicorn services.agent.main:app --host 0.0.0.0 --port 8002

# Terminal 3 — Frontend
cd /root/graphknows && PYTHONPATH=/root/graphknows uvicorn apps.web.server:app --host 0.0.0.0 --port 8080
```

### Health-check all services

```bash
curl -s http://localhost:8001/health | python3 -m json.tool
curl -s http://localhost:8002/health | python3 -m json.tool
curl -s http://localhost:8080/        | head -5
```

---

## 6. Starting the Public Tunnel

`apps/web/launch.py` starts the frontend server **and** opens a public HTTPS tunnel — first trying **pyngrok** (anonymous ngrok), then falling back to **cloudflared**.

```bash
cd /root/graphknows
python3 apps/web/launch.py
```

**What it does:**

1. Spawns `uvicorn apps.web.server:app` on `:8080` in a daemon thread.
2. Waits 2 s for uvicorn to bind.
3. Tries `pyngrok.ngrok.connect(8080)` → prints `🌐 GRAPHKNOWS PUBLIC URL: https://xxxx.ngrok.io`
4. If pyngrok fails, downloads `/tmp/cloudflared` (linux-amd64) and runs:
   ```
   /tmp/cloudflared tunnel --url http://localhost:8080
   ```
   Parses `*.trycloudflare.com` URL from stderr and prints it.
5. Blocks until `Ctrl+C`, then tears down the tunnel gracefully.

**Prerequisite:** Frontend needs Agent running first on `:8002`.

```bash
# Full public demo — start all three in background first:
cd /root/graphknows
PYTHONPATH=/root/graphknows uvicorn graphrag.api:app --host 0.0.0.0 --port 8001 &
PYTHONPATH=/root/graphknows uvicorn services.agent.main:app --host 0.0.0.0 --port 8002 &
python3 apps/web/launch.py   # this one blocks in foreground
```

---

## Related Pages

- [[product-architecture-v2]] — design rationale and Memgraph session graph schema
- [[service-design-agent]] — agent service spec and tool definitions
- [[service-design-search]] — search/ingest service
- [[administration]] — repo structure, PR workflow, infra endpoints
- [[product-roadmap-v2]] — what's shipped, what's next
