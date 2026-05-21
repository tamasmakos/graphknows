# Graphknows

An agentic knowledge graph system for deep investigative research. Transforms unstructured documents into a queryable semantic memory using hybrid graph + vector retrieval, then delivers structured briefings via a LangGraph agent.

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Agent Service  :8002                               │
│  LangGraph StateGraph — search loop → dual-write    │
│  WebSocket: thought_token, tool_start, graph_update │
└───────────┬──────────────────────┬──────────────────┘
            │                      │
            ▼                      ▼
┌───────────────────┐   ┌──────────────────────────┐
│  graphknows       │   │  Search Service  :8040   │
│  HybridRAG :8025  │   │  Document search + index │
│  MCP over HTTP    │   │  /api/v1/health           │
└──────┬────────────┘   └──────────────────────────┘
       │
       ├──────────────────────────────────┐
       ▼                                  ▼
┌──────────────┐                 ┌────────────────────┐
│  Neo4j       │                 │  pgvector          │
│  :7687 Bolt  │                 │  :5432             │
│  Knowledge   │                 │  Vector embeddings │
│  graph       │                 │  (384-dim BGE)     │
└──────────────┘                 └────────────────────┘

┌──────────────────────┐
│  Memgraph  :17687    │
│  Per-session graph   │
│  overlay (ephemeral) │
└──────────────────────┘

┌──────────────────────┐
│  Frontend  :8080     │
│  WebSocket chat UI   │
└──────────────────────┘
```

### Services

| Service | Port | Role |
|---|---|---|
| agent | 8002 | LangGraph StateGraph — main orchestrator |
| graphknows | 8025 | HybridRAG over Neo4j + pgvector (MCP HTTP) |
| search | 8040 | Document search and indexing |
| frontend | 8080 | WebSocket chat UI |
| Neo4j | 7474 / 7687 | Persistent knowledge graph |
| pgvector | 5432 | Vector embedding store |
| Memgraph | 17687 | Ephemeral per-session graph overlay |

### Infrastructure

- **Neo4j**: Stores long-term knowledge graph topology and properties.
- **pgvector (Postgres)**: Stores 1536-dimensional vector embeddings (`openai/text-embedding-3-small` via OpenRouter) for hybrid retrieval.
- **Memgraph**: Ephemeral in-memory graph for per-session overlays. Uses MAGE algorithms (Leiden, PageRank).

---

## Developer Quickstart

> One command to start the full local stack and verify everything is healthy.

### Prerequisites

- Docker + Docker Compose (infrastructure: Neo4j, pgvector, Memgraph)
- Python ≥3.11 and [uv](https://github.com/astral-sh/uv)
- `.env` file — see setup below

### First time

```bash
cd graphknows
make setup        # copies .env.example → .env; tells you exactly what to fill in
# edit .env, fill in the three REQUIRED keys
make dev          # preflight check → infra → all services
```

### Every subsequent time

```bash
make dev          # one command: validates env, starts everything, waits for health
```

Without Cloudflare tunnel:

```bash
make dev-local-no-tunnel
```

### Health check

```bash
make doctor       # pre-flight GO/NO-GO: env vars, Python, infra reachability, ports
make status       # health check all 8 service endpoints
make logs         # tail all four service logs in one view
```

Expected output when everything is running:

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

### Start infrastructure only

```bash
make infra-up
```

### All Makefile targets

```bash
make help
```

Key targets:

| Target | Description |
|---|---|
| `make dev` | **Recommended:** preflight → infra check → start all services |
| `make setup` | First-time: copy `.env.example → .env` if missing, then doctor |
| `make doctor` | Pre-flight GO/NO-GO check (env vars, Python, infra, ports) |
| `make logs` | Tail all four service logs in one view |
| `make status` | Health check all 8 service endpoints |
| `make dev-local` | Start services directly (skip preflight) |
| `make dev-local-no-tunnel` | Same, without Cloudflare tunnel |
| `make infra-up` | Start Neo4j, pgvector, Memgraph containers only |
| `make infra-down` | Stop infrastructure containers |
| `make up` | Start full stack via Docker Compose (production images) |
| `make down` | Stop and remove containers |

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

## Manual Service Start (single service debugging)

Source environment first — always:

```bash
set -a && source .env && set +a
export LANG=C LC_ALL=C
PYPATH="/root/graphknows/services:/root/graphknows/services/graphknows:/root/graphknows/services/graphknows/graphknows"
```

**Agent:**

```bash
PYTHONPATH="$PYPATH" /root/graphknows/services/agent/.venv/bin/uvicorn \
  services.agent.main:app --host 0.0.0.0 --port 8002 --log-level info
```

**graphknows (HybridRAG):**

```bash
PYTHONPATH="$PYPATH" python -m graphknows.run --transport http --port 8025
```

**Search:**

```bash
PYTHONPATH="$PYPATH" uvicorn services.search.main:app \
  --host 0.0.0.0 --port 8040 --log-level info
```

**Frontend:**

```bash
AGENT_URL=http://localhost:8002 \
uvicorn apps.web.server:app --host 0.0.0.0 --port 8080
```

---

## Testing the Agent

Health check:

```bash
curl -s http://127.0.0.1:8002/health
```

Blocking HTTP chat:

```bash
curl -s \
  -H 'content-type: application/json' \
  -d '{"message": "health validation"}' \
  http://127.0.0.1:8002/chat
```

WebSocket smoke test:

```bash
cd /root/graphknows
python - <<'PY'
import asyncio, json, websockets

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
                'briefing_len': len(data.get('briefing') or ''),
            }))
            if data.get('type') in {'done', 'answer', 'error'}:
                break

asyncio.run(main())
PY
```

Expected WebSocket event sequence: `status(thinking)` → `plan_ready` → `search_start` → `docs_fetched` → `graph_update` → `brief_done` → `done`.

---

## Data Schema

```
(:Document {doc_id, title, source_path, created_at})
    -[:CONTAINS]->
(:Chunk {chunk_id, text, position, doc_id, embedding[1536]})
    -[:MENTIONS]->
(:Entity {entity_id, name, type, embedding[1536]})
    -[:RELATED_TO {relation}]->
(:Entity)
```

---

## Configuration

```bash
cp .env.example .env
```

Required keys:
Three required keys (everything else has sensible defaults):

```env
OPENROUTER_API_KEY=...   # embeddings (text-embedding-3-small) + LLM calls
NEO4J_PASSWORD=...       # must match docker-compose.yaml
PGVECTOR_PASSWORD=...    # must match docker-compose.yaml
```

All other variables are documented with `[REQUIRED]` / `[OPTIONAL]` markers in `.env.example`.

> Infrastructure is accessed via Docker bridge IPs (`172.18.0.1`), not `localhost`, because app services run bare-metal while databases run in Docker containers.

---

## Project Layout

```
/root/graphknows/
  services/
    graphknows/    # HybridRAG MCP service — Neo4j + pgvector
    agent/         # LangGraph StateGraph agent + WebSocket API
    search/        # Document search and indexing service
  apps/
    web/           # Frontend WebSocket chat UI
  tests/           # Committed pytest tests (CI-run)
  research/        # GITIGNORED — prototypes, demos, scratch scripts
  scripts/
    preflight.sh   # Pre-flight GO/NO-GO validator (env, Python, infra, ports)
    dev-local.sh   # Smart local dev launcher (called by make dev)
    health.sh      # Full-stack health check (called by make status)
  .env.example     # All env vars documented — copy to .env to start
  Makefile         # All developer commands (run: make help)
  docker-compose.yaml
  docker-compose.dev.yaml
```

### The golden rule for code placement

| What | Where |
|---|---|
| Prototype / demo / one-off | `research/` — gitignored, never committed |
| Production code | `services/<service>/` |
| Tests (committed) | `tests/` or `services/<service>/tests/` |
| Implementation plans | `.hermes/plans/` |

**Never** create `.py` files at the repo root.
