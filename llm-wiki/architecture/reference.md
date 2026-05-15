# Architecture Reference — Graphknows v2

> Purpose: authoritative reference for the current live stack, service boundaries, ports, routes, and local run commands.
> Last updated: 2026-05-15

---

## 1. Live Service Topology

```text
Browser
  -> frontend proxy :8080
     -> agent :8002
        -> search :8000
        -> Neo4j :7687
        -> pgvector :5432
        -> Memgraph :17687
```

### Frontend (`apps/web/`)

- serves the static SPA at `/`
- proxies `/api/*` to the agent
- proxies `/ws/chat/{session_id}` to the agent websocket

### Agent (`services/agent/`)

- owns the live investigative workflow
- serves blocking HTTP `/chat`
- serves websocket `/sessions/{session_id}/stream`
- exposes `/ws/chat/{session_id}` compatibility alias
- owns session creation, deletion, and merge

### Search (`services/search/`)

- EU-source retrieval service used during the investigate phase
- not a browser-facing service in the normal local workflow

---

## 2. Current Query Path

### Browser websocket path

1. browser connects to `ws://localhost:8080/ws/chat/{session_id}`
2. frontend proxy forwards to agent `ws://localhost:8002/ws/chat/{session_id}`
3. agent emits:
   - `status`
   - wrapped `ws_event` messages (`plan_ready`, `search_start`, `docs_fetched`, `chunks_written`, `entities_written`, `graph_update`, `novelty_check`, `search_done`, `retrieval_done`, `brief_done`)
   - `done`
4. frontend renders the final answer card and graph state

### Blocking HTTP path

1. browser posts to `/api/chat`
2. frontend proxy forwards to `POST /chat`
3. agent runs the same macro workflow without streaming and returns final JSON

---

## 3. Public Routes

### Frontend proxy (`:8080`)

| Route | Method | Behavior |
|---|---|---|
| `/` | GET | serve `apps/web/index.html` |
| `/static/*` | GET | serve static assets |
| `/api/*` | ANY | proxy to agent |
| `/ws/chat/{session_id}` | WS | websocket proxy to agent |

### Agent (`:8002`)

| Route | Method | Behavior |
|---|---|---|
| `/health` | GET | liveness probe |
| `/chat` | POST | blocking investigative answer |
| `/sessions` | POST | create session |
| `/sessions/{session_id}` | GET | session state |
| `/sessions/{session_id}` | DELETE | close session |
| `/session/{session_id}/merge` | POST | merge session graph |
| `/sessions/{session_id}/query` | POST | SSE final briefing |
| `/sessions/{session_id}/investigate` | POST | typed SSE stream |
| `/sessions/{session_id}/stream` | WS | primary stream |
| `/ws/chat/{session_id}` | WS | compatibility alias |

---

## 4. Ports and Defaults

| Component | Port | Default |
|---|---|---|
| frontend | `8080` | `PORT=8080` |
| agent | `8002` | bound explicitly in local run |
| search | `8000` | service-local |
| Neo4j | `7687` | persistent KG |
| pgvector | `5432` | vector DB |
| Memgraph | `17687` | session graph |

Frontend defaults:

| Variable | Default |
|---|---|
| `AGENT_URL` | `http://localhost:8002` |
| `PORT` | `8080` |

Agent defaults commonly used in local dev:

| Variable | Typical value |
|---|---|
| `SEARCH_URL` | `http://localhost:8000` |
| `NEO4J_URI` | `bolt://172.18.0.1:7687` |
| `PGVECTOR_DSN` | `postgresql://postgres:pgvectorpass123@172.18.0.1:5432/vectordb` |
| `MEMGRAPH_URI` | `bolt://172.18.0.1:17687` |

---

## 5. Exact Local Startup

### Start the agent

```bash
cd /root/graphknows
set -a && source .env && set +a
PYTHONPATH="$PWD/services/agent:$PWD/services:$PWD/services/graphgen:$PWD/services/scrapers" \
python3 -B -m uvicorn services.agent.main:app --host 0.0.0.0 --port 8002 --log-level info
```

### Start the frontend

```bash
cd /root/graphknows
AGENT_URL=http://localhost:8002 \
uvicorn apps.web.server:app --host 0.0.0.0 --port 8080
```

### Optional: start the search service

```bash
cd /root/graphknows
PYTHONPATH="$PWD/services" uvicorn services.search.main:app --host 0.0.0.0 --port 8000
```

---

## 6. Backend-Only Validation

### Health

```bash
curl -s http://127.0.0.1:8002/health
```

### Blocking `/chat`

```bash
curl -s \
  -H 'content-type: application/json' \
  -d '{"message":"health validation"}' \
  http://127.0.0.1:8002/chat
```

### Websocket smoke test

```bash
cd /root/graphknows
python - <<'PY'
import asyncio
import json
import websockets

async def main():
    uri = 'ws://127.0.0.1:8002/ws/chat/reference-smoke'
    async with websockets.connect(uri, open_timeout=10, close_timeout=10) as ws:
        await ws.send(json.dumps({
            'message': 'Build a short investigation plan for the EU AI Act',
            'session_id': 'reference-smoke',
        }))
        for _ in range(20):
            msg = await asyncio.wait_for(ws.recv(), timeout=120)
            data = json.loads(msg)
            print(data.get('type'), (data.get('payload') or {}).get('event_type'))
            if data.get('type') in {'done', 'answer', 'error'}:
                break

asyncio.run(main())
PY
```

Expected high-level sequence:

- `status`
- `ws_event plan_ready`
- repeated search and graph events
- `done`

Long prompts can take several minutes.

---

## 7. Current Known Gap

The current live frontend can complete an investigative run and render the graph correctly while the Sources side panel still shows `0`. The workflow data contract for sources and retrieval hits is not yet wired through to that side panel.

## Links

[[architecture/service-agent]] [[product/frontend-design]] [[ops/administration]] [[ops/technical-debt]]
