---
title: Project Administration
updated: 2026-05-15
type: concept
tags: [admin, deployment, local-dev]
confidence: high
---

# Project Administration

## Repository

- root: `/root/graphknows`
- frontend: `apps/web/`
- agent: `services/agent/`
- search: `services/search/`
- wiki: `llm-wiki/`

## Current Local Ports

- frontend proxy: `8080`
- agent: `8002`
- search: `8000`
- Neo4j: `7687`
- pgvector: `5432`
- Memgraph: `17687`

## Environment Setup

- `.env` lives at the repo root
- do not commit `.env`
- typical startup loads env with:

```bash
set -a && source .env && set +a
```

## Exact Local Startup

### Start the agent backend

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

### Optional helper

```bash
cd /root/graphknows
./apps/web/start.sh
```

The helper also opens a tunnel. For plain local work, the direct `uvicorn` command is simpler.

## Backend-Only Smoke Tests

### Health check

```bash
curl -s http://127.0.0.1:8002/health
```

### Blocking chat check

```bash
curl -s \
  -H 'content-type: application/json' \
  -d '{"message":"health validation"}' \
  http://127.0.0.1:8002/chat
```

### Websocket smoke test without the frontend

```bash
cd /root/graphknows
python - <<'PY'
import asyncio
import json
import websockets

async def main():
    uri = 'ws://127.0.0.1:8002/ws/chat/admin-smoke'
    async with websockets.connect(uri, open_timeout=10, close_timeout=10) as ws:
        await ws.send(json.dumps({
            'message': 'Build a short investigation plan for the EU AI Act',
            'session_id': 'admin-smoke',
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

Expected sequence:

- `status`
- `ws_event plan_ready`
- repeated search and graph events
- `done`

## Operational Notes

- If the frontend looks broken, check `:8002` first; `apps/web` is a thin proxy
- The browser currently uses the compatibility websocket route `/ws/chat/{session_id}`
- Long investigative prompts can take minutes; an apparently idle UI is not necessarily a dead run

## Links

[[architecture/reference]] [[architecture/service-agent]] [[product/frontend-design]] [[ops/technical-debt]]
