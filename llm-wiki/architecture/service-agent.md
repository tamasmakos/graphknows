---
title: Service Design — services/agent/
created: 2026-05-04
updated: 2026-05-15
type: concept
tags: [architecture, agent, memgraph, search, service-design]
confidence: high
---

# Service Design — `services/agent/`

The agent service is the single application-facing backend for the live frontend. It owns session lifecycle, the investigative workflow, session graph writes, retrieval, and final briefing generation.

## Responsibility

The current live agent does all of the following:

- receives chat requests from the frontend
- creates and tracks investigation sessions
- runs the 3-phase workflow: `scope -> investigate -> communicate`
- writes chunks to pgvector and entities/relations to the session graph backend
- streams progress over websocket and SSE
- returns final answer + graph snapshots on blocking HTTP `/chat`

The browser path no longer depends on GraphRAG to answer `/chat` requests.

## Current Directory Layout

```text
services/agent/
  main.py
  router.py
  config.py
  orchestrator.py              # legacy iterative path still present
  graph/
    workflow.py
    state.py
    events.py
    nodes/
      scope.py
      investigate.py
      communicate.py
  retrieval/
    hybrid_retriever.py
  session/
    manager.py
  tools/
    search_client.py
    tavily_tool.py
    cellar_tool.py
    extractor.py
    search.py
```

## Public API Surface

### HTTP

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | liveness probe |
| `/chat` | POST | blocking investigative answer path |
| `/sessions` | POST | create session |
| `/sessions/{session_id}` | GET | get session state |
| `/sessions/{session_id}` | DELETE | close session |
| `/session/{session_id}/merge` | POST | merge session graph to Neo4j |
| `/sessions/{session_id}/query` | POST | SSE with final briefing |
| `/sessions/{session_id}/investigate` | POST | typed SSE investigative stream |

### WebSocket

| Route | Role |
|---|---|
| `/sessions/{session_id}/stream` | primary investigative stream |
| `/ws/chat/{session_id}` | compatibility alias used by the current frontend |

## `/chat` Contract

Request:

```json
{
  "message": "Build an investigation plan for the EU AI Act",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Response shape:

```json
{
  "answer": "# Investigation Plan for the EU AI Act ...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "graph": {
    "nodes": [],
    "edges": []
  },
  "reasoning_steps": [],
  "sources": []
}
```

Important note: the current `/chat` path returns `sources: []` even when the answer text cites `[Doc-N]`. That is a frontend/data-contract gap, not a failure of the investigative workflow itself.

## WebSocket Contract

Observed live websocket flow:

```json
{"type":"status","status":"thinking"}
{"type":"ws_event","payload":{"event_type":"plan_ready","payload":{}}}
{"type":"ws_event","payload":{"event_type":"search_start","payload":{}}}
{"type":"ws_event","payload":{"event_type":"docs_fetched","payload":{}}}
{"type":"ws_event","payload":{"event_type":"chunks_written","payload":{}}}
{"type":"ws_event","payload":{"event_type":"entities_written","payload":{}}}
{"type":"ws_event","payload":{"event_type":"graph_update","payload":{}}}
{"type":"ws_event","payload":{"event_type":"brief_done","payload":{}}}
{"type":"done","briefing":"..."}
```

Key detail: workflow events are wrapped inside `type = ws_event`; `graph_update` is an inner `event_type`, not the top-level message type.

## Workflow Phases

### 1. Scope

- decomposes the topic into sub-questions
- emits `plan_ready`

### 2. Investigate

- fetches documents from search adapters and Tavily
- writes chunks to pgvector
- writes entities/relations to the session graph
- emits `search_start`, `docs_fetched`, `chunks_written`, `entities_written`, `graph_update`, `novelty_check`, and eventually `search_done`

### 3. Communicate

- runs retrieval over the accumulated state
- writes the final briefing
- emits `retrieval_done`, `brief_chunk`, and `brief_done`

## Dependencies

| Dependency | Used for |
|---|---|
| `services/search` | structured EU-source retrieval |
| Tavily | live web retrieval |
| CELLAR | legal/publications lookup |
| Neo4j | persistent KG |
| Memgraph / pluggable graph backend | session-scoped graph state |
| pgvector | chunk embeddings and retrieval |

## Session Lifecycle

1. create or resolve `session_id`
2. create session graph if new
3. run workflow
4. stream progress over websocket or SSE
5. return final briefing and graph snapshot
6. optionally merge the session graph into Neo4j
7. delete the session when finished

## Current Local Run Command

```bash
cd /root/graphknows
set -a && source .env && set +a
PYTHONPATH="$PWD/services/agent:$PWD/services:$PWD/services/graphgen:$PWD/services/scrapers" \
python3 -B -m uvicorn services.agent.main:app --host 0.0.0.0 --port 8002 --log-level info
```

## Links

[[architecture/reference]] [[architecture/agent-design]] [[product/frontend-design]] [[ops/administration]]
