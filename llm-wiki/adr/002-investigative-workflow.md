---
title: Investigative Agent Architecture Redesign — 2026-05-10
created: 2026-05-10
updated: 2026-05-10
type: concept
tags: [architecture, agent, langgraph, graphgen, search, planning]
sources: [langgraph-agent-architecture-2026-05-08]
confidence: high
---

# Investigative Agent Architecture Redesign — 2026-05-10

## Decision

Replace the current ad-hoc LangGraph/ReAct research flow with a **3 macro-phase workflow**:

1. **Scope** — frame hypothesis, unit of analysis, actors, evidence plan
2. **Investigate** — gather evidence, trigger session graph build once, validate contradictions
3. **Communicate** — synthesize findings into the final report after the graph is ready

The key architectural constraint is **strict HTTP-only boundaries between services**. The agent service must not import `services/search/*` or `services/graphgen/*` internals via `sys.path` hacks or direct package imports.

## Why this change

Current code violates the service boundary in multiple places:
- `services/agent/tools/search.py` imports `search.adapters.*` directly
- `services/agent/langgraph_agent.py` imports `search.adapters.*` and `search.models` directly inside the tool
- `services/agent/tools/graphgen_tool.py` imports `graphgen.pipeline.*` directly

This makes the supposed microservice architecture behave like a single Python process with filesystem coupling. It also makes it harder to evolve `services/search/` and `services/graphgen/` independently.

## Architectural decisions

### 1. HTTP-only service boundaries

Add two explicit clients under `services/agent/tools/`:
- `search_client.py` — talks to `services/search` over HTTP
- `graphgen_client.py` — talks to `services/graphgen` over HTTP

These replace direct adapter/pipeline imports.

### 2. Session graph build is a phase transition, not a mid-loop side effect

`graphgen` should run **once**, deliberately, at the **Investigate → Communicate** transition:
- gather documents first
- enqueue async graph build job
- enter validation / contradiction pass
- communicate only after the graph is ready or timed out with an explicit fallback path

This avoids wasting cycles on repeated graph rebuilds during gathering.

### 3. Session graph writes go to Memgraph, not Neo4j

The new graphgen job mode is `session_graph`:
- input: collected documents + `session_id`
- output: entities/relations written into the session namespace in Memgraph
- persistent Neo4j remains the background corpus graph

### 4. Workflow state becomes explicit and typed

Introduce `InvestigativeState` with phase-local fields instead of an implicit message-only loop. This makes recursion limits, retry edges, and WebSocket progress reporting legible.

## Planned implementation phases

## Phase A — service boundaries

### A1. Search client boundary
- create `services/agent/tools/search_client.py`
- call `POST /api/v1/search`
- preserve document hydration behind the client boundary
- remove direct `search.adapters.*` usage from agent code

**Important current-state nuance:** the current search API returns `SearchResult` metadata, not hydrated full documents. Until the search service exposes a fuller search payload, the boundary-safe client should own the follow-up scrape/poll flow internally rather than reintroducing adapter imports.

### A2. Graphgen client boundary
- create `services/agent/tools/graphgen_client.py`
- call `POST /api/v1/jobs`
- poll via `GET /api/v1/jobs/{task_id}`
- replace direct `graphgen.pipeline.*` imports in the agent

### A3. Graphgen session-ingest mode
- add `POST /api/v1/jobs` to `services/graphgen/api.py`
- add `GET /api/v1/jobs/{task_id}` polling endpoint
- add `services/graphgen/pipeline/session_ingest.py`
- implement `mode="session_graph"` to write into Memgraph session namespace

## Phase B — state + workflow graph

- create `services/agent/graph/state.py`
- create `services/agent/graph/workflow.py`
- top-level flow:

```text
START -> scope_subgraph -> investigate_subgraph -> communicate_subgraph -> END
```

Each macro-phase is a compiled subgraph with its own conditional edges and recursion limits.

## Phase C — Scope subgraph

Create `services/agent/graph/nodes/scope.py`:
- `frame_node`
- `map_node`
- `plan_node`

Outputs:
- `hypothesis`
- `unit_of_analysis`
- `key_actors`
- `evidence_plan`

## Phase D — Investigate subgraph

Create `services/agent/graph/nodes/investigate.py`:
- `gather_node`
- `graphgen_trigger_node`
- `validate_node`

Outputs:
- `collected_docs`
- `search_iterations`
- `graphgen_task_id`
- `graphgen_complete`
- `contradictions`

## Open point

The user note introduced **Phase E — Communicate** but did not yet specify the exact node set in this message. Keep the workflow prepared for a communicate subgraph, but do not overspecify its nodes until the report contract is finalized.

## Acceptance criteria

- no `services/agent/*` runtime path mutators or cross-service imports for search/graphgen internals remain on the main investigative path
- session graph builds are triggered by explicit job enqueue, not in-process pipeline calls
- graphgen session jobs write to Memgraph session scope, not Neo4j
- the workflow has explicit macro-phase state and a visible `current_macro_phase`
- gather/validate loops are bounded and test-covered
- WebSocket events expose the graph-build transition explicitly (`graph_building`, task id)

## First implementation slice

Start with the smallest boundary-safe step:
1. add `search_client.py`
2. refactor the active `search_eu` path in `services/agent/langgraph_agent.py` to use it
3. add focused tests for the HTTP client behavior
4. leave graphgen jobs/stategraph decomposition for follow-up slices

## Links

[[langgraph-agent-architecture-2026-05-08]] [[service-design-agent]] [[product-architecture-v2]] [[kanban-plan]]
