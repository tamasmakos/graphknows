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

## Iteration notes — 2026-05-10 quality hardening

A small hardening wave improved the current agent path while the larger redesign is pending:
- session-graph persistence now canonicalizes obvious label/name variants before write-back (`ORGANISATION -> ORGANIZATION`, `EU -> European Union`), merges duplicate nodes, remaps edges, and stores canonical session entities for the LLM
- `search_eu` now expands the brittle short EPRS query for the EDIS/EDIP case with a conservative secondary query (`... EDIS EDIP defence readiness 2024`)
- query-aware EPRS ranking now adds a narrow title bonus so exact EDIS / EDIP documents outrank generic defence digests in focused tests
- `SessionGraphManager.add_edge()` now resolves existing typed nodes by `(name, session_id)` before creating generic fallback `Entity` stubs, eliminating duplicate `Entity` nodes in the persisted session graph

### Verified outcomes from live capture
- artifact: `.opencode-logs/live-capture-20260510-173118/`
- persisted session graph no longer showed duplicate generic `Entity` stubs for `European Commission` / `European Union`
- final session graph sample included canonical triplets such as:
  - `European Union --ADOPTED--> European Defence Industrial Strategy`
  - `EDIS --STANDS_FOR--> European defence industrial strategy`
  - `Ursula von der Leyen --TASKED--> Andrius Kubilius`

### Remaining quality gap
The live `tool_result` summary for `search_eu` can still surface `White paper on the future of European defence` first for the short query `Ursula von der Leyen European Defence Industrial Strategy`, even though the final answer and the persisted graph now center on the exact EDIS briefing. This indicates a remaining discrepancy between the focused ranking tests and the live tool-result ordering that should be resolved before the larger state-graph redesign.

## External best-practice synthesis — 2026-05-10

Recent agentic-workflow and KG-grounded research reinforces the redesign direction, but also suggests several important upgrades beyond the current 3-phase sketch:

### 1. Make the workflow evidence-led, not tool-led
- The orchestrator should decompose the investigation into explicit sub-questions, source classes, and verification targets before any broad search loop begins.
- High-value claims should be verified across at least two independent sources when possible.
- Contradictions should be treated as first-class outputs, not as noise to smooth away.

### 2. Use the session graph as a bounded hypothesis graph
- Keep the Memgraph session graph separate from the canonical long-term corpus graph.
- Build the session graph deliberately during the Investigate -> Communicate transition, then allow lightweight graph updates only for explicitly approved follow-up evidence.
- Treat the graph as a working model of the investigation: useful for triangulation, actor mapping, and gap detection, but never a substitute for source documents.

### 3. Pair graph retrieval with document retrieval on every important reasoning step
- Use hybrid retrieval: semantic/document search to find candidate evidence, graph traversal to expose relationships, paths, and missing links.
- Report-writing and validation nodes should cite both the underlying documents and the graph paths they rely on.
- The agent should not answer from graph structure alone when the supporting source passages are missing or weak.

### 4. Add an explicit verification layer between Investigate and Communicate
- After evidence gathering and graph build, run a dedicated verification pass that:
  - decomposes the draft findings into atomic claims
  - labels each claim as supported / contradicted / uncertain
  - records the exact evidence bundle for each claim
  - escalates thinly supported or conflicting claims for another gather loop or for explicit human review

### 5. Add provenance as a first-class system concern
- Every extracted entity, relation, claim, and report paragraph should retain pointers to:
  - source document id / URL
  - passage or chunk id
  - retrieval time
  - extraction method
  - confidence / verification status
- Prefer a separate provenance layer or metadata envelope over polluting the core session graph with every audit field.

### 6. Deep research should be parallelized by evidence thread
- Complex investigations should split into parallel sub-investigations (policy text, actors, chronology, funding, opposition/critique, historical precedent, cross-jurisdiction comparison).
- The orchestrator should synthesize these threads, compare overlap/conflict, and only then move to final communication.
- For especially important claims, redundancy checks (independent re-investigation of the same claim) are justified.

### 7. Human-auditable outputs are part of the architecture, not just the UI
- The final report should expose:
  - key findings
  - evidence ledger
  - contradictions / unresolved questions
  - graph-derived relationship paths used in reasoning
  - methodology and limitations
- This is especially important for policy and investigative journalism where trust depends on legibility and provenance.

## Recommended workflow upgrade

A stronger production workflow for Graphknows is:

1. **Scope**
   - define hypothesis, unit of analysis, stakeholders, jurisdictions, time window
   - generate an evidence plan with explicit sub-questions and target source types
   - seed the session graph from prior corpus knowledge only as orientation

2. **Investigate / Gather**
   - run structured multi-query retrieval loops per sub-question
   - maintain an evidence ledger, not just a document pile
   - score sources by authority, independence, recency, and directness
   - stop only when either coverage thresholds are met or gaps are explicit

3. **Graph Build / Enrichment**
   - trigger async graph generation once enough evidence exists
   - canonicalize entities and relations
   - preserve provenance from relation -> passage -> document
   - compute useful derived views: actor map, event timeline, institution-position map, contradiction clusters

4. **Verify / Stress-Test**
   - convert draft conclusions into atomic claims
   - run contradiction detection and alternative-hypothesis checks
   - loop back to gather for missing evidence
   - mark every claim as supported / contested / uncertain

5. **Communicate**
   - synthesize only from verified evidence bundles
   - cite documents and, where helpful, graph paths
   - include uncertainty, competing explanations, and known blind spots
   - write back the validated session graph and report artifacts

## Concrete next-step implications for implementation

1. Keep the 3 macro-phases, but split **Investigate** internally into:
   - gather
   - graph build trigger
   - verify / contradiction pass
2. Introduce an **evidence ledger** structure in state alongside `collected_docs`.
3. Make **claim verification** a dedicated node family, not just a paragraph in the answer step.
4. Require **hybrid retrieval** in validation and reporting: graph path + supporting passages.
5. Add **source authority / independence scoring** for policy and journalism use cases.
6. Add **parallel evidence-thread execution** for deeper investigations.
7. Preserve a **separate provenance layer** for extracted claims and relations.

## Source-aligned rationale
- Multi-agent orchestrator-worker research systems outperform single linear loops on breadth-heavy research tasks and are better suited for parallel evidence-thread collection.
- Investigative and journalistic best practice requires verification discipline, contradiction surfacing, and explicit attribution.
- KG-grounded agent research suggests hybrid document + graph retrieval, session-bounded working graphs, and uncertainty-aware reasoning rather than blind trust in graph topology.

## Links

[[langgraph-agent-architecture-2026-05-08]] [[service-design-agent]] [[product-architecture-v2]] [[kanban-plan]]