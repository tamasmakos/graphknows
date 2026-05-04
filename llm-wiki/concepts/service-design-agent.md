---
title: Service Design — services/agent/
created: 2026-05-04
updated: 2026-05-04
type: concept
tags: [architecture, agent, memgraph, graphrag, graphgen, service-design]
sources: [product-architecture-v2]
confidence: high
---

# Service Design — `services/agent/`

> The agent service is the **only** entry point from the frontend. All other services are internal. This document specifies its structure, API, session lifecycle, and tool set.

---

## Responsibility

The agent receives a user query in the context of an ongoing **investigation session**, orchestrates calls to graphrag, graphgen, and search, maintains a per-session **Memgraph overlay graph**, and streams reasoning steps back to the frontend via WebSocket.

---

## Directory Layout

```
services/agent/
  main.py              # FastAPI app entry point
  router.py            # HTTP + WebSocket route definitions
  orchestrator.py      # LLM-driven reasoning loop (IterativeOrchestrator pattern)
  tools/
    retrieve.py        # calls graphrag /retrieve → dense+graph hybrid search
    build.py           # calls graphgen /build → triggers KG construction for new docs
    search.py          # calls search service → find + scrape public record documents
    get_path.py        # calls graphrag /path → LKIF multi-hop traversal
    get_community.py   # returns community summary for a topic cluster
    get_chunks.py      # returns source chunks grounding an entity/claim
  session/
    manager.py         # SessionGraphManager — Memgraph CRUD + Neo4j seed
    writeback.py       # write confirmed session discoveries back to Neo4j
    models.py          # Pydantic models: Session, SessionGraph, ReasoningStep
  config.py            # env vars, model config, thresholds
  Dockerfile
  pyproject.toml
```

---

## HTTP API

```
POST   /sessions                     — create new investigation session
GET    /sessions/{session_id}        — get session state + graph snapshot
DELETE /sessions/{session_id}        — close session (triggers write-back prompt)
POST   /sessions/{session_id}/query  — run agent query, returns stream
GET    /sessions/{session_id}/graph  — get current Memgraph session graph as JSON

WS     /sessions/{session_id}/stream — WebSocket: reasoning steps + graph deltas
```

### POST /sessions/{session_id}/query

Request:
```json
{
  "query": "How has Macron's position on EU defense evolved since 2022?",
  "stream": true
}
```

Response (streamed JSON events):
```json
{"type": "step_start",  "step_id": "s1", "tool": "retrieve", "query": "Macron EU defense"}
{"type": "step_result", "step_id": "s1", "nodes_added": 3, "edges_added": 5, "summary": "..."}
{"type": "step_start",  "step_id": "s2", "tool": "get_path", "query": "Macron → EU defense policy"}
{"type": "step_result", "step_id": "s2", "path": [...], "summary": "..."}
{"type": "answer",      "text": "...", "confidence": 0.82, "sources": [...]}
{"type": "graph_delta", "nodes": [...], "edges": [...]}
```

---

## Tool Specification

The agent has **6 tools**:

### 1. `retrieve(query: str, top_k: int = 10) → RetrievalResult`
Calls graphrag `/retrieve`. Returns: entity nodes, community summaries, source chunks (dense+graph RRF hybrid). Adds retrieved nodes/edges to session Memgraph graph.

### 2. `get_path(entity_a: str, entity_b: str, max_hops: int = 3) → PathResult`
Calls graphrag `/path`. BFS over LKIF_REL typed edges in Neo4j. Returns: path nodes, edge types, source chunks per edge. Requires t_b2627d7f (typed LKIF edges in Neo4j).

### 3. `get_community(entity_id: str) → CommunityResult`
Returns the Leiden community containing the entity: community label, top entities, LLM summary, all source segments.

### 4. `get_chunks(entity_id: str, top_k: int = 5) → list[Chunk]`
Returns raw source text chunks grounding an entity. Calls NeoStore.get_chunks_for_entity(). Requires t_40558195.

### 5. `search(query: str, sources: list[str] = None) → list[Document]`
Calls services/search/. Finds + scrapes public records matching the query. Returns: document metadata + full text. Can trigger graphgen /build if documents are new.

### 6. `build(document_ids: list[str]) → BuildResult`
Calls graphgen `/build`. Triggers KG construction for specified documents. Async — returns job_id. Frontend polls or waits for completion event.

---

## Session Graph Manager

The `SessionGraphManager` wraps both Neo4j (persistent, port 7687) and Memgraph (ephemeral, port 7688). Both use the same `neo4j` Python driver.

```python
class SessionGraphManager:
    def seed_from_neo4j(self, session_id: str, entity_ids: list[str]) -> None:
        """Load subgraph from Neo4j into Memgraph session namespace."""

    def merge_entity(self, session_id: str, entity: Entity) -> None:
        """Add or update entity in session graph."""

    def merge_edge(self, session_id: str, edge: Edge) -> None:
        """Add or update edge in session graph."""

    def get_graph_snapshot(self, session_id: str) -> GraphSnapshot:
        """Export current session graph as JSON for frontend."""

    def run_mage(self, session_id: str, algorithm: str, params: dict) -> dict:
        """Run a MAGE graph algorithm on the session graph."""

    def teardown(self, session_id: str) -> None:
        """Clear session data from Memgraph."""
```

### Memgraph Cypher Schema

```cypher
// Entity node (mirrors Neo4j schema + session metadata)
CREATE (:Entity {
  id: "e_...",
  name: "Emmanuel Macron",
  label: "PERSON",
  session_id: "sess_...",
  confidence: 0.91,
  source: "neo4j"  // "neo4j" | "session" | "inferred"
})

// Edge
CREATE (:Entity {id: "e1"})-[:RELATED_TO {
  relation_type: "SUPPORTS",
  weight: 0.85,
  session_id: "sess_...",
  source_chunks: ["chunk_..."],
  source: "neo4j"
}]->(:Entity {id: "e2"})

// Reasoning step node
CREATE (:ReasoningStep {
  id: "step_...",
  session_id: "sess_...",
  tool: "retrieve",
  query: "Macron EU defense",
  summary: "Found 3 relevant entities",
  timestamp: datetime()
})
CREATE (:ReasoningStep {id: "step_1"})-[:RETRIEVED]->(:Entity {id: "e1"})
```

### Port Mapping

| Service | Bolt Port | Notes |
|---|---|---|
| Neo4j | 7687 | persistent corpus KG |
| Memgraph | 7688 (host) / 7687 (container) | ephemeral session overlay |

Both accessed with `neo4j` Python driver. Memgraph uses `IN_MEMORY_ANALYTICAL` mode for bulk seed, switches to `IN_MEMORY_TRANSACTIONAL` for active session.

---

## Session Lifecycle

```
1. POST /sessions           → create session, assign session_id
2. seed_from_neo4j()        → optionally pre-seed graph with known entities from previous sessions
3. user sends queries       → agent runs tool loop, Memgraph accumulates session graph
4. WebSocket streams        → each step pushes graph delta to frontend (live D3.js updates)
5. DELETE /sessions/{id}    → close session
   → if confirmed entities > threshold: prompt user to write back to Neo4j
   → writeback.py exports session discoveries to persistent Neo4j
   → teardown() clears Memgraph
```

---

## Write-Back to Neo4j

On session close + user confirmation, write back entities and edges where:
- `confidence >= 0.7` (configurable)
- entity does not already exist in Neo4j (deduplication via name + label)
- edge does not already exist (deduplication via source/target + relation_type)

This closes the feedback loop: user investigation sessions improve the shared KG over time.

---

## Dependencies

| Service | Called By | Protocol | Notes |
|---|---|---|---|
| graphrag | tools/retrieve.py, tools/get_path.py, tools/get_chunks.py | HTTP REST | internal Docker network |
| graphgen | tools/build.py | HTTP REST | internal Docker network |
| search | tools/search.py | HTTP REST | internal Docker network |
| Neo4j | session/manager.py | Bolt 7687 | persistent KG |
| Memgraph | session/manager.py | Bolt 7688 | session overlay |

---

## Related Tasks

- t_6f42543e — extract agent service
- t_71bc0adb — session graph manager
- t_88cdb519 — WebSocket stream
- t_9afb9e25 — write-back
- t_b9101d18 — get_path tool

## Links

[[product-architecture-v2]] [[graphrag-and-downstream]] [[service-design-graphrag]] [[service-design-search]]
