---
title: "ADR-005: Pluggable Session Graph Backend (Memgraph / Neo4j + GDS)"
created: 2026-05-13
updated: 2026-05-14
type: adr
status: accepted
tags: [session-graph, memgraph, neo4j, gds, pluggable, graph-features]
confidence: high
---

# ADR-005 — Pluggable Session Graph Backend (Memgraph / Neo4j + GDS)

## Status
accepted

## Context

The session KG (`SessionGraphManager`, `DualWriter`, `HybridRetriever`) is tightly coupled to
Memgraph via the bolt-compatible neo4j Python driver. Memgraph is deployed as a local container and
has no built-in graph data science library.

Two requirements drove this ADR:

1. **Replaceability** — operators must be able to run the session KG on Neo4j (production SaaS or
   local desktop) instead of Memgraph, without touching business logic. The wire protocol is the
   same (bolt), but Memgraph and Neo4j differ on: supported APOC calls, index DDL, TTL, GDS
   availability, and auth.

2. **Graph-feature enrichment for reasoning** — after each investigation iteration the agent must
   annotate session-graph nodes with centrality scores (PageRank, betweenness), community
   assignments (Louvain), and graph-embedding vectors (fastRP). These features are exposed to the
   `brief_node` via a `NodeFeatures` dataclass injected into `InvestigativeState`, so the final LLM
   call can explicitly refer to "high-centrality entities" and "community labels" in the EPRS brief.

Neo4j Graph Data Science (GDS) is the natural provider for requirement 2: it is already available
on the running Neo4j instance (`gds.pageRank`, `gds.betweenness`, `gds.louvain`, `gds.fastRP`,
`gds.graph.project`). Memgraph does not ship GDS.

## Decision

### 1. GraphBackend abstraction

Introduce `services/agent/graph/backend.py` with a `GraphBackend` protocol:

```python
class GraphBackend(Protocol):
    async def ensure_indexes(self) -> None: ...
    async def write_entities(self, session_id: str, entities: list[Entity]) -> None: ...
    async def write_relations(self, session_id: str, relations: list[Relation]) -> None: ...
    async def two_hop_neighbours(self, session_id: str, entity_ids: list[str]) -> list[str]: ...
    async def delete_session(self, session_id: str) -> None: ...
    async def compute_node_features(self, session_id: str) -> NodeFeatures: ...
```

Two concrete implementations:

| Class | Module |
|---|---|
| `MemgraphBackend` | `graph/backends/memgraph.py` |
| `Neo4jGDSBackend` | `graph/backends/neo4j_gds.py` |

`Neo4jGDSBackend.compute_node_features()` runs:
1. `gds.graph.project` → in-memory projected graph for the session
2. `gds.pageRank.stream` → per-node PageRank
3. `gds.betweenness.stream` → betweenness centrality
4. `gds.louvain.stream` → community ID per node
5. `gds.fastRP.stream` (dim=64) → graph embedding vector
6. Returns `NodeFeatures(by_entity_id: dict[str, EntityFeatures])`

`MemgraphBackend.compute_node_features()` returns a stub (zeros/None) so the interface contract
is always satisfied; the `brief_node` guards against None gracefully.

### 2. Config selection

`services/agent/config.py` gains:

```python
SESSION_GRAPH_BACKEND: Literal["memgraph", "neo4j"] = os.getenv("SESSION_GRAPH_BACKEND", "memgraph")
```

`main.py` constructs the correct backend at startup and injects it via dependency injection into
`SessionGraphManager`, `DualWriter`, and `HybridRetriever`.

### 3. NodeFeatures → InvestigativeState

`graph/state.py` gains:
```python
node_features: dict[str, Any] = field(default_factory=dict)
# key = entity_id, value = {pagerank, betweenness, community, fastrp_embedding}
```

After each `investigate_node` iteration, `DualWriter.write()` calls
`backend.compute_node_features()` and stores results in the state. `brief_node` receives the top-N
entities by PageRank and includes them in the LLM prompt as a structured section.

### 4. Backward compatibility

- Default `SESSION_GRAPH_BACKEND=memgraph` → zero behaviour change for existing deployments.
- `MemgraphBackend` stubs `compute_node_features` → brief prompt includes features only if backend
  is `neo4j`.
- In-memory fallback in `SessionGraphManager` is preserved and not affected.

## Consequences

**Good:**
- Neo4j GDS provides production-quality centrality, community, and embedding algorithms for free.
- Operators can switch backend with one env var.
- The `brief_node` can now say "Ursula von der Leyen is the highest-centrality entity (PageRank 0.42,
  community 3: EU Commission cluster)" — grounding the EPRS brief in graph science.
- `MemgraphBackend` keeps Memgraph deployments working with zero regression.

**Bad / risks:**
- GDS `graph.project` creates a named graph in Neo4j's in-memory catalog. We must drop it after
  feature computation or it leaks across sessions. Use `gds.graph.drop` in a finally block.
- GDS is not available in Neo4j Community Edition — only Neo4j Enterprise and AuraDB. The local
  instance confirmed GDS procs present. Document this as an ops requirement.
- Session-scoped GDS projections mean concurrent sessions each create a named projection; use
  `session_id` as the projection name to avoid collisions.

## Alternatives Rejected

- **In-process networkx**: would require extracting the full session subgraph each iteration; slower
  and removes the ability to run GDS batch algorithms natively. Also doesn't help if we want to
  leverage Neo4j's query engine for graph traversal.
- **Keeping Memgraph + adding GDS manually**: Memgraph does not have GDS. Adding manual Python
  implementations of PageRank etc. is reinventing the wheel and diverges from the Neo4j GDS API.

## Related
- [[architecture/dual-write-retrieval]]
- [[architecture/service-agent]]
- [[adr/001-langgraph-agent]]
- [[adr/003-services-refactor]]
