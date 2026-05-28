# FLUSH-004: Neo4jStore.run_gds_enrichment — PageRank + betweenness

## Ticket: t_7f3c8bd6
## File: services/graphknows/graphknows/retrieval/stores/neo4j.py
## Tests: tests/test_flush.py (extend)

## What to build

Add `run_gds_enrichment(session_id: str) -> None` to `Neo4jStore` (async).
GDS 2026.04 is confirmed available.

### Algorithm:

**Step 1: Project entity CO_OCCURS graph**
```cypher
CALL gds.graph.project(
  'entity-cooccurs-' + $session_id,
  'ENTITY',
  'CO_OCCURS'
)
```
Use a unique graph name per session to avoid collisions.

**Step 2: PageRank**
```cypher
CALL gds.pageRank.write(
  'entity-cooccurs-' + $session_id,
  {writeProperty: 'pagerank', dampingFactor: 0.85, maxIterations: 20}
)
```

**Step 3: Betweenness (approximate)**
```cypher
CALL gds.betweenness.write(
  'entity-cooccurs-' + $session_id,
  {writeProperty: 'betweenness', samplingSize: 100}
)
```

**Step 4: Drop projection**
```cypher
CALL gds.graph.drop('entity-cooccurs-' + $session_id)
```
Always drop — prevent graph catalog pollution.

Wrap each GDS call in try/except — if GDS fails for any reason, log warning and continue.
This method is called only when `FlushOptions.run_gds=True`.

## Tests to write
- `test_run_gds_enrichment_success` — mock neo4j driver, assert all 4 Cypher calls made in order
- `test_run_gds_enrichment_graceful_fail` — driver raises on pageRank call, assert no exception propagates

## CI gate (same as brief-001)
