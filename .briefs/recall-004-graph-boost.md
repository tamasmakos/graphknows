# RECALL-004: Add graph-neighbour RRF signal to STM recall

## Working directory
`/root/graphknows`

## Coding standards
- `.hermes/instructions/python-style.instructions.md`
- `.hermes/instructions/tdd-test-driven-development.instructions.md`

## Problem
RRF with K=60 across 2 lanes (stm_chunks ANN + stm_turns ANN) compresses all recall
scores into a 0.016–0.033 band. `:REL` edges between entities (added in the recent
STM relation layer) are not used in ranking at all.

## Current flow (read recall_stm.py first)
1. `_run_stm_ann()` — ANN on stm_chunks + stm_turns → RRF merged into `results`
2. `_entity_boost()` — entity keyword match → boost or insert entity-matched chunks
3. `_shape_score()` — memory_class weighting
4. Sort and return top_k

## Fix: add a third RRF lane — 1-hop :REL graph traversal

### New method on MemgraphStore
Add `query_related_entity_chunks(session_id, entity_names, top_k)`:

```python
def query_related_entity_chunks(
    self,
    session_id: str,
    entity_names: list[str],
    top_k: int,
) -> list[dict]:
    """1-hop :REL traversal from matched entities to find neighbour entity chunks.

    Returns list of dicts: chunk_id, text, entity_names (list of neighbour names).
    These are chunks that co-mention entities one hop away from the matched set.
    """
    # Cypher (labels UPPERCASE, use id(n) not elementId):
    # MATCH (e:ENTITY {session_id: $sid})-[:REL]-(n:ENTITY {session_id: $sid})
    # WHERE e.name IN $names
    # MATCH (c:CHUNK {session_id: $sid})-[:MENTIONS]->(n)
    # RETURN DISTINCT c.id AS chunk_id, c.text AS text,
    #        collect(DISTINCT n.name) AS entity_names
    # LIMIT $top_k
```

Note: `:REL` edges in Memgraph are generic with a `name` property — the relationship
type itself is `:REL`, not a dynamic type. Edge direction may be either way, use `-[:REL]-`
(undirected) not `->` or `<-` to catch both directions.

### New function in recall_stm.py
Add `_graph_neighbour_boost(mg_store, session_id, matched_entities, top_k, results)`:
```python
GRAPH_BOOST_WEIGHT = 0.5  # fraction of direct-entity-match RRF rank score
```
- Gets matched entity names from current results (entities on existing hits)
- Calls `query_related_entity_chunks(session_id, matched_entity_names, top_k)`
- For each returned chunk:
  - If `cid in results`: add `GRAPH_BOOST_WEIGHT / (RRF_K + rank + 1)` to score
  - If `cid not in results`: insert with score `GRAPH_BOOST_WEIGHT / (RRF_K + rank + 1)`

Call `_graph_neighbour_boost()` after `_entity_boost()` in `recall_short_term()`.

Only call it if there are any entity-bearing results (skip if no entities found).

## Files to touch (ONLY these)
1. `graphknows/retrieval/recall_stm.py` — add `_graph_neighbour_boost()`, call after `_entity_boost()`
2. `graphknows/retrieval/stores/memgraph.py` — add `query_related_entity_chunks()`
3. `tests/test_recall.py` — add unit tests for the new function

## Constraints
- Labels UPPERCASE: `:ENTITY`, `:CHUNK`, `:REL` (relationship type), `:MENTIONS` (relationship type)
- Use undirected `-[:REL]-` for `:REL` traversal (edges stored in either direction)
- Use `id(n)` not `elementId(n)` in Memgraph
- MemgraphStore is synchronous (not async)
- `GRAPH_BOOST_WEIGHT = 0.5` is a module constant, not hardcoded

## Acceptance criteria
- Unit test: mock `query_related_entity_chunks` returning a chunk NOT in ANN results;
  assert that chunk appears in `recall_short_term()` output with score < direct entity match score
- Unit test: mock `query_related_entity_chunks` returning a chunk already in results;
  assert its score increases by exactly `GRAPH_BOOST_WEIGHT / (RRF_K + 1 + 1)`
- If `matched_entity_names` is empty, `_graph_neighbour_boost` skips the Memgraph call entirely
- All existing tests still green (197 currently passing)
- `ruff check --no-fix --config ruff.toml graphknows/retrieval/recall_stm.py graphknows/retrieval/stores/memgraph.py` clean

## What NOT to touch
- Any file not listed above
- `recall_ltm.py` — do not touch
- `stdio_server.py` — do not touch
- No refactors outside the new functions

## End-of-task summary format (mandatory)
```
Files changed: <list>
What was modified: <one line per file>
Files intentionally not touched: <list>
Tests added: <list>
Test results: <pytest output tail>
```
