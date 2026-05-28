# RECALL-002: Fix entity_names population on STM recall hits

## Working directory
`/root/graphknows`

## Coding standards
- `.hermes/instructions/python-style.instructions.md`
- `.hermes/instructions/tdd-test-driven-development.instructions.md`

## Problem
`recall_short_term()` in `graphknows/retrieval/recall_stm.py` returns `RecallResult`
objects with `entity_names=[]` on all STM hits. Two bugs:

**Bug A:** `_run_stm_ann()` builds `RecallResult` with `entity_names=[]` — never populated.

**Bug B:** `_entity_boost()` has `if cid in results: results[cid][1].entity_names = ...`
but no `else` branch. Entity-matched chunks NOT in the ANN top-k are silently dropped.

## Root cause
Read `graphknows/retrieval/recall_stm.py` in full before touching anything.

`_entity_boost()` (around line 109):
- queries `mg_store.query_entity_chunks(session_id, words, top_k)` — returns entity-matched chunks
- iterates results: `for ec in entity_chunks: cid = ec["chunk_id"]; if cid in results: update`
- no `else`: entity-matched chunks not already in ANN results are silently dropped

After all lanes run, nothing backfills `entity_names` for ANN-only hits.

## Fix (two-part)

### Part 1: Fix `_entity_boost` to INSERT missing entity-matched chunks
Add an `else` branch that inserts entity-matched chunks not already in ANN results:

```python
if cid in results:
    # update entity_names on existing hit, apply boost
    old_score, old_result = results[cid]
    old_result.entity_names = list(ec.get("entity_names", []))
    new_score = old_score + 1.0 / (RRF_K + len(results) + 1)
    results[cid] = (new_score, old_result)
else:
    # INSERT: entity-matched chunk not in ANN results
    results[cid] = (
        1.0 / (RRF_K + len(results) + 1),
        RecallResult(
            chunk_id=cid,
            text=ec.get("text", ""),
            score=0.0,
            cosine_score=0.0,
            source_type="entity_match",
            entity_names=list(ec.get("entity_names", [])),
            provenance={"session_id": session_id},
        ),
    )
```

### Part 2: Backfill `entity_names` for ANN-only hits
Add a `_backfill_entity_names()` helper called at the end of `recall_short_term()`
before sorting. It queries Memgraph for entity_names by chunk_id for any result
that still has `entity_names=[]`.

First check what `query_entity_chunks` returns — look at memgraph.py — and whether
there is already a method to batch-query entity names by chunk_id. If not, add:

```python
def query_chunks_entity_names(
    self, session_id: str, chunk_ids: list[str]
) -> dict[str, list[str]]:
    """Return {chunk_id: [entity_names]} for all given chunk_ids in session."""
    # Cypher (note: ENTITY and CHUNK labels are UPPERCASE in this codebase):
    # MATCH (c:CHUNK {session_id: $sid})-[:MENTIONS]->(e:ENTITY)
    # WHERE c.id IN $ids
    # RETURN c.id AS chunk_id, collect(e.name) AS entity_names
```

In `recall_short_term()`, after `_entity_boost()`:
```python
_backfill_entity_names(mg_store, session_id, results)
```

## Files to touch (ONLY these)
1. `graphknows/retrieval/recall_stm.py` — fix `_entity_boost` + add `_backfill_entity_names`
2. `graphknows/retrieval/stores/memgraph.py` — add `query_chunks_entity_names()` if needed
3. `tests/test_recall.py` — add tests asserting entity_names populated

## Constraints
- Node labels are UPPERCASE: `:ENTITY`, `:CHUNK`, relationship `:MENTIONS`
- Use `id(n)` not `elementId(n)` for Memgraph (elementId throws)
- Memgraph driver is synchronous (`mg_store` — not async)
- The `query_entity_chunks` method already exists on MemgraphStore — read it before adding new methods

## Acceptance criteria
- Unit test: mock Memgraph to return entity-matched chunk NOT in ANN results; assert
  it appears in `recall_short_term()` output with non-empty entity_names
- Unit test: mock ANN results with known chunk_ids; mock Memgraph entity query;
  assert all returned RecallResult objects have non-empty entity_names
- All existing tests still green
- `ruff check --no-fix --config ruff.toml graphknows/retrieval/recall_stm.py graphknows/retrieval/stores/memgraph.py` clean

## What NOT to touch
- Any file not listed above
- No refactors outside the listed functions
- `recall_ltm.py` — do not touch

## End-of-task summary format (mandatory)
```
Files changed: <list>
What was modified: <one line per file>
Files intentionally not touched: <list>
Tests added: <list>
Test results: <pytest output tail>
```
