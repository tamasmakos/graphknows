# Brief: t_75070e0c — _entity_boost drops entity-only chunks (no ANN overlap)

## Goal
Fix `_entity_boost` in `graphknows/retrieval/recall_stm.py` so entity-matched chunks that
are NOT already in the ANN results are **added** to the results dict, not silently dropped.

## Root Cause (already confirmed)
File: `graphknows/retrieval/recall_stm.py`, function `_entity_boost` (~line 107).

Current logic (simplified):
```python
for rank, ec in enumerate(entity_chunks):
    cid = ec["chunk_id"]
    if cid in results:   # ← only boosts chunks already in ANN results
        ...update score and entity_names...
    # no else branch — entity-only chunks are silently dropped
```

This means if a chunk was found via entity-graph lookup but NOT via ANN, it is lost.
The user's query mentions an entity but the embedding-based ANN didn't retrieve that chunk.
Entity-graph recall is supposed to complement ANN recall, not only boost it.

## Change 1 — Write failing test first (RED)

File: `tests/test_recall_stm.py`

Add test class `TestEntityBoostAddsNewChunks`:
```python
from graphknows.retrieval.recall_stm import _entity_boost
from graphknows.retrieval.recall_stm import RecallResult

class TestEntityBoostAddsNewChunks:
    def test_entity_only_chunk_is_added_to_results(self):
        """Chunks found via entity graph but NOT in ANN results must be added."""
        results: dict = {}  # empty — simulates no ANN hits

        class FakeMg:
            def query_entity_chunks(self, session_id, entity_names, top_k):
                import asyncio
                async def _coro():
                    return [{"chunk_id": "c1", "text": "entity chunk", "entity_names": ["obama"]}]
                return _coro()

        _entity_boost(FakeMg(), "s1", ["obama"], top_k=5, results=results)

        assert "c1" in results, "Entity-matched chunk must be added even if not in ANN results"
        score, result = results["c1"]
        assert score > 0
        assert "obama" in result.entity_names
```

Run — expect FAILURE (RED):
```bash
cd /root/graphknows && uv run pytest tests/test_recall_stm.py::TestEntityBoostAddsNewChunks -v 2>&1 | tail -20
```

Note: `_entity_boost` calls `_run(mg_store.query_entity_chunks(...))` where `_run` is a
private helper that runs a coroutine synchronously. The test must work with real async mock.
Look at `_run` in `recall_stm.py` — it's likely:
```python
def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)
```
or similar. Make the FakeMg return an async coroutine that `_run` can execute.

## Change 2 — Fix production code (GREEN)

In `_entity_boost`, add an `else` branch that inserts new entity-only chunks:

```python
for rank, ec in enumerate(entity_chunks):
    rrf = 1.0 / (RRF_K + rank + 1)
    cid = ec["chunk_id"]
    if cid in results:
        old_score, old_result = results[cid]
        new_score = old_score + rrf
        old_result.score = new_score
        old_result.entity_names = ec.get("entity_names", [])
        results[cid] = (new_score, old_result)
    else:
        # Entity-only chunk — add it with entity_names from the graph result
        results[cid] = (
            rrf,
            RecallResult(
                text=ec.get("text", ""),
                source_type="chunk",
                scope="short_term",
                score=rrf,
                entity_names=ec.get("entity_names", []),
                heading_path=ec.get("heading_path", ""),
                provenance={
                    "session_id": session_id,
                    "file_id": ec.get("file_id", ""),
                    "memory_class": ec.get("memory_class", "raw_evidence"),
                },
                metadata=ec.get("metadata", {}),
            ),
        )
```

Look at how `RecallResult` is constructed in `_run_stm_ann` (same file) for the exact
field names. Use the same pattern.

## Acceptance Criteria
- `TestEntityBoostAddsNewChunks::test_entity_only_chunk_is_added_to_results` passes (GREEN)
- Full unit suite still passes: `uv run pytest tests/ -q -m 'not integration' --tb=no 2>&1 | tail -5`
- `uv run ruff check --config ruff.toml graphknows/` — clean
- `uv run ruff format --check --config ruff.toml graphknows/` — clean

## Files to Modify
- `tests/test_recall_stm.py` — add `TestEntityBoostAddsNewChunks`
- `graphknows/retrieval/recall_stm.py` — `_entity_boost`: add `else` branch

## What NOT to do
- Do NOT change `_run_stm_ann` or `recall_short_term`
- Do NOT modify existing tests
- Do NOT touch any other files
- Do NOT add noqa comments
- Do NOT commit
