# RECALL-003: Filter tool/raw-output turns from LTM flush

## Working directory
`/root/graphknows`

## Coding standards
- `.hermes/instructions/python-style.instructions.md`
- `.hermes/instructions/tdd-test-driven-development.instructions.md`

## Problem
`SessionFlusher.flush()` → `pg.promote_to_ltm(session_id)` does a blind bulk copy:
```sql
INSERT INTO ltm_chunks SELECT ... FROM stm_chunks WHERE session_id = $1
INSERT INTO ltm_turns  SELECT ... FROM stm_turns  WHERE session_id = $1
```
Tool call blobs, terminal output, raw JSON dumps all get promoted to LTM verbatim
and surface as recall hits. Confirmed: one of 4 live `ltm_chunks` rows is an
`ASSISTANT:\n\nTOOL_CALLS:\nexecute_code\n\n...` blob.

## Root cause
`graphknows/retrieval/stores/pgvector.py` — `promote_to_ltm()`:
no WHERE filter on `metadata->>'atom_type'` or `role`.

## Fix
Add WHERE filters to BOTH INSERT SELECTs inside `promote_to_ltm()`.

**First, read the function** (line ~506) to see the exact SQL and confirm the
`stm_chunks.metadata` column is JSONB and `stm_turns.role` is TEXT. Then apply:

For the stm_chunks → ltm_chunks INSERT:
```sql
WHERE session_id = $1
  AND (
    metadata->>'atom_type' IS NULL
    OR metadata->>'atom_type' NOT IN ('tool_call', 'tool_result', 'terminal')
  )
```

For the stm_turns → ltm_turns INSERT:
```sql
WHERE session_id = $1
  AND role IN ('user', 'assistant')
```

The return tuple `(chunks_promoted, turns_promoted)` should reflect the filtered counts.
Use `GET DIAGNOSTICS` or `SELECT count(*)` with the same WHERE to get the right number.

## Files to touch (ONLY these)
1. `graphknows/retrieval/stores/pgvector.py` — `promote_to_ltm()` only
2. `tests/test_flush.py` — add tests for filtered promotion (or new file if test_flush.py doesn't exist — check first)

## Acceptance criteria
- Unit test: mock stm_chunks with rows having atom_type='tool_call', 'tool_result', and None.
  Assert only the None row is promoted to ltm_chunks.
- Unit test: mock stm_turns with role='tool', 'user', 'assistant'.
  Assert only 'user' and 'assistant' rows are promoted.
- Return tuple counts match number of rows actually promoted (not total stm rows)
- All existing tests green
- `ruff check --no-fix --config ruff.toml graphknows/retrieval/stores/pgvector.py` clean

## What NOT to touch
- `memory/flush.py` — do not touch
- Any other file not listed above
- No refactors outside `promote_to_ltm()`

## End-of-task summary format (mandatory)
```
Files changed: <list>
What was modified: <one line per file>
Files intentionally not touched: <list>
Tests added: <list>
Test results: <pytest output tail>
```
