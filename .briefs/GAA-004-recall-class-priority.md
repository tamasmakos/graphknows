# GAA-004: Metadata-aware recall preference — Brief

## Context
Working directory: `/root/graphknows`
GAA-001, GAA-002, GAA-003 are complete.

Read before starting:
```
cat /root/graphknows/.hermes/instructions/python-style.instructions.md
cat /root/graphknows/.hermes/instructions/tdd-test-driven-development.instructions.md
```

Also read these files before writing any code:
- `graphknows/retrieval/recall_stm.py`
- `graphknows/retrieval/recall_ltm.py`
- `graphknows/memory/adapter_types.py` (for `MemoryClass` values)

## Objective
Add `memory_class`-aware score shaping to STM and LTM recall so distilled memory
(`state`, `episode`, `fact`, `summary`) is preferred over `raw_evidence` in
default continuity queries. No new indices, no separate lanes.

## Problem being fixed
Discovery phase showed: raw `terminal_execution` chunks outranked distilled
`state`/`episode` objects because RRF scoring is blind to `memory_class`.
Solution: apply a multiplicative boost/de-boost factor after RRF fusion,
before returning results.

## Score-shaping rules (deterministic, no LLM)

### Boost factors (multiply final RRF score by)
| memory_class value  | factor |
|---|---|
| `fact`              | 1.5    |
| `state`             | 1.4    |
| `episode`           | 1.3    |
| `summary`           | 1.2    |
| `raw_evidence`      | 0.6    |
| anything else / None | 1.0   |

### Where to apply
The shaping applies to the result list **after** RRF fusion, before returning.
Read `memory_class` from `result["metadata"].get("memory_class")` (string).

### When NOT to apply
If the caller passes `prefer_classes=[]` explicitly or `include_raw=True` and
the query looks like a debug/terminal query (heuristic: not needed for v1 —
always apply shaping in v1 to keep it simple).

## Changes required

### `graphknows/retrieval/recall_stm.py`
- Add a module-level helper: `_shape_score(score: float, memory_class: str | None) -> float`
  that applies the boost factor table above.
- After building the RRF-fused results list, apply `_shape_score` to each
  result's score, then re-sort by descending score before returning.
- Do not change the function signature of `recall_short_term`.
- The helper must be importable as a private function (prefix `_`).

### `graphknows/retrieval/recall_ltm.py`
- Apply the same shaping pattern: add `_shape_score` (or import from recall_stm
  if it makes sense — but prefer a local copy to keep modules independent).
- Apply after RRF fusion, before returning results.
- Do not change the function signature of `recall_long_term`.

## Tests to write

### `tests/test_recall_memory_class_priority.py`

Required test cases (all unit tests, no live DB):
- `test_shape_score_fact_boosts` — `_shape_score(1.0, "fact") == 1.5`
- `test_shape_score_raw_evidence_deboosts` — `_shape_score(1.0, "raw_evidence") == 0.6`
- `test_shape_score_unknown_neutral` — `_shape_score(1.0, "unknown") == 1.0`
- `test_shape_score_none_neutral` — `_shape_score(1.0, None) == 1.0`
- `test_stm_recall_distilled_outranks_raw` — mock STM stores to return:
  - one `raw_evidence` hit with RRF score 0.05
  - one `fact` hit with RRF score 0.03
  After shaping, the `fact` hit must outrank the `raw_evidence` hit.
  (Expected: fact score = 0.03 * 1.5 = 0.045, raw score = 0.05 * 0.6 = 0.03 → fact wins)
- `test_ltm_recall_distilled_outranks_raw` — same scenario for LTM recall.
- `test_shaping_preserves_top_k` — verify result count does not exceed `top_k`.
- `test_shaping_handles_missing_metadata` — hit with no `metadata` key or no `memory_class` in metadata → factor 1.0 (no crash).

## How to mock STM/LTM results
Look at `tests/test_recall.py` for the existing mock pattern.
The recall functions return a list of dicts like:
```python
{"id": "...", "text": "...", "score": 0.05, "metadata": {"memory_class": "raw_evidence"}}
```
Mock `pg_store.search_stm_chunks`, `pg_store.search_stm_turns`, and
`mg_store.graph_neighbors` to return lists of dicts that include `memory_class`
in their `metadata` field.
Then call `recall_short_term(...)` and inspect the order of returned hits.

## Acceptance criteria
- `uv run pytest tests/test_recall_memory_class_priority.py -v` — all 8 tests pass
- `uv run pytest tests/test_recall.py -v` — existing tests still pass (no regression)
- `ruff format graphknows/ tests/ && ruff check --no-fix graphknows/retrieval/recall_stm.py graphknows/retrieval/recall_ltm.py tests/test_recall_memory_class_priority.py` — no new errors
- Function signatures of `recall_short_term` and `recall_long_term` are unchanged
- `_shape_score` is a private module-level helper in each recall module

## End-of-task report required
```
Files changed: <list>
What was modified: <one line per file>
Files not touched: <list>
Test result: <pytest summary lines>
Follow-up: <any gaps>
```

## Do NOT
- Change function signatures
- Add new DB stores or indices
- Add LLM calls
- Touch any file outside recall_stm.py, recall_ltm.py, __init__.py, and the new test file
- Commit or push anything
