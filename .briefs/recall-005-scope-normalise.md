# RECALL-005: Normalise cross-scope scores when scope=both

## Working directory
`/root/graphknows`

## Coding standards
- `.hermes/instructions/python-style.instructions.md`
- `.hermes/instructions/tdd-test-driven-development.instructions.md`

## Problem
`memory_recall()` in `graphknows/mcp/stdio_server.py` with `scope="both"` does:
```python
all_hits = stm_hits + ltm_hits
all_hits.sort(key=lambda r: r.score, reverse=True)
return {"hits": _fmt(all_hits[:top_k])}
```
STM hits (same session, dense ANN) always outscore LTM hits (cross-session, sparser)
regardless of semantic relevance. Prior-session knowledge is systematically buried.

## Root cause
No normalisation between STM and LTM score distributions before merge.

## Fix
Before merging (only in the `scope == "both"` branch), normalise each scope's scores
to [0, 1] with min-max, then apply scope weights (stm=0.6, ltm=0.4):

```python
def _normalise_and_weight(hits: list, weight: float) -> None:
    """Min-max normalise scores in-place, then apply weight."""
    if not hits:
        return
    scores = [h.score for h in hits]
    mn, mx = min(scores), max(scores)
    rng = mx - mn
    for h in hits:
        h.score = weight * ((h.score - mn) / rng if rng else 1.0)
```

Call it only in the `scope in ("both",)` branch, AFTER both recall calls complete
and BEFORE sorting/slicing. Single-scope paths (`scope="stm"`, `scope="ltm"`)
must remain completely unchanged.

## Files to touch (ONLY this file)
1. `graphknows/mcp/stdio_server.py` — `memory_recall()` function only

## Acceptance criteria
- When `scope="stm"` or `scope="ltm"`: zero change to behaviour
- When `scope="both"`: a LTM hit with higher cosine score than a STM hit is not
  unconditionally buried (LTM weight 0.4 × normalised LTM score can exceed
  STM weight 0.6 × normalised STM score)
- Unit test: construct stm_hits=[score=0.016], ltm_hits=[score=0.033], assert
  after normalise_and_weight the ltm hit has higher final score
- All existing tests green
- `ruff check --no-fix --config ruff.toml graphknows/mcp/stdio_server.py` clean

## What NOT to touch
- `recall_stm.py`, `recall_ltm.py` — do not touch
- Any other file or function in stdio_server.py
- No refactors outside `memory_recall()`

## End-of-task summary format (mandatory)
```
Files changed: <list>
What was modified: <one line per file>
Files intentionally not touched: <list>
Tests added: <list>
Test results: <pytest output tail>
```
