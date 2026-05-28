# MEM-004: expose cosine_score in recall output

## Context
- Kanban: t_60927e88
- Working dir: /root/graphknows
- Venv: /root/graphknows/.venv
- Model: openrouter/deepseek/deepseek-v4-flash

## Problem
All recall hits return identical scores (~0.016393). RRF with a single source list
gives 1/(60+rank+1) — rank 0 = 0.01639, rank 1 = 0.01613, etc. These are not
meaningful for comparing result quality. Cosine similarity is already computed by
pgvector but is discarded.

## Known facts (pre-investigated)

pgvector search returns cosine_score: `1 - (embedding <=> $1::vector) AS cosine_score`
It's available in the dicts returned by:
- pg_store.search_stm_chunks() → dict keys include "cosine_score"
- pg_store.search_ltm_chunks() → same

RecallResult dataclass (recall_stm.py line 17):
```python
@dataclass
class RecallResult:
    text: str
    source_type: str  # "chunk" | "turn" | "entity"
    scope: str        # "short_term" | "long_term"
    score: float      # RRF score
    heading_path: str = ""
    entity_names: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
```

## Your task

### 1. graphknows/retrieval/recall_stm.py
Add `cosine_score: float = 0.0` field to RecallResult dataclass.
In `_run_stm_ann()` where RecallResult is constructed (line ~60), set:
  `cosine_score=chunk.get("cosine_score", 0.0)`

### 2. graphknows/retrieval/recall_ltm.py
In `_ann_chunks()` where RecallResult is constructed (line ~42), set:
  `cosine_score=chunk.get("cosine_score", 0.0)`

### 3. graphknows/mcp/stdio_server.py
In `_fmt()` inside `memory_recall`, add `"cosine_score"` to the output dict:
```python
{
    "score": round(r.score, 6),        # RRF score (unchanged)
    "cosine_score": round(r.cosine_score, 6),  # NEW: raw cosine similarity
    "type": r.source_type,
    "text": _clean(r.text),
    "session_id": r.provenance.get("session_id", session_id),
    "entities": list(r.entity_names or []),
}
```

### 4. tests/test_mcp_stdio_server.py
Add a test asserting:
- recall result hits contain "cosine_score" key
- cosine_score is a float between 0.0 and 1.0

## Acceptance criteria
- cosine_score present in recall output
- value is in range [0.0, 1.0]
- all existing tests pass
- Do not commit
