# REL-003: Wire DSPy OntologyRelExtractor into MCP ingest path

## Context
REL-001 and REL-002 must be completed before this ticket.

After REL-001 + REL-002:
- STM :REL edges (from GLiNER2 `relation_extraction`) are carried into LTM on flush.
- SESSION→FILE→CHUNK→ENTITY chain is intact so chunks are available.

But GLiNER2 relation extraction is weak — it often returns 0 relations from short episodic
text. DSPy `OntologyRelExtractor` produces richer semantic labels (USES, IMPLEMENTS, FIXES,
DEPENDS_ON) but currently only runs at flush time in `neo4j.py extract_and_write_relations()`.

## Goal
Run DSPy relation extraction **at ingest time** (after GLiNER2), per chunk, when ≥2 entities
are found. Merge DSPy relations into `ExtractionResult.relations` before writing :REL edges
to STM. This means LTM gets richer relations via REL-001's flush path.

## Investigation first

### Step 1 — understand OntologyRelExtractor
Read `graphknows/ontology/rel_extractor.py`:
- What is the class interface? What method to call?
- What does it return? A list of dicts? Tuples? What fields?
- Does it need an LLM key? Which env var?
- Is it synchronous or async?

### Step 2 — understand current extractor pipeline
Read `graphknows/memory/extractor.py`:
- `InProcessExtractor.extract(text)` returns `ExtractionResult(entities, relations)`
- `relations` is currently populated from GLiNER2's `relation_extraction` dict
- Fields per relation: `head`, `tail`, `relation`, `score`

Read `graphknows/memory/ingestor.py`:
- `self._extractor.extract(seg.text)` → `ExtractionResult`
- Relations from result are written to STM via `write_relation_edge()`
- DSPy extractor is NOT called here currently

### Step 3 — check DSPy extractor in neo4j.py (existing flush path)
Read `graphknows/retrieval/stores/neo4j.py` `extract_and_write_relations()`:
- It instantiates `DSPyExtractor` and calls `extract_relations(text, keywords, entities)`
- Returns `(relations, _, _audit)` — a 3-tuple
- `relations` is a list of tuples: `(source, relation_label, target, metadata_dict)`

## Implementation plan

### Option A — wire into ingestor.py (preferred)
In `graphknows/memory/ingestor.py`, after calling `self._extractor.extract(seg.text)`:

```python
# If ≥2 entities, run DSPy relation extraction as second pass
if len(extraction_result.entities) >= 2 and self._rel_extractor is not None:
    entity_names = [e["name"] for e in extraction_result.entities]
    try:
        dspy_rels = self._rel_extractor.extract(seg.text, entity_names)
        # merge into extraction_result.relations
        for r in dspy_rels:
            extraction_result.relations.append({
                "head": r[0],
                "relation": r[1],
                "tail": r[2],
                "score": r[3].get("confidence", 0.5) if len(r) > 3 else 0.5,
            })
    except Exception as exc:
        log.warning("DSPy rel extraction failed: %s", exc)
```

Add `_rel_extractor: OntologyRelExtractor | None = None` as optional init param to
`MemoryIngestor.__init__()`. The MCP server (`graphknows/mcp/stdio_server.py`) instantiates
`MemoryIngestor` — check whether to wire the rel extractor there or make it optional with
lazy init.

### Option B — add to ExtractionResult pipeline in extractor.py
Less preferred — extractor.py is GLiNER2-specific. Keep DSPy separate.

## Acceptance criteria
- After ingesting text with ≥2 entities (e.g. "Memgraph and Neo4j are both graph databases
  used by graphknows for STM and LTM respectively"), STM Memgraph has ≥1
  `ENTITY-[:REL {name: <semantic_label>}]->ENTITY` edge for that session.
- The relation name is a non-empty semantic label (USES, RELATED_TO, STORES, etc.), not
  just the GLiNER2 raw relation type.
- All existing tests pass (202 baseline).
- At least 2 new tests covering the dual-extraction path (mock the DSPy extractor in tests
  — do not require a live LLM call in unit tests).
- If DSPy extraction fails (exception), ingest continues without error (graceful fallback).

## Canonical test command
```bash
cd /root/graphknows
uv run pytest tests/ -q -m "not integration" \
  --ignore=tests/test_ontology.py \
  --ignore=tests/test_ontology_extractor.py \
  --ignore=tests/test_langgraph_agent_search_eu.py \
  --ignore=tests/test_shared_config.py \
  --ignore=tests/test_hybridrag_api.py
```

## Files to touch
- `graphknows/memory/ingestor.py` — add optional DSPy second-pass
- `graphknows/mcp/stdio_server.py` — wire OntologyRelExtractor into MemoryIngestor init
  (optional, only if it doesn't require live LLM at import time)
- `tests/` — 2+ new tests with mocked DSPy extractor

## Files NOT to touch
- `graphknows/ontology/rel_extractor.py` — read only
- `graphknows/retrieval/stores/neo4j.py` — do not remove existing DSPy flush path yet

## Working dir: /root/graphknows
## Do NOT commit.
