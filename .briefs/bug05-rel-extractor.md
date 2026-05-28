# BUG-05: Wire _rel_extractor in MemoryIngestor._ingest_parsed()

## Working directory
/root/graphknows

## Context

`MemoryIngestor.__init__` accepts `rel_extractor: Any | None = None` and stores it as
`self._rel_extractor`. But `_ingest_parsed()` NEVER calls it. REL edges are therefore
never written to Memgraph STM.

The MCP server already instantiates `OntologyRelExtractor` and passes it. It just never runs.

## Key facts (pre-investigated — use these, don't re-derive)

### rel_extractor interface
- Class: `graphknows/ontology/rel_extractor.py` → `OntologyRelExtractor`
- Method: `.extract(text: str, entities: list[dict], ontology_schema: str, ontology_id: str = "unknown") -> list[dict]`
- Return: list of `{subject: str, predicate: str, object: str, confidence: float}`
- **SYNCHRONOUS** — not async, no `await` needed
- ontology_schema: we can pass `"{}"` as a fallback empty schema (the method handles JSON parse errors gracefully)

### Memgraph store interface for writing relations
- Method: `MemgraphStore.write_relations(session_id: str, relations: list[Any]) -> None` (async)
- Each relation object in the list must have `.source`, `.target`, `.type` attributes
- The existing method: `graphknows/retrieval/stores/memgraph.py` line 488
- BUT: `test_memgraph_store.py` tests a `write_relation_edge()` method that doesn't exist yet

### Decision
Do NOT use `write_relations()` (deprecated) and do NOT add `write_relation_edge()` to MemgraphStore.
Instead: write a small Cypher call inline in `_ingest_parsed()` using `self._mg._driver.session()`.

Actually — re-read ingestor.py. It uses `_run(self._mg.some_async_method(...))` pattern.
The simplest approach: add a new async method `write_relation_edge` to `MemgraphStore`
that takes `(session_id, subject_name, relation_name, object_name, confidence)` and writes
a `RELATES` edge in Memgraph. This also fixes the pre-existing test failure in
`tests/test_memgraph_store.py::test_write_relation_edge_generates_correct_cypher`.

## Required Changes

### Change 1: Add `write_relation_edge` to `MemgraphStore`
File: `graphknows/retrieval/stores/memgraph.py`

Add AFTER the existing `write_relations` method (line ~510):

```python
async def write_relation_edge(
    self,
    session_id: str,
    subject_name: str,
    relation_name: str,
    object_name: str,
    fact: str = "",
    confidence: float = 0.5,
    valid_at: Any | None = None,
    created_at: Any | None = None,
) -> None:
    """Write a typed REL edge between two named entities in Memgraph STM."""
    def _sync() -> None:
        with self._driver.session() as s:
            s.run(
                """MATCH (a:Entity {name: $subject, session_id: $session_id})
                   MATCH (b:Entity {name: $object, session_id: $session_id})
                   MERGE (a)-[r:REL {name: $name, session_id: $session_id}]->(b)
                   SET r.fact = $fact,
                       r.confidence = $confidence""",
                subject=subject_name,
                object=object_name,
                name=relation_name,
                session_id=session_id,
                fact=fact,
                confidence=confidence,
            )

    await asyncio.to_thread(_sync)
```

Note: `asyncio` is already imported at the top of memgraph.py. Check the import.
Note: `Any` is already imported from `typing` or `__future__`. Check and use what's there.

### Change 2: Wire `_rel_extractor` in `MemoryIngestor._ingest_parsed()`
File: `graphknows/memory/ingestor.py`

In `_ingest_parsed()`, AFTER the entity extraction loop and BEFORE the pgvector write
(so after the block that writes entity nodes and chunk-entity edges to Memgraph), add:

```python
# Wire relation extractor — write REL edges to Memgraph STM
if self._rel_extractor is not None and len(entities) >= 2:
    try:
        relations = self._rel_extractor.extract(
            text=clean_text,
            entities=entities,
            ontology_schema="{}",
        )
        for rel in relations:
            _run(
                self._mg.write_relation_edge(
                    session_id=session_id,
                    subject_name=rel["subject"],
                    relation_name=rel["predicate"],
                    object_name=rel["object"],
                    confidence=rel.get("confidence", 0.5),
                )
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("rel_extractor failed for session=%s: %s", session_id, exc)
```

## Tests to run (existing — must pass)

The pre-existing test at `tests/test_memgraph_store.py::test_write_relation_edge_generates_correct_cypher`
tests the EXACT signature we're adding. It must pass after Change 1.

Check what the test expects:
- `await store.write_relation_edge(session_id="s1", subject_name="Viktor Orbán", relation_name="OPPOSES", object_name="Ukraine EU membership", fact="Viktor Orbán opposes Ukraine EU membership", confidence=1.0, valid_at=now, created_at=now)`
- Asserts `":REL"` in cypher, `"$name"` or `"name"` in cypher, `"$fact"` in cypher
- Asserts params: `session_id="s1"`, `subject="Viktor Orbán"`, `object="Ukraine EU membership"`, `name="OPPOSES"`, `fact=...`, `confidence=1.0`

Make sure the Cypher and params match exactly.

## New tests to write

File: `tests/test_ingestor_rel_extractor.py` (new file)

Write at minimum 4 tests:
1. When `rel_extractor is not None` and `len(entities) >= 2`, `write_relation_edge` is called once per relation
2. When `rel_extractor is None`, `write_relation_edge` is NOT called
3. When `len(entities) < 2`, `write_relation_edge` is NOT called
4. When rel_extractor `.extract()` raises an exception, ingest continues (no exception propagated)

Use mocks throughout — do NOT call GLiNER2 or OntologyRelExtractor in tests.
Mock pattern for `_run`: `patch("graphknows.memory.ingestor._run", side_effect=lambda coro: None)`
Mock `self._mg.write_relation_edge` as a `AsyncMock`.

## Coding standards

- Type annotations on all new public APIs.
- No bare `except:` — always `except Exception as exc:  # noqa: BLE001`
- 80-char line length.
- After writing code: `ruff format graphknows/ tests/ && ruff check --no-fix --config ruff.toml graphknows/ tests/`
- Run: `uv run pytest tests/test_memgraph_store.py::test_write_relation_edge_generates_correct_cypher tests/test_ingestor_rel_extractor.py -xvs`
- Run full suite: `uv run pytest tests/ --ignore=tests/test_extractor.py --ignore=tests/test_ontology.py --ignore=tests/test_ontology_extractor.py --tb=no -q 2>&1 | tail -5`
- Baseline: 201 pass, 9 pre-existing failures. After your changes: expect 203 pass, 8 pre-existing failures (test_write_relation_edge was failing, now should pass).

## Acceptance criteria

1. `MemgraphStore.write_relation_edge()` added with correct Cypher and params.
2. `MemoryIngestor._ingest_parsed()` calls `_rel_extractor.extract()` when set and entities >= 2.
3. Failures in rel_extractor are caught and logged; ingest continues.
4. `test_write_relation_edge_generates_correct_cypher` passes (was pre-existing failure).
5. New `tests/test_ingestor_rel_extractor.py` with ≥4 tests, all green.
6. `ruff check` passes with no new errors.
7. ≥201 tests passing, no regressions vs current baseline.

## End-of-task summary (mandatory)

```
Files changed: <list every file touched>
What was modified: <one line per file>
Files intentionally not touched: <list>
Follow-up needed: <list or "none">
```

Do NOT commit. Do NOT push. Leave changes in working tree for principal engineer review.
