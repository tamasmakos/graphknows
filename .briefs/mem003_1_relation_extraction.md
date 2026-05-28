# MEM-003/1 — Wire DSPyExtractor into flush; dynamic ontology-driven edge labels

## Working directory
`/root/graphknows` — always run from here.

## Read first
```
cat .hermes/instructions/python-style.instructions.md
cat .hermes/instructions/tdd-test-driven-development.instructions.md
```

## Goal
During `SessionFlusher.flush()`, after entities are written to Neo4j, run the existing
`DSPyExtractor` over each chunk's text. The extractor returns triplets with **dynamic
semantic relation labels** from the taxonomy (e.g. `REGULATES`, `FUNDS`, `APPOINTS`,
`SANCTIONS`). These become **actual Cypher relationship types** in Neo4j — not a
generic `:RELATION` wrapper with a property.

**CO_OCCURS must be completely removed** — from `_write_evidence_graph` and from the
GDS projection in `run_gds_enrichment`. If relation extraction raises for ANY chunk,
flush must propagate that exception immediately (fail-hard, no catch).

## Files to touch (ONLY these)
- `services/graphknows/graphknows/retrieval/stores/neo4j.py`
- `services/graphknows/graphknows/memory/flush.py`
- `tests/test_neo4j_store.py` (new file)
- `tests/test_flush.py`

Do NOT touch any other file.

## Step 1 — Remove CO_OCCURS from _write_evidence_graph

In `neo4j.py`, function `_write_evidence_graph` (around line 33):
Delete the final loop that calls `itertools.combinations` and writes `CO_OCCURS` edges.
Keep the `chunk_to_ents` dict — we still need it, but now it's passed to the relation
extractor instead of generating co-occurrence pairs.

The function signature must also return the `chunk_to_ents` map so `flush.py` can use
it. Change return type to `tuple[int, int, dict[str, list[str]]]` and add
`chunk_to_ents` as the third element.

## Step 2 — Dynamic relation writing helper in neo4j.py

Add a new **module-level** function (not a method) in `neo4j.py`:

```python
from graphknows.graphgen.utils.utils import standardize_label

def _write_relation_edges(
    s: Any,
    triplets: list[tuple[str, str, str]],  # (source_name, relation_label, target_name)
    entity_by_name: dict[str, dict[str, Any]],
) -> int:
    """Write typed directed edges between ENTITY nodes. Returns edge count written."""
    edges = 0
    for src, rel, tgt in triplets:
        if src not in entity_by_name or tgt not in entity_by_name:
            continue  # skip unrecognised endpoints — not an error
        # Sanitise label: uppercase, underscores only (Cypher identifier rules)
        safe_label = standardize_label(rel).upper()
        if not safe_label:
            continue
        s.run(
            f"MATCH (e1:ENTITY {{name: $src}}), (e2:ENTITY {{name: $tgt}}) "
            f"MERGE (e1)-[r:{safe_label}]->(e2) "
            f"SET r.weight = coalesce(r.weight, 0) + 1, r.evidence_count = coalesce(r.evidence_count, 0) + 1",
            src=src,
            tgt=tgt,
        )
        edges += 1
    return edges
```

Note: `standardize_label` already exists in `graphknows.graphgen.utils.utils` — import
it at the top of the module (it's already used elsewhere in the project).

## Step 3 — GDS projection update

In `run_gds_enrichment` (around line 769), the current GDS graph projection projects
`CO_OCCURS`. Replace with a Cypher projection that picks up ALL relationship types
between ENTITY nodes:

```python
s.run(
    f"CALL gds.graph.project.cypher($gname, "
    f"  'MATCH (e:ENTITY {{session_id: $sid}}) RETURN id(e) AS id', "
    f"  'MATCH (e1:ENTITY {{session_id: $sid}})-[r]->(e2:ENTITY {{session_id: $sid}}) "
    f"   RETURN id(e1) AS source, id(e2) AS target, coalesce(r.weight, 1.0) AS weight', "
    f"  {{parameters: {{sid: $sid}}}}) YIELD graphName",
    gname=graph_name,
    sid=session_id,
)
```

## Step 4 — _extract_and_write_relations in flush.py

Add a new private async method to `SessionFlusher`:

```python
async def _extract_and_write_relations(
    self,
    neo4j_session: Any,
    stm_data: dict[str, Any],
    entity_by_name: dict[str, dict[str, Any]],
    chunk_to_ents: dict[str, list[str]],
) -> int:
    """Run DSPyExtractor per chunk; write dynamic typed edges. Returns total edges."""
    from graphknows.graphgen.pipeline.entity_relation.extractors import DSPyExtractor

    extractor = DSPyExtractor(config={
        "llm": {"model": "openrouter/google/gemini-flash-1.5"},
        "openrouter_api_key": self._openrouter_key,
    })

    all_triplets: list[tuple[str, str, str]] = []
    for chunk in stm_data.get("chunks", []):
        cid = chunk["id"]
        ents = chunk_to_ents.get(cid, [])
        if len(ents) < 2:
            continue  # need at least 2 entities to form a relation
        # extract_relations returns (relations_list, nodes_data_list)
        relations, _ = await extractor.extract_relations(
            text=chunk["text"],
            keywords=ents,
            entities=ents,
        )
        # relations elements are tuples: (source, relation, target, metadata_dict)
        for item in relations:
            src, rel, tgt = item[0], item[1], item[2]
            all_triplets.append((src, rel, tgt))

    # Raises are intentionally NOT caught — fail-hard per spec
    return _write_relation_edges(neo4j_session, all_triplets, entity_by_name)
```

Then in `_write_ltm_graph`, call it after `_write_evidence_graph`:

```python
nodes, edges, chunk_to_ents = _write_evidence_graph(s, session_id, stm_data, entity_by_name, now)
rel_edges = await self._extract_and_write_relations(s, stm_data, entity_by_name, chunk_to_ents)
edges += rel_edges
```

Note: `_write_evidence_graph` now returns a 3-tuple (update the call-site accordingly).

## Step 5 — Tests (TDD — write failing tests FIRST)

### tests/test_neo4j_store.py (new file)
```python
"""Unit tests for neo4j.py module-level helpers."""
from unittest.mock import MagicMock, call
from graphknows.retrieval.stores.neo4j import _write_evidence_graph, _write_relation_edges

def _mock_session():
    s = MagicMock()
    s.run = MagicMock()
    return s

def test_write_evidence_graph_no_co_occurs():
    s = _mock_session()
    stm = {
        "files": [{"id": "f1", "title": "T", "parser_name": "plain", "source_hint": ""}],
        "chunks": [{"id": "c1", "text": "text", "heading_path": "", "chunk_index": 0, "file_id": "f1"}],
        "chunk_entity_edges": [
            {"chunk_id": "c1", "entity_name": "A"},
            {"chunk_id": "c1", "entity_name": "B"},
        ],
        "dialogs": [], "turns": [], "turn_entity_edges": [], "next_edges": [],
    }
    entity_by_name = {"A": {"type": "ORG", "confidence": 0.9}, "B": {"type": "ORG", "confidence": 0.9}}
    nodes, edges, c2e = _write_evidence_graph(s, "sess-1", stm, entity_by_name, "2026-01-01")
    calls_str = " ".join(str(c) for c in s.run.call_args_list)
    assert "CO_OCCURS" not in calls_str
    assert "HAS_ENTITY" in calls_str
    assert "A" in c2e.get("c1", [])
    assert "B" in c2e.get("c1", [])

def test_write_relation_edges_dynamic_labels():
    s = _mock_session()
    eb = {"EU": {}, "ECB": {}}
    triplets = [("EU", "REGULATES", "ECB"), ("EU", "FUNDS", "ECB")]
    count = _write_relation_edges(s, triplets, eb)
    assert count == 2
    calls_str = " ".join(str(c) for c in s.run.call_args_list)
    assert "REGULATES" in calls_str
    assert "FUNDS" in calls_str

def test_write_relation_edges_skips_unknown_endpoint():
    s = _mock_session()
    eb = {"EU": {}}
    triplets = [("EU", "REGULATES", "GHOST")]  # GHOST not in entity_by_name
    count = _write_relation_edges(s, triplets, eb)
    assert count == 0
    s.run.assert_not_called()
```

### tests/test_flush.py — add these two tests
```python
@pytest.mark.asyncio
async def test_flush_calls_relation_extraction(monkeypatch):
    from graphknows.memory.flush import SessionFlusher
    from graphknows.graphgen.pipeline.entity_relation.extractors import DSPyExtractor

    async def fake_extract(self, text, keywords=None, entities=None, **kw):
        return ([("EU", "REGULATES", "ECB", {})], [])

    monkeypatch.setattr(DSPyExtractor, "extract_relations", fake_extract)
    # ... mock mg, neo4j, pg stores, call flusher.flush("test-sess")
    # assert neo4j session.run was called with "REGULATES" in the query string

@pytest.mark.asyncio
async def test_flush_propagates_extraction_error(monkeypatch):
    from graphknows.memory.flush import SessionFlusher
    from graphknows.graphgen.pipeline.entity_relation.extractors import DSPyExtractor

    async def boom(self, *a, **kw):
        raise RuntimeError("LLM failed")

    monkeypatch.setattr(DSPyExtractor, "extract_relations", boom)
    # ... mock stores with a chunk that has 2+ entities
    # assert flush() raises RuntimeError
```

(Adapt the mock setup to match the existing test_flush.py fixture patterns — look at
how other tests there mock mg/neo4j/pg.)

## Validation after implementation

```bash
cd /root/graphknows
/root/graphknows/.venv/bin/python -B -m pytest tests/test_neo4j_store.py tests/test_flush.py -x -q -m "not integration" 2>&1 | tail -20
ruff format services/ && ruff check --no-fix --config ruff.toml services/
python3 scripts/check_imports.py
```

Then verify live Neo4j (after a real flush):
```bash
python3 .opencode/tools/internal/db_debug.py neo4j-query "MATCH ()-[r:CO_OCCURS]->() RETURN count(r) as c" neo4j 5
# must be 0

python3 .opencode/tools/internal/db_debug.py neo4j-query \
  "MATCH (e1:ENTITY)-[r]->(e2:ENTITY) WHERE type(r) NOT IN ['HAS_ENTITY','IN_SESSION','HAS_CHUNK','HAS_TURN','NEXT'] RETURN type(r) as label, count(r) as c ORDER BY c DESC LIMIT 15" \
  neo4j 15
# should show REGULATES, FUNDS, etc.
```

## Acceptance criteria
1. `CO_OCCURS` does not appear in production code or Neo4j after flush
2. Dynamic edge labels (from taxonomy) appear in Neo4j after flush with entity-rich chunks
3. `_write_evidence_graph` returns 3-tuple; call-sites updated
4. `_write_relation_edges` is a module-level function (not a method) — testable standalone
5. flush raises on extraction failure — unit test confirms
6. All unit tests pass; ruff + check_imports clean

## End-of-task report
```
Files changed: <list>
What was modified: <one line per file>
Files intentionally not touched: <list>
Follow-up needed: <list or none>
```

Do NOT commit. Do NOT push.
