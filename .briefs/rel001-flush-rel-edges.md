# REL-001: Carry :REL edges from STM into LTM on flush

## Problem
`read_session_subgraph()` in `graphknows/retrieval/stores/memgraph.py` never queries `:REL`
edges. The flush path therefore silently drops all entity-to-entity relations that GLiNER2
wrote to Memgraph STM. After flush, Neo4j LTM has zero ENTITY→ENTITY edges.

## Root cause (confirmed)
- `write_relation_edge()` in memgraph.py writes `:REL {name, session_id, confidence}` edges
  between ENTITY nodes in Memgraph.
- `read_session_subgraph()` returns a dict with keys:
  entities, files, chunks, dialogs, turns, chunk_entity_edges, turn_entity_edges, next_edges
  — no `rel_edges` key at all.
- `_write_ltm_graph()` in flush.py passes `stm_data` to neo4j store but there is nothing
  to pass for relations.
- `extract_and_write_relations()` in neo4j.py runs DSPy independently but only when
  chunk_entity_edges are non-empty (which they often aren't for MCP-ingested sessions).

## Fix — three files, minimal changes

### 1. `graphknows/retrieval/stores/memgraph.py` — add rel_edges to read_session_subgraph()

Inside the `_sync()` closure, after `next_edges`, add:

```python
rel_edges = [
    {
        "subject": r["subject"],
        "relation": r["relation"],
        "object": r["object"],
        "confidence": r["confidence"] or 0.0,
    }
    for r in s.run(
        "MATCH (a:ENTITY)-[r:REL]->(b:ENTITY) "
        "WHERE r.session_id = $sid "
        "RETURN a.name AS subject, r.name AS relation, "
        "b.name AS object, r.confidence AS confidence",
        sid=session_id,
    )
]
```

Add `"rel_edges": rel_edges` to the returned dict.

### 2. `graphknows/retrieval/stores/neo4j.py` — write rel_edges in write_session_subgraph()

Find `write_session_subgraph()`. After writing ENTITY nodes and HAS_ENTITY edges, add:

```python
# Write entity-to-entity REL edges from STM
for rel in stm_data.get("rel_edges", []):
    subj = rel.get("subject", "")
    obj = rel.get("object", "")
    name = rel.get("relation", "")
    conf = rel.get("confidence", 0.0)
    if subj and obj and name:
        await self._arun(
            "MATCH (a:ENTITY {name: $subj}), (b:ENTITY {name: $obj}) "
            "MERGE (a)-[r:REL {name: $name, session_id: $sid}]->(b) "
            "ON CREATE SET r.confidence = $conf, r.created_at = $now "
            "ON MATCH SET r.confidence = $conf",
            subj=subj, obj=obj, name=name, sid=session_id,
            conf=conf, now=datetime.now(UTC).isoformat(),
        )
        edges += 1
```

Use `_arun` (or whatever async helper the file uses). Import `datetime`/`UTC` if not already.
Look at how `_arun` is used elsewhere in the file and follow the same pattern exactly.

### 3. `graphknows/memory/flush.py` — no changes needed
`_write_ltm_graph()` already passes the full `stm_data` dict to `write_session_subgraph()`.
Once `read_session_subgraph()` includes `rel_edges`, they flow through automatically.

## Acceptance criteria
- `MATCH (a:ENTITY)-[r:REL]->(b:ENTITY) RETURN count(r)` in Neo4j returns > 0 after
  flushing a session that had :REL edges in STM.
- All existing tests pass (202 baseline — run the canonical test command below).
- At least 2 new tests in `tests/test_flush.py` covering the rel_edges flush path.
  Test names should start with `test_flush_rel_edges_`.

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
- `graphknows/retrieval/stores/memgraph.py` — add rel_edges query in read_session_subgraph()
- `graphknows/retrieval/stores/neo4j.py` — write REL edges in write_session_subgraph()
- `tests/test_flush.py` — add 2+ new tests

## Files NOT to touch
Everything else. No refactors. No style fixes.

## Working dir: /root/graphknows
## Do NOT commit.
