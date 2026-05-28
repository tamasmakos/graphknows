# FLUSH-001: MemgraphStore.read_session_subgraph + _merge_count fix

## Ticket: t_3db4f50e
## File: services/graphknows/graphknows/retrieval/stores/memgraph.py
## Tests: tests/test_flush.py (add), tests/test_memgraph_store.py (add if needed)

## What to build

Add `read_session_subgraph(session_id: str) -> dict` to `MemgraphStore`.
Returns:
```python
{
  "entities": [{"name": str, "type": str, "confidence": float}],
  "files": [{"id": str, "title": str, "parser_name": str, "source_hint": str}],
  "chunks": [{"id": str, "file_id": str, "text": str, "heading_path": str, "chunk_index": int}],
  "dialogs": [{"id": str, "session_id": str, "started_at": str}],
  "turns": [{"id": str, "dialog_id": str, "session_id": str, "turn_index": int, "role": str, "text": str}],
  "chunk_entity_edges": [{"chunk_id": str, "entity_name": str}],
  "turn_entity_edges": [{"turn_id": str, "entity_name": str}],
  "next_edges": [{"from_turn_id": str, "to_turn_id": str}],
  "_merge_count": 0,
}
```

### Cypher queries (all sync, use `self._conn.execute()` pattern):
1. Entities: `MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)-[:HAS_ENTITY]->(e:ENTITY) RETURN DISTINCT e.name AS name, e.type AS type, e.confidence AS confidence`
2. Files: `MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE) RETURN f.id AS id, f.title AS title, f.parser_name AS parser_name, f.source_hint AS source_hint`
3. Chunks: `MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK) RETURN c.id AS id, f.id AS file_id, c.text AS text, c.heading_path AS heading_path, c.chunk_index AS chunk_index`
4. Dialogs: `MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(d:DIALOG) RETURN d.id AS id, d.session_id AS session_id, d.started_at AS started_at`
5. Turns: `MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t:TURN) RETURN t.id AS id, d.id AS dialog_id, t.session_id AS session_id, t.turn_index AS turn_index, t.role AS role, t.text AS text`
6. Chunk-entity edges: `MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)-[:HAS_ENTITY]->(e:ENTITY) RETURN c.id AS chunk_id, e.name AS entity_name`
7. Turn-entity edges: `MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t:TURN)-[:HAS_ENTITY]->(e:ENTITY) RETURN t.id AS turn_id, e.name AS entity_name`
8. Next edges: `MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t1:TURN)-[:NEXT]->(t2:TURN) RETURN t1.id AS from_turn_id, t2.id AS to_turn_id`

IMPORTANT Memgraph constraints:
- Use named node match, NOT inline property filters on non-indexed properties
- SESSION.id IS indexed — `{id: $sid}` is safe
- No ORDER BY on multi-hop paths — sort in Python if needed
- Use _sync() wrapper pattern consistent with rest of file

## Fix _resolve_entities in flush.py
Change signature to return `(resolved: list, merge_count: int)`.
Update caller in `flush()` to unpack the tuple.
Remove the dead `_merge_count` attribute hack (lines 160-165).

## Tests to write
- `test_read_session_subgraph_empty` — session with no data returns empty lists
- `test_read_session_subgraph_structure` — mock conn returns fixed rows, assert dict shape
- `test_resolve_entities_merge_count` — two entities with same lowercase name → merge_count=1

## CI gate
python3 /root/graphknows/scripts/check_imports.py
ruff format --check graphknows/ && ruff check --no-fix graphknows/
python3 -B -m pytest /root/graphknows/tests/ -x -q -m "not integration" \
  --ignore=.../test_ontology.py --ignore=.../test_shared_config.py \
  --ignore=.../test_pgvector_retrieval.py --ignore=.../test_langgraph_agent_search_eu.py \
  --ignore=.../test_ontology_classifier.py --ignore=.../test_ontology_extractor.py \
  --ignore=.../test_runner_dry_run.py
