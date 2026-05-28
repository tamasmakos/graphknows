# FLUSH-002: Neo4jStore.write_session_subgraph — MERGE full session into LTM

## Ticket: t_8dc85250
## File: services/graphknows/graphknows/retrieval/stores/neo4j.py
## Tests: tests/test_flush.py (extend)

## What to build

Add `write_session_subgraph(session_id: str, stm_data: dict) -> dict` to `Neo4jStore` (async).
- Takes the dict produced by `read_session_subgraph`
- MERGEs everything into Neo4j
- Returns `{"nodes": int, "edges": int}`

### Cypher operations (all use MERGE for idempotency):

1. SESSION node:
```cypher
MERGE (s:SESSION {id: $session_id})
SET s.flushed_at = $now
```

2. FILE nodes + IN_SESSION edges:
```cypher
MERGE (f:FILE {id: $file_id})
SET f.title = $title, f.parser_name = $parser_name, f.source_hint = $source_hint
WITH f
MATCH (s:SESSION {id: $session_id})
MERGE (s)-[:IN_SESSION]->(f)
```

3. CHUNK nodes + HAS_CHUNK edges:
```cypher
MERGE (c:CHUNK {id: $chunk_id})
SET c.text = $text, c.heading_path = $heading_path, c.chunk_index = $chunk_index
WITH c
MATCH (f:FILE {id: $file_id})
MERGE (f)-[:HAS_CHUNK]->(c)
```

4. ENTITY nodes + HAS_ENTITY edges (from chunk_entity_edges):
```cypher
MERGE (e:ENTITY {name: $name})
SET e.type = $type, e.confidence = $confidence
WITH e
MATCH (c:CHUNK {id: $chunk_id})
MERGE (c)-[:HAS_ENTITY]->(e)
```

5. CO_OCCURS edges between entities sharing a chunk (build from chunk_entity_edges, group by chunk_id):
```cypher
MATCH (e1:ENTITY {name: $name1}), (e2:ENTITY {name: $name2})
MERGE (e1)-[:CO_OCCURS]->(e2)
```
Group chunk_entity_edges by chunk_id in Python, then for each chunk emit all pairs (avoid self-loops).

6. DIALOG nodes + IN_SESSION edges:
```cypher
MERGE (d:DIALOG {id: $dialog_id})
SET d.session_id = $session_id, d.started_at = $started_at
WITH d
MATCH (s:SESSION {id: $session_id})
MERGE (s)-[:IN_SESSION]->(d)
```

7. TURN nodes + HAS_TURN edges:
```cypher
MERGE (t:TURN {id: $turn_id})
SET t.session_id = $session_id, t.turn_index = $turn_index, t.role = $role, t.text = $text
WITH t
MATCH (d:DIALOG {id: $dialog_id})
MERGE (d)-[:HAS_TURN]->(t)
```

8. TURN NEXT edges:
```cypher
MATCH (t1:TURN {id: $from}), (t2:TURN {id: $to})
MERGE (t1)-[:NEXT]->(t2)
```

9. Turn-entity edges:
```cypher
MATCH (t:TURN {id: $turn_id}), (e:ENTITY {name: $name})
MERGE (t)-[:HAS_ENTITY]->(e)
```

### Use `_run()` (sync) inside `_arun()` / thread pattern consistent with rest of neo4j.py.
Count nodes = sessions+files+chunks+entities+dialogs+turns inserted (sum MERGE rows).
Count edges = all edge MERGEs.

## Tests to write
- `test_write_session_subgraph_empty` — empty stm_data → 0 nodes, 0 edges
- `test_write_session_subgraph_counts` — mock neo4j driver, assert cypher calls and counts

## CI gate (same as brief-001)
