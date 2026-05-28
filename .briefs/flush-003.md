# FLUSH-003: Neo4jStore.summarize_topics — Leiden + LLM community summarization

## Ticket: t_8432f057
## File: services/graphknows/graphknows/retrieval/stores/neo4j.py
## Also: services/graphknows/graphknows/retrieval/stores/pgvector.py (write_ltm_topic already exists)
## Tests: tests/test_flush.py (extend)

## What to build

Add `summarize_topics(session_id, openrouter_key, llm_model, pg_store) -> int` to `Neo4jStore` (async).

### Algorithm:

**Step 1: Fetch co-occurrence pairs from Neo4j for this session**
```cypher
MATCH (s:SESSION {id: $session_id})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)-[:HAS_ENTITY]->(e1:ENTITY)
MATCH (c)-[:HAS_ENTITY]->(e2:ENTITY)
WHERE e1.name < e2.name
RETURN e1.name AS a, e2.name AS b, count(c) AS weight
```
Returns entity pairs that co-occur in same chunk, weighted by how many chunks.

**Step 2: Build igraph from pairs, run Leiden**
```python
import igraph as ig
import leidenalg

g = ig.Graph()
# add vertices and edges from pairs
partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
communities = {i: list(partition[i]) for i in range(len(partition))}
# communities: {community_id: [entity_names]}
```

**Step 3: For each community, gather evidence chunks**
```cypher
MATCH (s:SESSION {id: $session_id})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)-[:HAS_ENTITY]->(e:ENTITY)
WHERE e.name IN $member_names
RETURN DISTINCT c.text AS text, c.heading_path AS heading_path
LIMIT 5
```

**Step 4: LLM summarization**
POST to `https://openrouter.ai/api/v1/chat/completions`
Model: `llm_model` param (default `meta-llama/llama-3.1-8b-instruct`)
System: "You are a knowledge graph analyst. Given a set of entities and text passages, produce a short topic title and a 2-3 sentence summary."
User: f"Entities: {', '.join(member_names)}\n\nPassages:\n{passages_text}\n\nOutput JSON: {{title: ..., summary: ...}}"
Parse JSON from response.

Skip community if < 2 members (singletons are noise).
Skip LLM call if no openrouter_key — store title=entity names joined, summary="".

**Step 5: Upsert to Neo4j TOPIC + pgvector ltm_topics**
Use existing `upsert_topics()` for Neo4j.
Use existing `write_ltm_topic()` for pgvector.
Also create `TOPIC`-`[:IN_TOPIC]`->`ENTITY` edges for each member.

### Return: count of topics created

## Tests to write
- `test_summarize_topics_no_key` — empty openrouter_key, assert topics created with entity names as title, no HTTP call
- `test_summarize_topics_skips_singletons` — community of 1 entity is skipped

## CI gate (same as brief-001)
