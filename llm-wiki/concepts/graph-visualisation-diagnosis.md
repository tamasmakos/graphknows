---
title: Graph Visualisation Diagnosis — Root Causes & Fix Plan
created: 2026-05-05
updated: 2026-05-05
type: concept
tags: [visualisation, graph, agent, diagnosis, fix-plan]
confidence: high
---

# Graph Visualisation Diagnosis

Comprehensive audit of the visualisation and agent pipeline.
All findings confirmed by direct Neo4j + pgvector + API inspection.

---

## Neo4j KG Schema (ground truth)

```
Chunk   {id, text, doc_id}
Entity  {name, label}
Topic   {id, community_id, summary, label}

(:Chunk)-[:MENTIONS]->(:Entity)
(:Entity)-[:IN_COMMUNITY]->(:Topic)
```

**Critical gaps:**
- Entity→Entity edges: **NONE**. Only `MENTIONS` and `IN_COMMUNITY` exist.
  The co-occurrence graph is implicit (shared chunks), not explicit edges.
- Chunk has NO `source`, `date`, `url` — only `doc_id` (e.g. `ep_debate_2024_03_13`)
- Entity has NO `community_id` property — only via `IN_COMMUNITY` relation to Topic
- pgvector `chunks` table: `(id, doc_id, text, embedding, entity_count, created_at)` — no url/date

---

## Agent Output Shape (confirmed)

```python
{
  "query": str,
  "answer": str,               # contains [chunk_X] and [ENTITY_NAME] references
  "trace": [                   # list of ReAct steps
    {
      "step": int,
      "thought": str,
      "action": str,           # e.g. "search_entities(query=\"EU strategic autonomy\", k=8)"
      "observation": list|dict,# tool output — full structured data
      "final": bool            # only on last step
    }
  ],
  "citations": [               # chunks cited in final answer
    {"chunk_id": str, "doc_id": str, "excerpt": str}
  ],
  "steps": int
}
```

**What the agent CAN retrieve but doesn't expose to frontend:**
- Topic/community nodes (via `get_community` tool)
- Entity neighbours (via `get_neighbours`)
- Multi-hop provenance chains

---

## Root Causes of Missing Graph Elements

### 1. No entity-to-entity edges in visualisation
**Cause:** Agent upserts only `MENTIONS` edges (chunk→entity).
`get_neighbours` returns co-mentioned entities but they are NOT added as edges.
**Fix:** In `_upsert_entities()` in `services/agent/main.py`: when agent calls
`get_neighbours`, parse the observation and add `RELATES_TO` edges in the session graph.

### 2. Topic nodes not shown
**Cause:** Agent sometimes calls `get_community` but the result is never upserted
into the session graph. `_upsert_entities()` only processes the final answer brackets.
**Fix:** Parse `get_community` observations from trace steps and upsert Topic nodes
with `IN_COMMUNITY` edges.

### 3. Chunk nodes have no metadata (url, date, source)
**Cause:** pgvector schema has `doc_id` but no `url`/`date`. The `doc_id`
`ep_debate_2024_03_13` encodes the date implicitly.
**Fix:** Parse `doc_id` to extract source and date. Add a lookup map from
doc_id → {institution, url, date}. Store in Chunk node properties.

### 4. Node names not readable (UNDERSCORE_CAPS)
**Cause:** Entity names stored as `EUROPEAN_STRATEGIC_AUTONOMY`.
**Fix:** Frontend: `name.replace(/_/g, ' ').toLowerCase()` then title-case.

### 5. Citations are empty (sources: [])
**Cause:** `cur.execute("SELECT doc_id, text FROM chunks WHERE id=%s", (cid,))`
The column is `id` in pgvector but the agent searches by `chunk_0`, `chunk_1` etc.
These ARE the `id` values — but it fails silently. The `WHERE id=%s` returns nothing
because `id` values don't match `chunk_X` pattern from answer.
**Fix:** Verify the id format. From inspection: ids ARE `chunk_0`, `chunk_1` etc.
The SQL should work — check that the psycopg2 connection is still open when citations run.
Better: use a fresh connection for citation lookup, not the shared `conn`.

---

## Agent Query Gaps (what's missing)

### Missing graph query: entity co-occurrence edges
Currently: `get_neighbours` returns a list but nothing stores those edges.
Should emit: `(Entity A)-[:CO_OCCURS {via_chunk: chunk_id, weight: N}]->(Entity B)`

### Missing graph query: source document nodes
Should add `Document` nodes with `{doc_id, institution, url, date}` and
`(Chunk)-[:FROM]->(Document)` edges. Makes provenance visible in graph.

### Missing graph query: topic clusters
`get_community` result should produce a `Topic` node in session graph with
`(Entity)-[:IN_COMMUNITY]->(Topic)` and `(Topic)-[:SUMMARISES]->(Chunk)` edges.

---

## Frontend Visualisation Gaps

### Node rendering
- Node labels show raw `UNDERSCORE_CAPS` — unreadable
- No metadata tooltip with full context (community, source doc, date)
- No distinction between Chunk / Entity / Topic in layout or size

### Layout
- Only force-directed layout (no hierarchical / circular / radial)
- No hierarchy: Chunk → Entity → Topic cluster
- Nodes cluster in centre — force tuning insufficient

### Edge rendering
- Only `MENTIONS` edges shown (chunk→entity)
- No co-occurrence edges (entity→entity)
- No community edges (entity→topic)
- Edge labels not legible at small sizes

---

## RAGAS Evaluation Plan

Questions to test (10 diverse, across KG topics):
1. What positions did EU leaders take on Ukraine military support?
2. Who discussed energy transition and what were their arguments?
3. What is EU strategic autonomy and which leaders mentioned it?
4. Which entities are in the maritime security community?
5. What peace negotiation efforts were discussed?
6. Who mentioned sanctions and in what context?
7. What is the economic impact of the Ukraine war according to EU debates?
8. Which EU leaders called for increased defence spending?
9. What was discussed about the Green Deal in relation to energy crisis?
10. What international law arguments were made about the conflict?

Metrics: faithfulness, answer relevance, context precision, context recall.
Tool: RAGAS library (`pip install ragas`).
Ground truth: manual annotations of 10 questions.
