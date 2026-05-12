---
title: "Dual-Write & Hybrid Retrieval"
created: 2026-05-12
updated: 2026-05-12
type: concept
tags: [retrieval, pgvector, memgraph, rrf, dual-write]
confidence: high
---

# Dual-Write & Hybrid Retrieval

Every search iteration writes evidence to **two stores** simultaneously: pgvector (dense chunk index) and Memgraph (structured entity-relation KG), both namespaced by `session_id`. Retrieval fuses both stores via Reciprocal Rank Fusion (RRF).

See [[deep-research-agent-design]] for the agent loop context. See [[product-architecture-v2]] for infrastructure addresses.

---

## DualWriter.write()

Implemented in `graph/nodes/investigate.py` as `DualWriter`:

```python
class DualWriter:
    def __init__(self, session_id: str, pg_conn, mg_driver):
        self.session_id = session_id
        self.pg = pg_conn       # asyncpg connection to pgvector db
        self.mg = mg_driver     # neo4j bolt driver → Memgraph

    async def write(self, chunks: list[Chunk], entities: list[Entity], relations: list[Relation]):
        await asyncio.gather(
            self._write_chunks(chunks),
            self._write_graph(entities, relations),
        )

    async def _write_chunks(self, chunks):
        # embed with BAAI/bge-small-en-v1.5
        vectors = embed(chunks)
        await self.pg.executemany(
            "INSERT INTO chunks (session_id, doc_id, text, embedding, metadata) VALUES ($1,$2,$3,$4,$5)",
            [(self.session_id, c.doc_id, c.text, v, c.metadata) for c, v in zip(chunks, vectors)]
        )

    async def _write_graph(self, entities, relations):
        async with self.mg.session() as s:
            for e in entities:
                await s.run(
                    "MERGE (n:Entity {id: $id, session_id: $sid}) SET n += $props",
                    id=e.id, sid=self.session_id, props=e.to_dict()
                )
            for r in relations:
                await s.run(
                    """MATCH (a:Entity {id:$src, session_id:$sid}), (b:Entity {id:$tgt, session_id:$sid})
                       MERGE (a)-[rel:RELATES {type:$rtype, session_id:$sid}]->(b)""",
                    src=r.source, tgt=r.target, rtype=r.type, sid=self.session_id
                )
```

**Key invariants:**
- Every `Entity` and `Relation` node/edge carries `session_id` as a property — not just a label — so Cypher filters are index-friendly.
- Writes are concurrent (`asyncio.gather`) to minimise per-iteration latency.
- `MERGE` semantics ensure idempotency across iterations for the same entity.

---

## Memgraph Session Namespace Pattern

Nodes are labelled `:Entity` (generic) with `session_id` as a property, rather than dynamic per-session labels. This is intentional:

- Memgraph does not support parameterised label names in Cypher; dynamic labels require string interpolation (injection risk).
- Property-based filtering on an indexed `session_id` field is equally fast for session-scoped queries.
- Node count per session is small enough (< 10k typical) that full-index scans are sub-millisecond.

Index created at startup:
```cypher
CREATE INDEX ON :Entity(session_id);
CREATE INDEX ON :Entity(id);
```

TTL cleanup (via `session/manager.py`):
```cypher
MATCH (n:Entity {session_id: $sid}) DETACH DELETE n;
DELETE FROM chunks WHERE session_id = $sid;  -- pgvector side
```

---

## pgvector Session ID Filtering

Schema (`vectordb.chunks` table):

```sql
CREATE TABLE chunks (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL,
    doc_id      TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   VECTOR(384),        -- BAAI/bge-small-en-v1.5 dim
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON chunks (session_id);
```

All retrieval queries **must** include `WHERE session_id = $sid` before the vector operator to use the btree index and avoid cross-session bleed:

```sql
SELECT id, text, metadata, 1 - (embedding <=> $query_vec) AS score
FROM chunks
WHERE session_id = $sid
ORDER BY embedding <=> $query_vec
LIMIT 20;
```

The `session_id` filter is enforced at the `HybridRetriever` interface level — callers cannot omit it.

---

## RRF Formula

Reciprocal Rank Fusion combines the pgvector dense ranking and the Memgraph graph-proximity ranking:

```
RRF_score(d) = Σ_r  1 / (k + rank_r(d))
```

Where:
- `k = 60` (standard RRF constant; dampens high-rank advantage)
- `r` ranges over {dense_rank, graph_rank}
- `rank_r(d)` is the 1-indexed position of document `d` in ranker `r`'s result list
- Documents absent from a ranker's list receive `rank = ∞` (score contribution = 0)

Implementation in `retrieval/hybrid_retriever.py`:

```python
def rrf(dense_hits: list[Chunk], graph_hits: list[Chunk], k=60) -> list[Chunk]:
    scores: dict[str, float] = defaultdict(float)
    for rank, chunk in enumerate(dense_hits, 1):
        scores[chunk.id] += 1 / (k + rank)
    for rank, chunk in enumerate(graph_hits, 1):
        scores[chunk.id] += 1 / (k + rank)
    all_chunks = {c.id: c for c in dense_hits + graph_hits}
    return [all_chunks[cid] for cid in sorted(scores, key=scores.__getitem__, reverse=True)][:8]
```

---

## HybridRetriever.retrieve() Interface

```python
class HybridRetriever:
    def __init__(self, session_id: str, pg_conn, mg_driver, embedder):
        ...

    async def retrieve(self, query: str) -> list[Chunk]:
        """
        1. Embed query with BAAI/bge-small-en-v1.5
        2. pgvector dense search: top-20, session_id-filtered
        3. Extract entity IDs from top-20 chunks (regex + NER-lite)
        4. 2-hop Memgraph walk from those entity IDs (session_id-filtered)
        5. Map Memgraph neighbour node IDs back to chunk IDs via doc_id join
        6. RRF re-rank dense_hits + graph_hits
        7. Return top-8
        """
```

**2-hop Memgraph walk** (step 4):
```cypher
MATCH (e:Entity {session_id: $sid})-[*1..2]-(neighbour:Entity {session_id: $sid})
WHERE e.id IN $entity_ids
RETURN DISTINCT neighbour.doc_ids AS doc_ids
```

Each entity stores `doc_ids: list[str]` — the set of chunk `doc_id` values that mention it — populated during `DualWriter._write_graph()`. This allows O(1) mapping from KG nodes back to pgvector rows without a join table.

The final top-8 chunks are passed to `brief_node` as `state.retrieval_hits`.
