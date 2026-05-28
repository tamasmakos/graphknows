# LTM-3 Brief: Topic Summarization — LLM per Leiden community + pgvector embedding

## Working Directory
/root/graphknows

## Your Tools (native DB access via .opencode/tools/)
- neo4j-query.ts  → Cypher against Neo4j bolt://172.18.0.1:7687
- pgvector-query.ts → SQL against PostgreSQL 172.18.0.1:5432 (vectordb, user=postgres)

## Model
claude-sonnet-4.6 via GitHub Copilot

## Context: What Exists After LTM-1 + LTM-2
- Entity nodes in Neo4j now have: name, type, pagerank, community_id, betweenness
- Segment nodes: session_id, timestamp, query
- Topic nodes: exist in schema but are not yet populated
- pgvector: has session_id-scoped chunk embeddings (BAAI/bge-small-en-v1.5, dim=384)
- OPENROUTER_API_KEY in .env

## Step 1: Inspect current state
Use neo4j-query.ts:
```cypher
MATCH (e:Entity) WHERE e.community_id IS NOT NULL
RETURN e.community_id, count(e) as size ORDER BY size DESC LIMIT 15
```
Use pgvector-query.ts:
```sql
SELECT table_name FROM information_schema.tables WHERE table_schema='public';
```

## Step 2: Create pgvector migration for ltm_topics table

Create file: /root/graphknows/migrations/add_ltm_topics.sql
```sql
CREATE TABLE IF NOT EXISTS ltm_topics (
    community_id   INTEGER PRIMARY KEY,
    summary        TEXT NOT NULL,
    embedding      vector(384) NOT NULL,
    entity_count   INTEGER,
    hub_entities   TEXT,   -- JSON array of top entity names
    updated_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ltm_topics_embedding_idx
    ON ltm_topics USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
```

Run it via pgvector-query.ts to apply.
Verify: SELECT * FROM ltm_topics LIMIT 1;

## Step 3: Create services/graphknows/topic_summarizer.py

```python
"""LTM Topic Summarizer — LLM summary per Leiden community + pgvector embedding."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

log = logging.getLogger("graphknows.topic_summarizer")

MIN_COMMUNITY_SIZE = 3
MAX_CONCURRENT_LLM = 4
HUB_ENTITIES_K = 10   # top-K entities per community for the prompt
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class TopicSummaryResult:
    topics_created: int = 0
    topics_updated: int = 0
    communities_skipped: int = 0
    errors: list[str] = field(default_factory=list)


class TopicSummarizer:
    """Generates LLM summaries for each Leiden community and stores in Neo4j + pgvector."""

    def __init__(
        self,
        neo4j_driver: Any,
        pg_conn_string: str,
        openrouter_api_key: str,
        llm_model: str = "meta-llama/llama-3.1-8b-instruct",
    ) -> None:
        self._driver = neo4j_driver
        self._pg_dsn = pg_conn_string
        self._api_key = openrouter_api_key
        self._llm_model = llm_model
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    def _embed(self, text: str) -> list[float]:
        emb = self._get_embedder().encode(text, normalize_embeddings=True)
        return emb.tolist()

    async def _llm_summarize(self, entities: list[str]) -> str:
        entity_list = ", ".join(entities[:HUB_ENTITIES_K])
        prompt = (
            "These entities belong to the same research cluster in a political intelligence knowledge graph. "
            "In 2-3 concise sentences, describe the theme or topic that connects them. "
            "Be specific — name the policy area, institution type, or geopolitical dynamic. "
            f"Entities: {entity_list}"
        )
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def run(self) -> TopicSummaryResult:
        """Summarize all communities with >= MIN_COMMUNITY_SIZE entities."""
        import asyncpg
        result = TopicSummaryResult()

        # 1. Fetch communities from Neo4j
        with self._driver.session() as s:
            communities = s.run(
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "WITH e.community_id as cid, count(e) as sz "
                "WHERE sz >= $min_size "
                "RETURN cid, sz ORDER BY sz DESC",
                min_size=MIN_COMMUNITY_SIZE,
            ).data()

        if not communities:
            log.info("No communities large enough for summarization")
            return result

        # 2. For each community, fetch top-K hub entities by pagerank
        community_entities: dict[int, list[str]] = {}
        with self._driver.session() as s:
            for c in communities:
                cid = c["cid"]
                ents = s.run(
                    "MATCH (e:Entity {community_id: $cid}) "
                    "RETURN e.name as name ORDER BY coalesce(e.pagerank,0) DESC LIMIT $k",
                    cid=cid, k=HUB_ENTITIES_K,
                ).data()
                community_entities[cid] = [e["name"] for e in ents]

        # 3. Summarize communities (rate-limited)
        sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)

        async def summarize_one(cid: int, entities: list[str]) -> tuple[int, str]:
            async with sem:
                try:
                    summary = await self._llm_summarize(entities)
                    return cid, summary
                except Exception as e:
                    log.warning("LLM summary failed for community %d: %s", cid, e)
                    return cid, f"Community of {len(entities)} entities including: {', '.join(entities[:3])}"

        summaries = await asyncio.gather(*[
            summarize_one(c["cid"], community_entities[c["cid"]])
            for c in communities
        ])

        # 4. Write Topic nodes to Neo4j + embeddings to pgvector
        pg = await asyncpg.connect(self._pg_dsn)
        try:
            for cid, summary in summaries:
                hub_names = community_entities[cid]
                entity_count = next(c["sz"] for c in communities if c["cid"] == cid)
                embedding = self._embed(summary)

                # Neo4j Topic node
                with self._driver.session() as s:
                    existing = s.run(
                        "MATCH (t:Topic {community_id: $cid}) RETURN t", cid=cid
                    ).single()

                    s.run(
                        "MERGE (t:Topic {community_id: $cid}) "
                        "SET t.summary=$summary, t.entity_count=$ec, t.updated_at=datetime() "
                        "WITH t "
                        "MATCH (e:Entity {community_id: $cid}) "
                        "MERGE (e)-[:BELONGS_TO]->(t)",
                        cid=cid, summary=summary, ec=entity_count,
                    )

                    if existing:
                        result.topics_updated += 1
                    else:
                        result.topics_created += 1

                # pgvector upsert
                await pg.execute(
                    """INSERT INTO ltm_topics (community_id, summary, embedding, entity_count, hub_entities, updated_at)
                    VALUES ($1, $2, $3::vector, $4, $5, NOW())
                    ON CONFLICT (community_id) DO UPDATE
                    SET summary=$2, embedding=$3::vector, entity_count=$4, hub_entities=$5, updated_at=NOW()""",
                    cid, summary, json.dumps(embedding), entity_count, json.dumps(hub_names),
                )
        finally:
            await pg.close()

        result.communities_skipped = len([
            c for c in (
                c2 for c2 in []  # communities below threshold were already filtered
            )
        ])
        log.info(
            "Topic summarization: %d created, %d updated",
            result.topics_created, result.topics_updated,
        )
        return result
```

## Step 4: Wire into WriteBackService

After feature_engineering call in write_back(), add:
```python
from graphknows.topic_summarizer import TopicSummarizer
import os, asyncio as _asyncio

async def _run_topics():
    try:
        ts = TopicSummarizer(
            neo4j_driver=GraphDatabase.driver(self._uri, auth=(self._user, self._password)),
            pg_conn_string=os.environ["PGVECTOR_DSN"],
            openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
        )
        res = await ts.run()
        log.info("Topic summarization result: %s", res)
    except Exception:
        log.exception("Topic summarization failed — non-fatal")

try:
    loop = _asyncio.get_event_loop()
    loop.create_task(_run_topics())
except RuntimeError:
    _asyncio.run(_run_topics())
```

## Step 5: Verify

Use neo4j-query.ts:
```cypher
MATCH (t:Topic) RETURN t.community_id, t.summary, t.entity_count LIMIT 10
```
Use pgvector-query.ts:
```sql
SELECT community_id, LEFT(summary, 80) as summary_preview, entity_count FROM ltm_topics ORDER BY entity_count DESC LIMIT 10;
```

Semantic query test (embed 'EU artificial intelligence regulation' and run ANN):
```sql
-- (embedding vector needs to be computed externally and inserted as literal)
SELECT community_id, summary FROM ltm_topics ORDER BY embedding <=> '[0.1,0.2,...]'::vector LIMIT 3;
```

Import check:
```bash
PYTHONPATH=/root/graphknows/services:/root/graphknows/services/graphknows python3 -B -c   "from graphknows.topic_summarizer import TopicSummarizer; print('OK')"
```

## Acceptance Criteria
- migrations/add_ltm_topics.sql created and applied (table exists in pgvector)
- services/graphknows/topic_summarizer.py created and importable
- Topic nodes appear in Neo4j after write-back with non-empty summary
- ltm_topics rows appear in pgvector with embedding column populated
- LLM concurrency limited to MAX_CONCURRENT_LLM=4 via semaphore
- Do NOT commit — leave for principal review

## Pitfalls
- asyncpg connect string: use PGVECTOR_DSN env var (postgresql://postgres:PASSWORD@172.18.0.1:5432/vectordb)
- vector() cast: pgvector expects the embedding as text like '[0.1,0.2,...]' cast to ::vector
- asyncio.gather inside an existing event loop: ensure TopicSummarizer.run() is called with await, not asyncio.run()
- asyncpg may not be in agent venv — check with: python3 -c "import asyncpg" ; if missing: uv pip install asyncpg --python /root/graphknows/services/agent/.venv/bin/python3
