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
