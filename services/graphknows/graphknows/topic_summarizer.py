"""LTM Topic Summarizer — LLM summary per Leiden community + pgvector embedding."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

log = logging.getLogger("graphknows.topic_summarizer")

MIN_COMMUNITY_SIZE = 3
MAX_CONCURRENT_LLM = 4
HUB_ENTITIES_K = 10  # top-K entities per community for the prompt
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SUBTOPIC_MIN_SIZE = 6  # minimum community size to attempt sub-Leiden


@dataclass
class TopicSummaryResult:
    topics_created: int = 0
    topics_updated: int = 0
    communities_skipped: int = 0
    subtopics_created: int = 0
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

    async def summarize_topics(self, changed_community_ids: list[int] | None = None) -> int:
        """Summarize communities. If changed_community_ids provided, only those communities.

        Returns:
            Number of topics summarized.
        """
        result = await self.run(changed_community_ids=changed_community_ids)
        return result.topics_created + result.topics_updated

    async def run(self, changed_community_ids: list[int] | None = None) -> TopicSummaryResult:
        """Summarize all (or specified) communities with >= MIN_COMMUNITY_SIZE entities."""
        import asyncpg

        result = TopicSummaryResult()

        # 1. Fetch communities from Neo4j
        with self._driver.session() as s:
            if changed_community_ids is not None:
                communities = s.run(
                    "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                    "AND e.community_id IN $cids "
                    "WITH e.community_id as cid, count(e) as sz "
                    "WHERE sz >= $min_size "
                    "RETURN cid, sz ORDER BY sz DESC",
                    cids=changed_community_ids,
                    min_size=MIN_COMMUNITY_SIZE,
                ).data()
            else:
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
                    cid=cid,
                    k=HUB_ENTITIES_K,
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
                    return (
                        cid,
                        f"Community of {len(entities)} entities including: {', '.join(entities[:3])}",
                    )

        summaries = await asyncio.gather(
            *[summarize_one(c["cid"], community_entities[c["cid"]]) for c in communities]
        )

        # 4. Write Topic nodes to Neo4j + embeddings to pgvector
        pg = await asyncpg.connect(self._pg_dsn)
        try:
            for cid, summary in summaries:
                hub_names = community_entities[cid]
                entity_count = next(c["sz"] for c in communities if c["cid"] == cid)
                embedding = self._embed(summary)

                # Compute community hash (sorted entity names → SHA1)
                all_entity_names: list[str] = []
                with self._driver.session() as s:
                    name_rows = s.run(
                        "MATCH (e:Entity {community_id: $cid}) RETURN e.name as name ORDER BY e.name",
                        cid=cid,
                    ).data()
                    all_entity_names = [r["name"] for r in name_rows]

                community_hash = hashlib.sha1(
                    "|".join(sorted(all_entity_names)).encode(),
                    usedforsecurity=False,  # content deduplication, not a security hash
                ).hexdigest()

                # Neo4j Topic node
                with self._driver.session() as s:
                    existing = s.run(
                        "MATCH (t:Topic {community_id: $cid}) RETURN t", cid=cid
                    ).single()

                    s.run(
                        "MERGE (t:Topic {community_id: $cid}) "
                        "SET t.summary=$summary, t.entity_count=$ec, "
                        "    t.community_hash=$community_hash, t.updated_at=datetime() "
                        "WITH t "
                        "MATCH (e:Entity {community_id: $cid}) "
                        "MERGE (e)-[:BELONGS_TO]->(t)",
                        cid=cid,
                        summary=summary,
                        ec=entity_count,
                        community_hash=community_hash,
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
                    cid,
                    summary,
                    json.dumps(embedding),
                    entity_count,
                    json.dumps(hub_names),
                )

                # 5. Sub-community detection for large communities
                if entity_count >= SUBTOPIC_MIN_SIZE:
                    await self._create_subtopics(cid, result)

        finally:
            await pg.close()

        log.info(
            "Topic summarization: %d created, %d updated, %d subtopics created",
            result.topics_created,
            result.topics_updated,
            result.subtopics_created,
        )
        return result

    async def _create_subtopics(self, cid: int, result: TopicSummaryResult) -> None:
        """Run sub-Leiden on entities of community cid and create SubTopic nodes."""
        sub_graph_name = f"ltm-subtopic-{cid}"
        with self._driver.session() as s:
            try:
                # Drop old sub-projection if exists
                s.run(
                    "CALL gds.graph.drop($name, false) YIELD graphName",
                    name=sub_graph_name,
                )

                # Project sub-graph for this community
                s.run(
                    f"""CALL gds.graph.project.cypher(
                        '{sub_graph_name}',
                        'MATCH (n:Entity {{community_id: {cid}}}) RETURN id(n) AS id',
                        'MATCH (a:Entity {{community_id: {cid}}})-[r]->(b:Entity {{community_id: {cid}}}) RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, 1.0) AS weight'
                    ) YIELD graphName"""
                )

                # Run sub-Leiden
                sub_result = s.run(
                    f"""CALL gds.leiden.write('{sub_graph_name}', {{
                        writeProperty: 'subcommunity_id',
                        relationshipWeightProperty: 'weight',
                        gamma: 2.0
                    }})
                    YIELD communityCount, modularity"""
                ).single()

                if sub_result and sub_result["communityCount"] > 1:
                    sub_community_count = sub_result["communityCount"]
                    log.info(
                        "Sub-communities for community %d: %d sub-communities",
                        cid,
                        sub_community_count,
                    )

                    # Fetch distinct sub-communities
                    sub_cids = s.run(
                        "MATCH (e:Entity {community_id: $cid}) "
                        "WHERE e.subcommunity_id IS NOT NULL "
                        "RETURN DISTINCT e.subcommunity_id as scid, count(e) as sz",
                        cid=cid,
                    ).data()

                    for sub in sub_cids:
                        scid = sub["scid"]
                        scid_size = sub["sz"]

                        # Get entities in this sub-community
                        sub_ents = s.run(
                            "MATCH (e:Entity {community_id: $cid, subcommunity_id: $scid}) "
                            "RETURN e.name as name ORDER BY coalesce(e.pagerank, 0) DESC LIMIT $k",
                            cid=cid,
                            scid=scid,
                            k=HUB_ENTITIES_K,
                        ).data()
                        sub_entity_names = [e["name"] for e in sub_ents]

                        # Generate sub-topic summary
                        try:
                            sub_summary = await self._llm_summarize(sub_entity_names)
                        except Exception as e:
                            log.warning("Sub-topic LLM failed for cid=%d scid=%d: %s", cid, scid, e)
                            sub_summary = f"Sub-cluster of {scid_size} entities including: {', '.join(sub_entity_names[:3])}"

                        # Create SubTopic node linked to parent Topic
                        s.run(
                            "MERGE (st:SubTopic {community_id: $cid, subcommunity_id: $scid}) "
                            "SET st.summary=$summary, st.entity_count=$ec, st.updated_at=datetime() "
                            "WITH st "
                            "MATCH (t:Topic {community_id: $cid}) "
                            "MERGE (st)-[:PART_OF]->(t)",
                            cid=cid,
                            scid=scid,
                            summary=sub_summary,
                            ec=scid_size,
                        )

                        # Link entities to SubTopic
                        s.run(
                            "MATCH (e:Entity {community_id: $cid, subcommunity_id: $scid}) "
                            "MATCH (st:SubTopic {community_id: $cid, subcommunity_id: $scid}) "
                            "MERGE (e)-[:BELONGS_TO_SUB]->(st)",
                            cid=cid,
                            scid=scid,
                        )

                        result.subtopics_created += 1

            except Exception as e:
                log.warning(
                    "Sub-community detection failed for community %d (non-fatal): %s", cid, e
                )
            finally:
                try:
                    s.run(
                        "CALL gds.graph.drop($name, false) YIELD graphName",
                        name=sub_graph_name,
                    )
                except Exception:
                    pass
