"""recall_node — query LTM at session start and inject prior knowledge into state."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from services.agent.graph.state import InvestigativeState

log = logging.getLogger("agent.graph.nodes.recall")

MAX_TOPICS = 5
MAX_HUB_ENTITIES = 3
MAX_PRIOR_CHARS = 1800  # hard cap — prior_knowledge is injected into LLM prompt


def _get_embedder():
    """Lazy singleton for BAAI/bge-small-en-v1.5."""
    if not hasattr(_get_embedder, "_model"):
        from sentence_transformers import SentenceTransformer
        _get_embedder._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _get_embedder._model


async def recall_node(state: InvestigativeState) -> InvestigativeState:
    """Query LTM topics semantically and inject top-K as prior_knowledge.

    Degrades gracefully if LTM is empty or unavailable.
    Must complete in < 2 seconds.
    """
    query = state.get("topic") or ""
    if not query:
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            query = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")

    if not query:
        log.debug("recall_node: no query — skipping")
        return {**state, "prior_knowledge": ""}

    try:
        import asyncpg

        pg_dsn = os.environ.get("PGVECTOR_DSN", "")
        if not pg_dsn:
            log.debug("recall_node: PGVECTOR_DSN not set — skipping")
            return {**state, "prior_knowledge": ""}

        # 1. Embed query
        embedding = _get_embedder().encode(query, normalize_embeddings=True).tolist()
        embedding_str = json.dumps(embedding)

        # 2. ANN search over ltm_topics
        pg = await asyncpg.connect(pg_dsn, timeout=5)
        try:
            rows = await pg.fetch(
                "SELECT community_id, summary, entity_count, hub_entities "
                "FROM ltm_topics "
                f"ORDER BY embedding <=> $1::vector LIMIT {MAX_TOPICS}",
                embedding_str,
            )
        finally:
            await pg.close()

        if not rows:
            log.debug("recall_node: ltm_topics empty — skipping")
            return {**state, "prior_knowledge": ""}

        # 3. Fetch hub entity names from Neo4j for each community
        from neo4j import AsyncGraphDatabase
        neo4j_uri = os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687")
        neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_pw = os.environ.get("NEO4J_PASSWORD", "")

        driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))
        topic_blocks = []
        try:
            async with driver.session() as s:
                for row in rows:
                    cid = row["community_id"]
                    summary = row["summary"]
                    # hub_entities stored as JSON array in pgvector
                    hub = json.loads(row["hub_entities"] or "[]")[:MAX_HUB_ENTITIES]
                    topic_blocks.append(
                        f"- {summary} [key entities: {', '.join(hub)}]"
                    )
        finally:
            await driver.close()

        # 4. Build prior_knowledge string
        prior = "Relevant prior research from long-term memory:\n" + "\n".join(topic_blocks)
        if len(prior) > MAX_PRIOR_CHARS:
            prior = prior[:MAX_PRIOR_CHARS] + "..."

        log.info("recall_node: injecting %d topic clusters (%d chars)", len(rows), len(prior))
        return {**state, "prior_knowledge": prior}

    except Exception:
        log.exception("recall_node failed — degrading gracefully")
        return {**state, "prior_knowledge": ""}
