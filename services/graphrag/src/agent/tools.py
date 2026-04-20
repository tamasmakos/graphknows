"""
Four retrieval tools used by the AgentWorkflow.

All tools take a Neo4j AsyncDriver and return plain Python dicts/lists
that the LLM can reason over.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 384


async def _get_embedding(text: str, model_name: str = "BAAI/bge-small-en-v1.5") -> list[float]:
    """Compute a sentence embedding using the configured model."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        model = SentenceTransformer(model_name)
        return model.encode(text, normalize_embeddings=True).tolist()
    except Exception as exc:
        logger.warning("Embedding failed (%s) — returning zero vector", exc)
        return [0.0] * _EMBEDDING_DIM


async def search_chunks(
    driver: AsyncDriver,
    query: str,
    k: int = 5,
    database: str = "neo4j",
) -> list[dict[str, Any]]:
    """
    Vector-search the Chunk embedding index and return the top-k chunks
    with their parent Document metadata.
    """
    embedding = await _get_embedding(query)
    cypher = """
    CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
    YIELD node AS c, score
    MATCH (d:Document)-[:CONTAINS]->(c)
    RETURN
        c.chunk_id   AS chunk_id,
        d.doc_id     AS doc_id,
        d.title      AS doc_title,
        c.heading_path AS heading_path,
        c.text       AS text,
        score
    ORDER BY score DESC
    """
    async with driver.session(database=database) as session:
        result = await session.run(cypher, {"k": k, "embedding": embedding})
        records = await result.data()
    return records


async def get_entity_neighbours(
    driver: AsyncDriver,
    entity_name: str,
    depth: int = 1,
    database: str = "neo4j",
) -> list[dict[str, Any]]:
    """
    Return all entities reachable from *entity_name* within *depth* hops.
    """
    cypher = """
    MATCH (e:Entity {name: $name})
    CALL apoc.path.subgraphNodes(e, {maxLevel: $depth, relationshipFilter: 'RELATED_TO'})
    YIELD node AS neighbour
    WHERE neighbour <> e
    RETURN
        neighbour.entity_id AS entity_id,
        neighbour.name      AS name,
        neighbour.type      AS type,
        neighbour.description AS description
    LIMIT 20
    """
    # Fallback without APOC
    cypher_no_apoc = """
    MATCH (e:Entity {name: $name})-[:RELATED_TO*1..2]-(n:Entity)
    WHERE n <> e
    RETURN DISTINCT
        n.entity_id   AS entity_id,
        n.name        AS name,
        n.type        AS type,
        n.description AS description
    LIMIT 20
    """
    async with driver.session(database=database) as session:
        try:
            result = await session.run(cypher, {"name": entity_name, "depth": depth})
            records = await result.data()
        except Exception:
            result = await session.run(cypher_no_apoc, {"name": entity_name})
            records = await result.data()
    return records


async def get_document_context(
    driver: AsyncDriver,
    doc_id: str,
    database: str = "neo4j",
) -> dict[str, Any]:
    """
    Return the Document metadata and its first 5 chunks.
    """
    cypher = """
    MATCH (d:Document {doc_id: $doc_id})
    OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
    WITH d, c ORDER BY c.position
    RETURN
        d.doc_id      AS doc_id,
        d.title       AS title,
        d.source_path AS source_path,
        d.created_at  AS created_at,
        collect(c.text)[0..5] AS sample_chunks
    LIMIT 1
    """
    async with driver.session(database=database) as session:
        result = await session.run(cypher, {"doc_id": doc_id})
        record = await result.single()
    return dict(record) if record else {}


async def search_entities(
    driver: AsyncDriver,
    query: str,
    k: int = 10,
    database: str = "neo4j",
) -> list[dict[str, Any]]:
    """
    Vector-search the Entity embedding index and return the top-k entities.
    Falls back to full-text name search if the vector index is unavailable.
    """
    embedding = await _get_embedding(query)
    cypher = """
    CALL db.index.vector.queryNodes('entity_embedding', $k, $embedding)
    YIELD node AS e, score
    RETURN
        e.entity_id   AS entity_id,
        e.name        AS name,
        e.type        AS type,
        e.description AS description,
        e.community_id AS community_id,
        score
    ORDER BY score DESC
    """
    cypher_fallback = """
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower($query)
    RETURN
        e.entity_id   AS entity_id,
        e.name        AS name,
        e.type        AS type,
        e.description AS description,
        e.community_id AS community_id,
        1.0           AS score
    LIMIT $k
    """
    async with driver.session(database=database) as session:
        try:
            result = await session.run(cypher, {"k": k, "embedding": embedding})
            records = await result.data()
            if not records:
                raise ValueError("empty vector result")
        except Exception:
            result = await session.run(cypher_fallback, {"query": query, "k": k})
            records = await result.data()
    return records
