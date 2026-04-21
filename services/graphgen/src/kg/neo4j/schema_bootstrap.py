"""
Neo4j schema bootstrap for GraphKnows.

Idempotent — safe to call on every startup. Extends create_indexes() with
fulltext indexes needed for keyword search.

Schema target:
    (:Document {doc_id, title, source_path, created_at})
        -[:CONTAINS]->
    (:Chunk {chunk_id, text, position, doc_id, embedding[384]})
        -[:MENTIONS]->
    (:Entity {entity_id, name, type, embedding[384]})
        -[:RELATED_TO {relation}]->
    (:Entity)
"""

from __future__ import annotations

import logging

from neo4j import AsyncDriver

from kg.neo4j.indexes import create_indexes

logger = logging.getLogger(__name__)

_FULLTEXT_INDEXES = [
    (
        "chunk_text_fulltext",
        "CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS "
        "FOR (c:Chunk) ON EACH [c.text]",
    ),
    (
        "entity_name_fulltext",
        "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS "
        "FOR (e:Entity) ON EACH [e.name]",
    ),
]


async def bootstrap_schema(driver: AsyncDriver, database: str = "neo4j") -> None:
    """
    Ensure the full GraphKnows schema is present in Neo4j:
      - UNIQUE constraints on Document.doc_id, Chunk.chunk_id, Entity.entity_id
      - Lookup indexes for efficient property queries
      - Vector indexes (384-dim cosine) on Chunk.embedding and Entity.embedding
      - Fulltext indexes on Chunk.text and Entity.name
    """
    # Constraints, lookup indexes, and vector indexes
    await create_indexes(driver, database=database)

    # Fulltext indexes
    async with driver.session(database=database) as session:
        for name, cypher in _FULLTEXT_INDEXES:
            try:
                await session.run(cypher)
                logger.info("Fulltext index '%s' ensured.", name)
            except Exception as exc:
                logger.warning("Fulltext index '%s' warning: %s", name, exc)

    logger.info("Schema bootstrap complete.")
