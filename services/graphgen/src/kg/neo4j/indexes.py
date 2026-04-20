"""
Neo4j index creation for the GraphKnows core schema.

Creates constraints and indexes required for efficient querying.
Safe to call multiple times — uses IF NOT EXISTS.
"""

from __future__ import annotations

import logging

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

# Constraints (also create implicit indexes)
_CONSTRAINTS = [
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
]

# Additional lookup indexes
_INDEXES = [
    "CREATE INDEX document_hash IF NOT EXISTS FOR (d:Document) ON (d.content_hash)",
    "CREATE INDEX chunk_doc_id IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
    "CREATE INDEX chunk_position IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id, c.position)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_community IF NOT EXISTS FOR (e:Entity) ON (e.community_id)",
]

# Vector indexes (Neo4j 5.11+)
_VECTOR_INDEXES = [
    (
        "chunk_embedding",
        "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS "
        "FOR (c:Chunk) ON (c.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    ),
    (
        "entity_embedding",
        "CREATE VECTOR INDEX entity_embedding IF NOT EXISTS "
        "FOR (e:Entity) ON (e.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    ),
]


async def create_indexes(driver: AsyncDriver, database: str = "neo4j") -> None:
    """Create all constraints, indexes, and vector indexes."""
    async with driver.session(database=database) as session:
        for cypher in _CONSTRAINTS:
            try:
                await session.run(cypher)
            except Exception as exc:
                logger.warning("Constraint creation warning: %s", exc)

        for cypher in _INDEXES:
            try:
                await session.run(cypher)
            except Exception as exc:
                logger.warning("Index creation warning: %s", exc)

        for name, cypher in _VECTOR_INDEXES:
            try:
                await session.run(cypher)
                logger.info("Vector index '%s' ensured.", name)
            except Exception as exc:
                logger.warning("Vector index '%s' warning: %s", name, exc)

    logger.info("Neo4j index setup complete.")
