"""Async pgvector store — merges PgStore (corpus) + PgVectorClient (session-scoped).

Schema
------
chunks   — session-scoped text chunks with JSONB metadata (VECTOR 384)
entities — corpus-level entity embeddings (VECTOR 384), no session_id
"""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL DEFAULT 'corpus',
    doc_id       TEXT,
    text         TEXT NOT NULL,
    embedding    VECTOR(384),
    metadata     JSONB,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS chunks_session_idx
    ON chunks (session_id);
CREATE INDEX IF NOT EXISTS chunks_emb_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 20);

CREATE TABLE IF NOT EXISTS entities (
    name        TEXT PRIMARY KEY,
    label       TEXT,
    embedding   VECTOR(384),
    chunk_count INT DEFAULT 0,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS entities_emb_idx
    ON entities USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
"""


class PgVectorStore:
    """Async asyncpg client for the hybridrag pgvector schema.

    Attributes:
        dsn: PostgreSQL connection string.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the asyncpg connection pool."""
        self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=8)

    async def close(self) -> None:
        """Drain and close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _pool_or_raise(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PgVectorStore not connected — call connect() first")
        return self._pool

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create tables and indexes idempotently."""
        async with self._pool_or_raise().acquire() as conn:
            await conn.execute(_DDL)
        logger.info("pgvector schema ensured")

    # ------------------------------------------------------------------
    # Chunk writes (session-scoped)
    # ------------------------------------------------------------------

    async def upsert_chunks(
        self,
        session_id: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Upsert *chunks* into the chunks table under *session_id*.

        Each chunk must have: id, doc_id, text, embedding (list[float]).
        Optional: metadata (dict).
        """
        if not chunks:
            return
        pool = self._pool_or_raise()
        records = [
            (
                c["id"],
                session_id,
                c.get("doc_id", ""),
                c["text"][:4000],
                json.dumps(c["embedding"]),
                json.dumps(c.get("metadata") or {}),
            )
            for c in chunks
        ]
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO chunks (id, session_id, doc_id, text, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    doc_id     = EXCLUDED.doc_id,
                    text       = EXCLUDED.text,
                    embedding  = EXCLUDED.embedding,
                    metadata   = EXCLUDED.metadata
                """,
                records,
            )
        logger.debug("Upserted %d chunks (session=%s)", len(chunks), session_id)

    # ------------------------------------------------------------------
    # Entity writes (corpus-level)
    # ------------------------------------------------------------------

    async def upsert_entities(self, entities: list[dict[str, Any]]) -> None:
        """Upsert corpus-level entity embeddings.

        Each entity must have: name, label, embedding (list[float]).
        Optional: chunk_count (int).
        """
        if not entities:
            return
        pool = self._pool_or_raise()
        records = [
            (
                e["name"],
                e.get("label", "Entity"),
                json.dumps(e["embedding"]),
                e.get("chunk_count", 0),
            )
            for e in entities
        ]
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO entities (name, label, embedding, chunk_count)
                VALUES ($1, $2, $3::vector, $4)
                ON CONFLICT (name) DO UPDATE SET
                    label       = EXCLUDED.label,
                    embedding   = EXCLUDED.embedding,
                    chunk_count = EXCLUDED.chunk_count,
                    updated_at  = NOW()
                """,
                records,
            )
        logger.debug("Upserted %d entities", len(entities))

    # ------------------------------------------------------------------
    # Chunk search (session-scoped)
    # ------------------------------------------------------------------

    async def search_chunks(
        self,
        query_emb: list[float],
        session_id: str,
        k: int = 20,
    ) -> list[dict[str, Any]]:
        """ANN search on chunks filtered to *session_id*.

        Returns list of dicts: chunk_id, doc_id, text, metadata, cosine_score.
        """
        pool = self._pool_or_raise()
        emb_str = json.dumps(query_emb)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, doc_id, text, metadata,
                       1 - (embedding <=> $1::vector) AS cosine_score
                FROM chunks
                WHERE session_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                emb_str,
                session_id,
                k,
            )
        return [
            {
                "chunk_id": r["id"],
                "doc_id": r["doc_id"] or "",
                "text": r["text"],
                "metadata": json.loads(r["metadata"] or "{}"),
                "cosine_score": float(r["cosine_score"]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Entity search (corpus-level)
    # ------------------------------------------------------------------

    async def search_entities(
        self,
        query_emb: list[float],
        k: int = 10,
        entity_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search on entity embeddings (corpus-level, no session filter).

        Args:
            query_emb: Query embedding vector.
            k: Number of results to return.
            entity_types: Optional list of label strings to filter by (e.g. ["ORG", "PERSON"]).

        Returns:
            List of dicts: name, label, score.
        """
        pool = self._pool_or_raise()
        emb_str = json.dumps(query_emb)
        async with pool.acquire() as conn:
            if entity_types:
                rows = await conn.fetch(
                    """
                    SELECT name, label,
                           1 - (embedding <=> $1::vector) AS score
                    FROM entities
                    WHERE label = ANY($2::text[])
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    emb_str,
                    entity_types,
                    k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT name, label,
                           1 - (embedding <=> $1::vector) AS score
                    FROM entities
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    emb_str,
                    k,
                )
        return [
            {"name": r["name"], "label": r["label"] or "Entity", "score": float(r["score"])}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Bulk fetch by IDs
    # ------------------------------------------------------------------

    async def get_chunks_by_ids(
        self,
        chunk_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Fetch full chunk records keyed by ID.

        Returns dict[chunk_id -> {doc_id, text, metadata}].
        """
        if not chunk_ids:
            return {}
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, doc_id, text, metadata FROM chunks WHERE id = ANY($1)",
                chunk_ids,
            )
        return {
            r["id"]: {
                "doc_id": r["doc_id"] or "",
                "text": r["text"],
                "metadata": json.loads(r["metadata"] or "{}"),
            }
            for r in rows
        }

    async def delete_session(self, session_id: str) -> int:
        """Delete all chunks for *session_id*. Returns deleted row count."""
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM chunks WHERE session_id = $1", session_id)
        deleted = int(result.split()[-1])
        logger.info("Deleted %d chunks for session %s", deleted, session_id)
        return deleted
