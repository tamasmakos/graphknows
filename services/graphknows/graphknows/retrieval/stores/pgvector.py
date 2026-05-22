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

-- Short-term memory (ephemeral, session-scoped)
CREATE TABLE IF NOT EXISTS stm_chunks (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    file_id      TEXT,
    text         TEXT NOT NULL,
    heading_path TEXT,
    embedding    VECTOR(1536),
    metadata     JSONB,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS stm_chunks_session_idx ON stm_chunks (session_id);
CREATE INDEX IF NOT EXISTS stm_chunks_emb_idx ON stm_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=20);

CREATE TABLE IF NOT EXISTS stm_turns (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    dialog_id    TEXT,
    turn_index   INT,
    role         TEXT NOT NULL,
    text         TEXT NOT NULL,
    embedding    VECTOR(1536),
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS stm_turns_session_idx ON stm_turns (session_id);
CREATE INDEX IF NOT EXISTS stm_turns_emb_idx ON stm_turns USING ivfflat (embedding vector_cosine_ops) WITH (lists=20);

-- Long-term memory (permanent, flushed from STM)
CREATE TABLE IF NOT EXISTS ltm_chunks (
    id           TEXT PRIMARY KEY,
    session_id   TEXT,
    file_id      TEXT,
    text         TEXT NOT NULL,
    heading_path TEXT,
    embedding    VECTOR(1536),
    metadata     JSONB,
    flushed_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ltm_chunks_session_idx ON ltm_chunks (session_id);
CREATE INDEX IF NOT EXISTS ltm_chunks_emb_idx ON ltm_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=50);

CREATE TABLE IF NOT EXISTS ltm_turns (
    id           TEXT PRIMARY KEY,
    session_id   TEXT,
    dialog_id    TEXT,
    turn_index   INT,
    role         TEXT,
    text         TEXT NOT NULL,
    embedding    VECTOR(1536),
    flushed_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ltm_turns_session_idx ON ltm_turns (session_id);
CREATE INDEX IF NOT EXISTS ltm_turns_emb_idx ON ltm_turns USING ivfflat (embedding vector_cosine_ops) WITH (lists=50);

CREATE TABLE IF NOT EXISTS ltm_topics (
    topic_id     TEXT PRIMARY KEY,
    summary      TEXT,
    embedding    VECTOR(1536),
    entity_names TEXT[],
    session_ids  TEXT[],
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ltm_topics_emb_idx ON ltm_topics USING ivfflat (embedding vector_cosine_ops) WITH (lists=10);
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
        logger.info("Deleted %d chunks for session %s", deleted, session_id)  # NOSONAR
        return deleted

    # ------------------------------------------------------------------
    # STM writes
    # ------------------------------------------------------------------

    async def write_stm_chunk(self, chunk: dict[str, Any]) -> None:
        """INSERT a chunk into stm_chunks.

        Args:
            chunk: Must have id, session_id, text. Optional: file_id,
                heading_path, embedding (list[float]), metadata (dict).
        """
        pool = self._pool_or_raise()
        embedding = json.dumps(chunk["embedding"]) if chunk.get("embedding") else None
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO stm_chunks
                    (id, session_id, file_id, text, heading_path, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::vector, $7::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    session_id   = EXCLUDED.session_id,
                    file_id      = EXCLUDED.file_id,
                    text         = EXCLUDED.text,
                    heading_path = EXCLUDED.heading_path,
                    embedding    = EXCLUDED.embedding,
                    metadata     = EXCLUDED.metadata
                """,
                chunk["id"],
                chunk["session_id"],
                chunk.get("file_id"),
                chunk["text"],
                chunk.get("heading_path"),
                embedding,
                json.dumps(chunk.get("metadata") or {}),
            )

    async def write_stm_turn(self, turn: dict[str, Any]) -> None:
        """INSERT a turn into stm_turns.

        Args:
            turn: Must have id, session_id, role, text. Optional: dialog_id,
                turn_index (int), embedding (list[float]).
        """
        pool = self._pool_or_raise()
        embedding = json.dumps(turn["embedding"]) if turn.get("embedding") else None
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO stm_turns
                    (id, session_id, dialog_id, turn_index, role, text, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
                ON CONFLICT (id) DO UPDATE SET
                    session_id = EXCLUDED.session_id,
                    dialog_id  = EXCLUDED.dialog_id,
                    turn_index = EXCLUDED.turn_index,
                    role       = EXCLUDED.role,
                    text       = EXCLUDED.text,
                    embedding  = EXCLUDED.embedding
                """,
                turn["id"],
                turn["session_id"],
                turn.get("dialog_id"),
                turn.get("turn_index"),
                turn["role"],
                turn["text"],
                embedding,
            )

    async def search_stm_chunks(
        self,
        session_id: str,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """ANN search on stm_chunks filtered to *session_id*.

        Args:
            session_id: Session namespace to filter by.
            embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of dicts: id, session_id, file_id, text, heading_path,
            metadata, cosine_score.
        """
        pool = self._pool_or_raise()
        emb_str = json.dumps(embedding)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, file_id, text, heading_path, metadata,
                       1 - (embedding <=> $1::vector) AS cosine_score
                FROM stm_chunks
                WHERE session_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                emb_str,
                session_id,
                top_k,
            )
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "file_id": r["file_id"],
                "text": r["text"],
                "heading_path": r["heading_path"],
                "metadata": json.loads(r["metadata"] or "{}"),
                "cosine_score": float(r["cosine_score"]),
            }
            for r in rows
        ]

    async def search_stm_turns(
        self,
        session_id: str,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """ANN search on stm_turns filtered to *session_id*.

        Args:
            session_id: Session namespace to filter by.
            embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of dicts: id, session_id, dialog_id, turn_index, role,
            text, cosine_score.
        """
        pool = self._pool_or_raise()
        emb_str = json.dumps(embedding)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, dialog_id, turn_index, role, text,
                       1 - (embedding <=> $1::vector) AS cosine_score
                FROM stm_turns
                WHERE session_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                emb_str,
                session_id,
                top_k,
            )
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "dialog_id": r["dialog_id"],
                "turn_index": r["turn_index"],
                "role": r["role"],
                "text": r["text"],
                "cosine_score": float(r["cosine_score"]),
            }
            for r in rows
        ]

    async def promote_to_ltm(self, session_id: str) -> tuple[int, int]:
        """Copy stm_chunks and stm_turns to their ltm_* counterparts.

        Args:
            session_id: Session namespace to promote.

        Returns:
            Tuple of (chunks_promoted, turns_promoted).
        """
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            chunks_result = await conn.execute(
                """
                INSERT INTO ltm_chunks
                    (id, session_id, file_id, text, heading_path, embedding, metadata)
                SELECT id, session_id, file_id, text, heading_path, embedding, metadata
                FROM stm_chunks
                WHERE session_id = $1
                ON CONFLICT (id) DO NOTHING
                """,
                session_id,
            )
            turns_result = await conn.execute(
                """
                INSERT INTO ltm_turns
                    (id, session_id, dialog_id, turn_index, role, text, embedding)
                SELECT id, session_id, dialog_id, turn_index, role, text, embedding
                FROM stm_turns
                WHERE session_id = $1
                ON CONFLICT (id) DO NOTHING
                """,
                session_id,
            )
        chunks_promoted = int(chunks_result.split()[-1])
        turns_promoted = int(turns_result.split()[-1])
        logger.info(
            "Promoted session %s: %d chunks, %d turns to LTM",
            session_id,
            chunks_promoted,
            turns_promoted,
        )
        return chunks_promoted, turns_promoted

    async def clear_stm(self, session_id: str) -> None:
        """DELETE all stm_chunks and stm_turns for *session_id*.

        Args:
            session_id: Session namespace to clear.
        """
        pool = self._pool_or_raise()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM stm_chunks WHERE session_id = $1", session_id)
            await conn.execute("DELETE FROM stm_turns WHERE session_id = $1", session_id)
        logger.info("Cleared STM for session %s", session_id)

    async def search_ltm_chunks(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search on ltm_chunks (corpus-wide, optionally filtered by session).

        Args:
            embedding: Query embedding vector.
            top_k: Number of results to return.
            session_id: Optional session_id to narrow results.

        Returns:
            List of dicts: id, session_id, file_id, text, heading_path,
            metadata, cosine_score.
        """
        pool = self._pool_or_raise()
        emb_str = json.dumps(embedding)
        async with pool.acquire() as conn:
            if session_id:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, file_id, text, heading_path, metadata,
                           1 - (embedding <=> $1::vector) AS cosine_score
                    FROM ltm_chunks
                    WHERE session_id = $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    emb_str,
                    session_id,
                    top_k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, file_id, text, heading_path, metadata,
                           1 - (embedding <=> $1::vector) AS cosine_score
                    FROM ltm_chunks
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    emb_str,
                    top_k,
                )
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "file_id": r["file_id"],
                "text": r["text"],
                "heading_path": r["heading_path"],
                "metadata": json.loads(r["metadata"] or "{}"),
                "cosine_score": float(r["cosine_score"]),
            }
            for r in rows
        ]

    async def write_ltm_topic(self, topic: dict[str, Any]) -> None:
        """UPSERT a topic into ltm_topics.

        Args:
            topic: Must have topic_id. Optional: summary, embedding
                (list[float]), entity_names (list[str]),
                session_ids (list[str]).
        """
        pool = self._pool_or_raise()
        embedding = json.dumps(topic["embedding"]) if topic.get("embedding") else None
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ltm_topics
                    (topic_id, summary, embedding, entity_names, session_ids)
                VALUES ($1, $2, $3::vector, $4, $5)
                ON CONFLICT (topic_id) DO UPDATE SET
                    summary      = EXCLUDED.summary,
                    embedding    = EXCLUDED.embedding,
                    entity_names = EXCLUDED.entity_names,
                    session_ids  = EXCLUDED.session_ids
                """,
                topic["topic_id"],
                topic.get("summary"),
                embedding,
                topic.get("entity_names") or [],
                topic.get("session_ids") or [],
            )
