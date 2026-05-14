"""Async pgvector client using asyncpg and sentence-transformers.

Provides embedding, upsert, and session-aware retrieval for chunks.
"""
from __future__ import annotations

import logging
from typing import Any

import asyncpg
from sentence_transformers import SentenceTransformer

from services.agent.graph.state import Chunk

logger = logging.getLogger(__name__)

EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBED_DIM: int = 384


class PgVectorClient:
    """Async client for pgvector with session_id namespacing.

    Attributes:
        dsn: PostgreSQL connection string.
    """

    def __init__(self, dsn: str) -> None:
        """Initialise PgVectorClient.

        Args:
            dsn: PostgreSQL DSN for the pgvector database.
        """
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._embedder: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the asyncpg connection pool.

        The embedder is lazy-loaded on first use to avoid startup delay.
        """
        self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4)

    async def close(self) -> None:
        """Close the connection pool and release resources."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-loaded sentence-transformer embedder (BAAI/bge-small-en-v1.5)."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create chunks table with session_id if not exists (idempotent).

        Table schema (ADR-002):
          id TEXT PK, session_id TEXT NOT NULL, doc_id TEXT, text TEXT,
          embedding VECTOR(384), metadata JSONB, created_at TIMESTAMPTZ

        Also creates an IVFFlat index on embedding and an index on session_id.
        If the table already exists but lacks session_id, adds the column.
        """
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id           TEXT PRIMARY KEY,
                    session_id   TEXT NOT NULL DEFAULT 'legacy',
                    doc_id       TEXT,
                    text         TEXT NOT NULL,
                    embedding    VECTOR(384),
                    metadata     JSONB,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            col_exists = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'chunks' AND column_name = 'session_id'
            """)
            if col_exists == 0:
                await conn.execute(
                    "ALTER TABLE chunks ADD COLUMN session_id TEXT NOT NULL DEFAULT 'legacy'"
                )
                logger.info("Added session_id column to chunks table")

            meta_exists = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'chunks' AND column_name = 'metadata'
            """)
            if meta_exists == 0:
                await conn.execute(
                    "ALTER TABLE chunks ADD COLUMN metadata JSONB"
                )
                logger.info("Added metadata column to chunks table")

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS chunks_session_id_idx
                ON chunks (session_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 20)
            """)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert_chunks(self, session_id: str, chunks: list[Chunk]) -> int:
        """Embed chunk texts and upsert into pgvector.

        Args:
            session_id: Session namespace for the chunks.
            chunks: List of Chunk dataclass instances to persist.

        Returns:
            Number of chunks upserted.
        """
        assert self._pool is not None
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings: list[list[float]] = self.embedder.encode(
            texts, show_progress_bar=False
        ).tolist()  # type: ignore[union-attr]

        inserted = 0
        async with self._pool.acquire() as conn:
            for chunk, emb in zip(chunks, embeddings):
                await conn.execute(
                    """
                    INSERT INTO chunks (id, session_id, doc_id, text, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb)
                    ON CONFLICT (id) DO UPDATE
                      SET text       = EXCLUDED.text,
                          embedding  = EXCLUDED.embedding,
                          metadata   = EXCLUDED.metadata,
                          session_id = EXCLUDED.session_id
                    """,
                    chunk.id,
                    session_id,
                    chunk.doc_id,
                    chunk.text,
                    "[" + ",".join(str(x) for x in emb) + "]",
                    json.dumps(chunk.metadata) if chunk.metadata else "{}",
                )
                inserted += 1
        logger.debug("Upserted %d chunks for session %s", inserted, session_id)
        return inserted

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def search_chunks(
        self, session_id: str, query: str, k: int = 20
    ) -> list[dict[str, Any]]:
        """Embed query and run cosine nearest-neighbour search filtered by session_id.

        Args:
            session_id: Session namespace to filter on.
            query: Natural-language query string to embed.
            k: Number of nearest neighbours to return (default 20).

        Returns:
            List of dicts with keys id, doc_id, text, score.
        """
        assert self._pool is not None
        q_emb: list[float] = self.embedder.encode(
            query, show_progress_bar=False
        ).tolist()  # type: ignore[union-attr]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, doc_id, text,
                       1 - (embedding <=> $3::vector) AS score
                FROM chunks
                WHERE session_id = $1
                ORDER BY embedding <=> $3::vector
                LIMIT $2
                """,
                session_id,
                k,
                str(q_emb),
            )
        return [
            {"id": r["id"], "doc_id": r["doc_id"], "text": r["text"], "score": round(float(r["score"]), 4)}
            for r in rows
        ]

    async def get_chunks_by_doc_ids(
        self, session_id: str, doc_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Fetch chunks by doc_id list, filtered by session_id.

        Args:
            session_id: Session namespace to filter on.
            doc_ids: List of document IDs to retrieve chunks for.

        Returns:
            List of dicts with keys id, doc_id, text.
        """
        assert self._pool is not None
        if not doc_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, doc_id, text
                FROM chunks
                WHERE session_id = $1 AND doc_id = ANY($2::text[])
                """,
                session_id,
                doc_ids,
            )
        return [
            {"id": r["id"], "doc_id": r["doc_id"], "text": r["text"]}
            for r in rows
        ]
