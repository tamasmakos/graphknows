"""
pgvector adapter — stores chunk embeddings and entity metadata for dense retrieval.

Schema:
  chunks(id TEXT PK, doc_id TEXT, text TEXT, embedding VECTOR(384),
         entity_count INT, created_at TIMESTAMPTZ)
  chunk_entities(chunk_id TEXT, entity_text TEXT, entity_label TEXT)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    text         TEXT NOT NULL,
    embedding    VECTOR(384),
    entity_count INT DEFAULT 0,
    created_at   TIMESTAMPTZ DEFAULT NOW()
)
"""

_CREATE_CHUNK_ENTITIES = """
CREATE TABLE IF NOT EXISTS chunk_entities (
    chunk_id     TEXT NOT NULL,
    entity_text  TEXT NOT NULL,
    entity_label TEXT NOT NULL
)
"""

_CREATE_EMBEDDING_INDEX = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 20)
"""

_CREATE_CHUNK_ENTITY_INDEX = """
CREATE INDEX IF NOT EXISTS chunk_entities_chunk_idx
    ON chunk_entities (chunk_id)
"""


class PgVectorAdapter:
    """
    Stores chunk embeddings and entity spans in pgvector for dense retrieval.
    Connection parameters are read from InfrastructureSettings or passed directly.
    """

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        import psycopg2

        try:
            self._conn = psycopg2.connect(self.dsn)
            return True
        except Exception as exc:
            logger.error("pgvector connect failed: %s", exc)
            return False

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def ensure_schema(self):
        """Create tables and indexes (idempotent)."""
        cur = self._conn.cursor()
        cur.execute(_CREATE_EXTENSION)
        cur.execute(_CREATE_CHUNKS)
        cur.execute(_CREATE_CHUNK_ENTITIES)
        self._conn.commit()
        # IVFFlat requires data — create after first bulk insert
        cur.close()
        logger.info("pgvector schema ensured")

    def _try_create_index(self):
        """Create IVFFlat index — call after bulk insert."""
        try:
            cur = self._conn.cursor()
            cur.execute(_CREATE_EMBEDDING_INDEX)
            cur.execute(_CREATE_CHUNK_ENTITY_INDEX)
            self._conn.commit()
            cur.close()
        except Exception as exc:
            self._conn.rollback()
            logger.warning("Could not create embedding index (need >= 1 row): %s", exc)

    # ── write ──────────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[dict]) -> int:
        """
        Insert / update chunks with embeddings.

        Each dict must have:
          id, doc_id, text, embedding (list[float] len 384), entities (list[{text, label}])
        """
        import psycopg2.extras

        cur = self._conn.cursor()
        inserted = 0
        for chunk in chunks:
            emb = chunk.get("embedding")
            if emb is None:
                continue
            cur.execute(
                """
                INSERT INTO chunks (id, doc_id, text, embedding, entity_count)
                VALUES (%s, %s, %s, %s::vector, %s)
                ON CONFLICT (id) DO UPDATE
                  SET text = EXCLUDED.text,
                      embedding = EXCLUDED.embedding,
                      entity_count = EXCLUDED.entity_count
                """,
                (
                    chunk["id"],
                    chunk.get("doc_id", ""),
                    chunk["text"],
                    str(emb),  # psycopg2 passes as string; pgvector parses it
                    len(chunk.get("entities", [])),
                ),
            )
            # Delete old entity rows then re-insert
            cur.execute("DELETE FROM chunk_entities WHERE chunk_id = %s", (chunk["id"],))
            for ent in chunk.get("entities", []):
                cur.execute(
                    "INSERT INTO chunk_entities (chunk_id, entity_text, entity_label) VALUES (%s, %s, %s)",
                    (chunk["id"], ent.get("text", ""), ent.get("label", "")),
                )
            inserted += 1
        self._conn.commit()
        cur.close()
        self._try_create_index()
        return inserted

    # ── read ───────────────────────────────────────────────────────────────────

    def search_chunks(self, query_embedding: list[float], k: int = 6) -> list[dict]:
        """Cosine nearest-neighbour search."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT id, doc_id, text, entity_count,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (str(query_embedding), str(query_embedding), k),
        )
        rows = cur.fetchall()
        cur.close()
        return [
            {
                "chunk_id": r[0],
                "doc_id": r[1],
                "text": r[2],
                "entity_count": r[3],
                "score": round(float(r[4]), 4),
            }
            for r in rows
        ]

    def get_entities_for_chunk(self, chunk_id: str) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT entity_text, entity_label FROM chunk_entities WHERE chunk_id = %s",
            (chunk_id,),
        )
        rows = cur.fetchall()
        cur.close()
        return [{"text": r[0], "label": r[1]} for r in rows]

    def chunk_count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        n = cur.fetchone()[0]
        cur.close()
        return n
