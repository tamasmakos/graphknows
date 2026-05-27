"""IVectorDb — Protocol for pgvector-style vector database operations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IVectorDb(Protocol):
    """Protocol for vector database operations (pgvector boundary)."""

    async def connect(self) -> None:
        """Open the asyncpg connection pool."""
        ...

    async def close(self) -> None:
        """Drain and close the connection pool."""
        ...

    async def ensure_schema(self) -> None:
        """Create tables and indexes idempotently."""
        ...

    async def upsert_chunks(
        self,
        session_id: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Upsert chunks into the chunks table under session_id."""
        ...

    async def upsert_entities(self, entities: list[dict[str, Any]]) -> None:
        """Upsert corpus-level entity embeddings."""
        ...

    async def search_chunks(
        self,
        query_emb: list[float],
        session_id: str,
        k: int = 20,
    ) -> list[dict[str, Any]]:
        """ANN search on chunks filtered to session_id."""
        ...

    async def search_entities(
        self,
        query_emb: list[float],
        k: int = 10,
        entity_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search on entity embeddings (corpus-level)."""
        ...

    async def get_chunks_by_ids(
        self,
        chunk_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Fetch full chunk records keyed by ID."""
        ...

    async def delete_session(self, session_id: str) -> int:
        """Delete all chunks for session_id. Returns deleted row count."""
        ...

    async def write_stm_chunk(self, chunk: dict[str, Any]) -> None:
        """INSERT a chunk into stm_chunks."""
        ...

    async def write_stm_turn(self, turn: dict[str, Any]) -> None:
        """INSERT a turn into stm_turns."""
        ...

    async def search_stm_chunks(
        self,
        session_id: str,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """ANN search on stm_chunks filtered to session_id."""
        ...

    async def search_stm_turns(
        self,
        session_id: str,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """ANN search on stm_turns filtered to session_id."""
        ...

    async def promote_to_ltm(self, session_id: str) -> tuple[int, int]:
        """Copy stm_chunks and stm_turns to their ltm_* counterparts."""
        ...

    async def clear_stm(self, session_id: str) -> None:
        """DELETE all stm_chunks and stm_turns for session_id."""
        ...

    async def search_ltm_chunks(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search on ltm_chunks (corpus-wide, optionally filtered by session)."""
        ...

    async def search_ltm_turns(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search on ltm_turns (interaction/dialog recall)."""
        ...

    async def write_ltm_topic(self, topic: dict[str, Any]) -> None:
        """UPSERT a topic into ltm_topics."""
        ...
