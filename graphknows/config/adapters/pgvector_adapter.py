# ruff: noqa: D102
"""PgVectorAdapter — thin delegating wrapper around PgVectorStore implementing IVectorDb."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.config.adapters.ports.vector_db import IVectorDb

if TYPE_CHECKING:
    from graphknows.memory.stores.pgvector import PgVectorStore


class PgVectorAdapter:
    """Thin adapter delegating IVectorDb calls to a PgVectorStore instance.

    Args:
        store: Underlying PgVectorStore to delegate to.
    """

    def __init__(self, store: PgVectorStore) -> None:
        self._store = store

    async def connect(self) -> None:
        return await self._store.connect()

    async def close(self) -> None:
        return await self._store.close()

    async def ensure_schema(self) -> None:
        return await self._store.ensure_schema()

    async def upsert_chunks(
        self,
        session_id: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        return await self._store.upsert_chunks(session_id, chunks)

    async def upsert_entities(self, entities: list[dict[str, Any]]) -> None:
        return await self._store.upsert_entities(entities)

    async def search_chunks(
        self,
        query_emb: list[float],
        session_id: str,
        k: int = 20,
    ) -> list[dict[str, Any]]:
        return await self._store.search_chunks(query_emb, session_id, k)

    async def search_entities(
        self,
        query_emb: list[float],
        k: int = 10,
        entity_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._store.search_entities(query_emb, k, entity_types)

    async def get_chunks_by_ids(
        self,
        chunk_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        return await self._store.get_chunks_by_ids(chunk_ids)

    async def delete_session(self, session_id: str) -> int:
        return await self._store.delete_session(session_id)

    async def write_stm_chunk(self, chunk: dict[str, Any]) -> None:
        return await self._store.write_stm_chunk(chunk)

    async def write_stm_turn(self, turn: dict[str, Any]) -> None:
        return await self._store.write_stm_turn(turn)

    async def search_stm_chunks(
        self,
        session_id: str,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        return await self._store.search_stm_chunks(session_id, embedding, top_k)

    async def search_stm_turns(
        self,
        session_id: str,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        return await self._store.search_stm_turns(session_id, embedding, top_k)

    async def promote_to_ltm(self, session_id: str) -> tuple[int, int]:
        return await self._store.promote_to_ltm(session_id)

    async def clear_stm(self, session_id: str) -> None:
        return await self._store.clear_stm(session_id)

    async def search_ltm_chunks(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._store.search_ltm_chunks(embedding, top_k, session_id)

    async def search_ltm_turns(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self._store.search_ltm_turns(embedding, top_k, session_id)

    async def write_ltm_topic(self, topic: dict[str, Any]) -> None:
        return await self._store.write_ltm_topic(topic)

    async def write_community_assignments(
        self,
        session_id: str,
        assignments: list[dict[str, Any]],
    ) -> None:
        return await self._store.write_community_assignments(session_id, assignments)


# Verify structural compatibility at import time
def _assert_protocol() -> None:
    assert issubclass(PgVectorAdapter, IVectorDb)


_assert_protocol()
