# ruff: noqa: D102
"""Neo4jAdapter — thin delegating wrapper around Neo4jStore implementing IGraphDb."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.config.adapters.ports.graph_db import IGraphDb

if TYPE_CHECKING:
    from graphknows.memory.stores.neo4j import Neo4jStore


class Neo4jAdapter:
    """Thin adapter delegating IGraphDb calls to a Neo4jStore instance.

    Args:
        store: Underlying Neo4jStore to delegate to.
    """

    def __init__(self, store: Neo4jStore) -> None:
        self._store = store

    # Satisfy structural subtyping check at runtime
    def __class_getitem__(cls, item: object) -> object:
        return cls

    async def ensure_indexes(self) -> None:
        return await self._store.ensure_indexes()

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        return await self._store.upsert_chunks(chunks)

    async def upsert_entities_and_mentions(self, chunks: list[dict[str, Any]]) -> None:
        return await self._store.upsert_entities_and_mentions(chunks)

    async def upsert_topics(self, topics: list[dict[str, Any]]) -> None:
        return await self._store.upsert_topics(topics)

    async def upsert_co_occurs(self, cooccurrences: list[tuple[str, str, int]]) -> None:
        return await self._store.upsert_co_occurs(cooccurrences)  # type: ignore[arg-type]

    async def upsert_session_entities(
        self,
        session_id: str,
        entities: list[Any],
        relations: list[Any],
    ) -> None:
        return await self._store.upsert_session_entities(session_id, entities, relations)

    async def seed_session_from_corpus(self, session_id: str, entity_names: list[str]) -> int:
        return await self._store.seed_session_from_corpus(session_id, entity_names)

    async def merge_session_entities_to_corpus(
        self,
        session_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[int, int]:
        return await self._store.merge_session_entities_to_corpus(session_id, nodes, edges)

    async def preview_session_merge(
        self,
        session_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return await self._store.preview_session_merge(session_id, nodes, edges)

    async def get_chunk_entities(self, chunk_id: str) -> list[dict[str, Any]]:
        return await self._store.get_chunk_entities(chunk_id)

    async def get_entity_neighbors(
        self,
        entity_name: str,
        depth: int = 1,
        relationship_types: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self._store.get_entity_neighbors(entity_name, depth, relationship_types)

    async def get_chunks_for_entity(self, entity_name: str, top_k: int = 5) -> list[dict[str, Any]]:
        return await self._store.get_chunks_for_entity(entity_name, top_k)

    async def get_entity_community(self, entity_name: str) -> dict[str, Any] | None:
        return await self._store.get_entity_community(entity_name)

    async def write_session_subgraph(
        self, session_id: str, stm_data: dict[str, Any]
    ) -> dict[str, int]:
        return await self._store.write_session_subgraph(session_id, stm_data)

    async def count_nodes(self) -> dict[str, int]:
        return await self._store.count_nodes()

    async def get_segment_info(self, segment_id: str) -> dict[str, Any] | None:
        return await self._store.get_segment_info(segment_id)

    async def list_segments(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._store.list_segments(limit)


# Verify structural compatibility at import time (will raise TypeError if broken)
def _assert_protocol() -> None:
    assert issubclass(Neo4jAdapter, IGraphDb)


_assert_protocol()
