# ruff: noqa: D102
"""MemgraphAdapter — thin delegating wrapper around MemgraphStore implementing ISessionDb."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.config.adapters.ports.session_db import ISessionDb

if TYPE_CHECKING:
    from graphknows.memory.stores.memgraph import MemgraphStore


class MemgraphAdapter:
    """Thin adapter delegating ISessionDb calls to a MemgraphStore instance.

    Args:
        store: Underlying MemgraphStore to delegate to.
    """

    def __init__(self, store: MemgraphStore) -> None:
        self._store = store

    def close(self) -> None:
        return self._store.close()

    async def ensure_indexes(self) -> None:
        return await self._store.ensure_indexes()

    async def write_session_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> None:
        return await self._store.write_session_metadata(session_id, metadata)

    async def write_file_node(
        self,
        session_id: str,
        file_id: str,
        title: str,
        parser_name: str,
        source_hint: str,
    ) -> None:
        return await self._store.write_file_node(
            session_id, file_id, title, parser_name, source_hint
        )

    async def write_chunk_node(
        self,
        session_id: str,
        file_id: str,
        chunk_id: str,
        text: str,
        heading_path: str,
        metadata: dict[str, Any],
    ) -> None:
        return await self._store.write_chunk_node(
            session_id, file_id, chunk_id, text, heading_path, metadata
        )

    async def write_entity_node(
        self,
        session_id: str,
        entity_name: str,
        entity_type: str | None,
        confidence: float,
    ) -> None:
        return await self._store.write_entity_node(session_id, entity_name, entity_type, confidence)

    async def write_chunk_entity_edge(
        self,
        chunk_id: str,
        entity_name: str,
        session_id: str,
    ) -> None:
        return await self._store.write_chunk_entity_edge(chunk_id, entity_name, session_id)

    async def write_dialog_node(
        self,
        session_id: str,
        dialog_id: str,
        started_at: str,
    ) -> None:
        return await self._store.write_dialog_node(session_id, dialog_id, started_at)

    async def write_turn_node(
        self,
        dialog_id: str,
        session_id: str,
        turn_id: str,
        turn_index: int,
        role: str,
        text: str,
        timestamp: str,
    ) -> None:
        return await self._store.write_turn_node(
            dialog_id, session_id, turn_id, turn_index, role, text, timestamp
        )

    async def write_turn_sequence(self, prev_turn_id: str, next_turn_id: str) -> None:
        return await self._store.write_turn_sequence(prev_turn_id, next_turn_id)

    async def write_turn_entity_edge(
        self,
        turn_id: str,
        entity_name: str,
        session_id: str,
        confidence: float,
    ) -> None:
        return await self._store.write_turn_entity_edge(
            turn_id, entity_name, session_id, confidence
        )

    async def query_entity_chunks(
        self,
        session_id: str,
        entity_names: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return await self._store.query_entity_chunks(session_id, entity_names, top_k)

    async def delete_session(self, session_id: str) -> None:
        return await self._store.delete_session(session_id)

    async def get_session_nodes(self, session_id: str) -> list[dict[str, Any]]:
        return await self._store.get_session_nodes(session_id)

    async def get_session_edges(self, session_id: str) -> list[dict[str, Any]]:
        return await self._store.get_session_edges(session_id)

    async def get_session_graph(self, session_id: str) -> dict[str, Any]:
        return await self._store.get_session_graph(session_id)

    async def read_session_subgraph(self, session_id: str) -> dict[str, Any]:
        return await self._store.read_session_subgraph(session_id)

    async def get_session_node_count(self, session_id: str) -> int:
        return await self._store.get_session_node_count(session_id)

    async def check_chain_integrity(self, session_id: str) -> dict[str, Any]:
        return await self._store.check_chain_integrity(session_id)

    async def write_relation_edge(
        self,
        session_id: str,
        subject_name: str,
        relation_name: str,
        object_name: str,
        fact: str = "",
        confidence: float = 0.5,
        valid_at: Any | None = None,
        created_at: Any | None = None,
    ) -> None:
        return await self._store.write_relation_edge(
            session_id,
            subject_name,
            relation_name,
            object_name,
            fact,
            confidence,
            valid_at,
            created_at,
        )


# Verify structural compatibility at import time
def _assert_protocol() -> None:
    assert issubclass(MemgraphAdapter, ISessionDb)


_assert_protocol()
