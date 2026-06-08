"""IGraphDb — Protocol for Neo4j-style graph database operations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IGraphDb(Protocol):
    """Protocol for corpus + session graph database operations (Neo4j boundary)."""

    async def ensure_indexes(self) -> None:
        """Create corpus + session indexes idempotently."""
        ...

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Write Chunk nodes (idempotent MERGE)."""
        ...

    async def upsert_entities_and_mentions(self, chunks: list[dict[str, Any]]) -> None:
        """Write Entity nodes and MENTIONS edges derived from chunk entity lists."""
        ...

    async def upsert_topics(self, topics: list[dict[str, Any]]) -> None:
        """Write Topic nodes and IN_COMMUNITY edges (idempotent MERGE)."""
        ...

    async def upsert_co_occurs(self, cooccurrences: list[tuple[str, str, int]]) -> None:
        """Write CO_OCCURS edges between entity pairs (idempotent MERGE)."""
        ...

    async def upsert_session_entities(
        self,
        session_id: str,
        entities: list[Any],
        relations: list[Any],
    ) -> None:
        """Write SessionEntity nodes and RELATES edges for a live agent session."""
        ...

    async def seed_session_from_corpus(self, session_id: str, entity_names: list[str]) -> int:
        """Copy named Entity nodes from corpus into session namespace."""
        ...

    async def merge_session_entities_to_corpus(
        self,
        session_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """Promote session nodes and edges into the persistent corpus graph."""
        ...

    async def preview_session_merge(
        self,
        session_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a preview of what merge_session_entities_to_corpus would do."""
        ...

    async def get_chunk_entities(self, chunk_id: str) -> list[dict[str, Any]]:
        """Return Entity nodes linked to a Chunk via MENTIONS edges."""
        ...

    async def get_entity_neighbors(
        self,
        entity_name: str,
        depth: int = 1,
        relationship_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return neighboring Entity/Topic nodes reachable from entity_name."""
        ...

    async def get_chunks_for_entity(self, entity_name: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return Chunk nodes linked to entity_name via MENTIONS edges."""
        ...

    async def get_entity_community(self, entity_name: str) -> dict[str, Any] | None:
        """Return the Topic node linked to entity_name via IN_COMMUNITY edge."""
        ...

    async def write_session_subgraph(
        self, session_id: str, stm_data: dict[str, Any]
    ) -> dict[str, int]:
        """MERGE a full STM subgraph into Neo4j LTM."""
        ...

    async def count_nodes(self) -> dict[str, int]:
        """Return counts of key node types."""
        ...

    async def get_segment_info(self, segment_id: str) -> dict[str, Any] | None:
        """Return provenance + statistics for a SEGMENT node."""
        ...

    async def list_segments(self, limit: int = 20) -> list[dict[str, Any]]:
        """List SEGMENT nodes ordered by ingestion time descending."""
        ...
