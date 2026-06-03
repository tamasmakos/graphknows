"""ISessionDb — Protocol for Memgraph-style session graph database operations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ISessionDb(Protocol):
    """Protocol for session-scoped entity graph operations (Memgraph boundary)."""

    def close(self) -> None:
        """Close the Bolt driver connection."""
        ...

    async def ensure_indexes(self) -> None:
        """Create STM indexes idempotently."""
        ...

    async def write_session_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a SESSION node with arbitrary metadata properties."""
        ...

    async def write_file_node(
        self,
        session_id: str,
        file_id: str,
        title: str,
        parser_name: str,
        source_hint: str,
    ) -> None:
        """Upsert a FILE node linked to a SESSION."""
        ...

    async def write_chunk_node(
        self,
        session_id: str,
        file_id: str,
        chunk_id: str,
        text: str,
        heading_path: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a CHUNK node linked to its FILE."""
        ...

    async def write_entity_node(
        self,
        session_id: str,
        entity_name: str,
        entity_type: str | None,
        confidence: float,
    ) -> None:
        """Upsert an ENTITY node scoped to a session."""
        ...

    async def write_chunk_entity_edge(
        self,
        chunk_id: str,
        entity_name: str,
        session_id: str,
    ) -> None:
        """Create a MENTIONS edge from CHUNK to ENTITY."""
        ...

    async def write_dialog_node(
        self,
        session_id: str,
        dialog_id: str,
        started_at: str,
    ) -> None:
        """Upsert a DIALOG node linked to a SESSION."""
        ...

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
        """Upsert a TURN node linked to its DIALOG."""
        ...

    async def write_turn_sequence(self, prev_turn_id: str, next_turn_id: str) -> None:
        """Create a NEXT edge between consecutive TURNs."""
        ...

    async def write_turn_entity_edge(
        self,
        turn_id: str,
        entity_name: str,
        session_id: str,
        confidence: float,
    ) -> None:
        """Create a MENTIONS edge from TURN to ENTITY."""
        ...

    async def query_entity_chunks(
        self,
        session_id: str,
        entity_names: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return CHUNK nodes linked to any of the given entity names."""
        ...

    async def delete_session(self, session_id: str) -> None:
        """Delete all nodes and edges for session_id."""
        ...

    async def get_session_nodes(self, session_id: str) -> list[dict[str, Any]]:
        """Return all entity nodes for session_id as plain dicts."""
        ...

    async def get_session_edges(self, session_id: str) -> list[dict[str, Any]]:
        """Return all RELATES edges for session_id as plain dicts."""
        ...

    async def get_session_graph(self, session_id: str) -> dict[str, Any]:
        """Return nodes and edges for session_id as a graph dict."""
        ...

    async def read_session_subgraph(self, session_id: str) -> dict[str, Any]:
        """Read the full STM subgraph for session_id from Memgraph."""
        ...

    async def get_session_node_count(self, session_id: str) -> int:
        """Return the number of Entity nodes stored for session_id."""
        ...

    async def check_chain_integrity(self, session_id: str) -> dict[str, Any]:
        """Check SESSION→FILE→CHUNK chain integrity for session_id."""
        ...

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
        """Write a typed REL edge between two named entities in Memgraph STM."""
        ...
