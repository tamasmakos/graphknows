"""ISTMStore — Protocol for the short-term-memory graph store.

A backend implements this to serve session-scoped evidence: files, chunks,
entities, dialogs, turns, and bi-temporal relation edges. The bundled
``ArcadeDBSTMStore`` is one implementation; a third party can plug another by
satisfying this contract and handing it to ``Memory(stores=...)``.

Only methods consumed outside ``graphknows.memory.stores`` are part of the
contract. ``runtime_checkable`` verifies method presence (not signatures), so
conformance tests assert ``isinstance(store, ISTMStore)``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ISTMStore(Protocol):
    """Session-scoped graph store (short-term memory)."""

    # --- lifecycle ---
    async def connect(self) -> None:
        """Open the store connection."""
        ...

    async def close(self) -> None:
        """Close the store connection."""
        ...

    async def ensure_schema(self) -> None:
        """Create vertex/edge types and indexes idempotently."""
        ...

    # --- writes ---
    async def write_session_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Upsert the SESSION node with arbitrary metadata."""
        ...

    async def write_file_node(
        self, session_id: str, file_id: str, title: str, parser_name: str, source_hint: str
    ) -> None:
        """Upsert a FILE node linked to its session."""
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
        """Upsert a CHUNK node linked to its file."""
        ...

    async def write_entity_node(
        self, session_id: str, entity_name: str, entity_type: str | None, confidence: float
    ) -> None:
        """Upsert a session-scoped ENTITY node."""
        ...

    async def write_chunk_entities_bulk(
        self, *, session_id: str, chunk_id: str, entities: list[dict[str, Any]]
    ) -> None:
        """Upsert entity nodes + CHUNK→ENTITY edges for one chunk in bulk."""
        ...

    async def write_dialog_node(self, session_id: str, dialog_id: str, started_at: str) -> None:
        """Upsert a DIALOG node linked to its session."""
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
        """Upsert a TURN node linked to its dialog."""
        ...

    async def write_turn_sequence(self, prev_turn_id: str, next_turn_id: str) -> None:
        """Create a NEXT edge between consecutive turns."""
        ...

    async def write_turn_entity_edge(
        self, turn_id: str, entity_name: str, session_id: str, confidence: float
    ) -> None:
        """Create a MENTIONS edge from a turn to an entity."""
        ...

    async def write_relation_edge(
        self,
        session_id: str,
        subject_name: str,
        relation_name: str,
        object_name: str,
        fact: str = ...,
        confidence: float = ...,
        valid_at: Any | None = ...,
        created_at: Any | None = ...,
        chunk_id: str = ...,
    ) -> None:
        """Write a bi-temporal relation edge between two named entities."""
        ...

    # --- reads ---
    async def query_entity_chunks(
        self, session_id: str, entity_names: list[str], top_k: int = ...
    ) -> list[dict[str, Any]]:
        """Return chunks reachable from the given entity names."""
        ...

    async def full_text_search_chunks(
        self, session_id: str, terms: list[str], top_k: int = ..., require_all: bool = ...
    ) -> list[dict[str, Any]]:
        """Return chunks whose text contains the given terms."""
        ...

    async def read_session_subgraph(self, session_id: str) -> dict[str, Any]:
        """Read the full STM subgraph for a session (used by flush)."""
        ...

    async def get_session_graph(self, session_id: str) -> dict[str, Any]:
        """Return the session's nodes and edges as a graph dict."""
        ...

    async def delete_session(self, session_id: str) -> None:
        """Delete all nodes for a session."""
        ...
