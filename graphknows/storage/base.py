"""Storage port interfaces (Protocols) for pluggable persistence.

Dependency-free by design: domain code depends on these abstractions, and the
concrete adapters in :mod:`graphknows.storage.arcadedb` implement them.

**Implementing a backend.** graphknows ships one storage backend (ArcadeDB),
but the store contract is three focused protocols — :class:`ISTMStore`
(session-scoped evidence graph), :class:`IVectorStore` (embeddings + STM/LTM
vector search), and :class:`ILTMStore` (consolidated corpus graph). Implement
all three for your datastore to plug in an alternative. Embedding is swappable
via :class:`IEmbedder`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IEmbedder(Protocol):
    """Protocol for text-to-vector embedding operations."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one float vector per text."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string and return a plain float list."""
        ...


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
        """Upsert entity nodes + CHUNK->ENTITY edges for one chunk in bulk."""
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


@runtime_checkable
class IVectorStore(Protocol):
    """Protocol for vector database operations."""

    async def connect(self) -> None:
        """Open the database connection."""
        ...

    async def close(self) -> None:
        """Drain and close the connection pool."""
        ...

    async def ensure_schema(self) -> None:
        """Create tables and indexes idempotently."""
        ...

    async def upsert_entities(self, entities: list[dict[str, Any]]) -> None:
        """Upsert corpus-level entity embeddings."""
        ...

    async def search_chunks(
        self, query_emb: list[float], session_id: str, k: int = 20
    ) -> list[dict[str, Any]]:
        """ANN search on chunks filtered to session_id."""
        ...

    async def search_entities(
        self, query_emb: list[float], k: int = 10, entity_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """ANN search on entity embeddings (corpus-level)."""
        ...

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
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
        self, session_id: str, embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """ANN search on stm_chunks filtered to session_id."""
        ...

    async def search_stm_turns(
        self, session_id: str, embedding: list[float], top_k: int
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
        self, embedding: list[float], top_k: int, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """ANN search on ltm_chunks (corpus-wide, optionally filtered by session)."""
        ...

    async def search_ltm_turns(
        self, embedding: list[float], top_k: int, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """ANN search on ltm_turns (interaction/dialog recall)."""
        ...

    async def write_ltm_topic(self, topic: dict[str, Any]) -> None:
        """UPSERT a topic into ltm_topics."""
        ...

    async def write_community_assignments(
        self, session_id: str, assignments: list[dict[str, Any]]
    ) -> None:
        """Write community metadata onto semantic memory rows."""
        ...


@runtime_checkable
class ILTMStore(Protocol):
    """Consolidated corpus graph store (long-term memory)."""

    # --- lifecycle ---
    async def connect(self) -> None:
        """Open the store connection."""
        ...

    async def close(self) -> None:
        """Close the store connection."""
        ...

    async def ensure_schema(self) -> None:
        """Create the LTM database, types, and indexes idempotently."""
        ...

    # --- promotion / writes ---
    async def write_session_subgraph(
        self, session_id: str, stm_data: dict[str, Any]
    ) -> dict[str, int]:
        """MERGE a full STM subgraph into LTM. Returns node/edge counts."""
        ...

    async def extract_and_write_relations(
        self,
        stm_data: dict[str, Any],
        llm_api_key: str = ...,
        llm_model: str = ...,
        session_id: str = ...,
    ) -> int:
        """LLM-assisted relation extraction into LTM. Returns edge count."""
        ...

    async def upsert_topics(self, topics: list[dict[str, Any]]) -> None:
        """Upsert TOPIC nodes."""
        ...

    async def write_frames(self, session_id: str, frames: list[dict[str, Any]]) -> int:
        """Bulk-write FRAME instances for a session. Returns count."""
        ...

    # --- analytics ---
    async def compute_and_persist_pagerank(self) -> int:
        """Compute PageRank over the entity graph and persist it."""
        ...

    async def persist_centrality(self, centrality: dict[str, dict[str, float]]) -> int:
        """Persist precomputed centrality scores onto ENTITY nodes."""
        ...

    async def detect_communities(self) -> Any:
        """Detect entity communities and persist community_id."""
        ...

    async def write_topic_summaries(
        self,
        session_id: str,
        summarizer: Any = ...,
        vec_store: Any = ...,
        gds_modularity: float | None = ...,
        community_algorithm: str | None = ...,
    ) -> int:
        """Summarise communities into TOPIC nodes. Returns count created."""
        ...

    async def prune_tombstoned_edges(self, retention_days: int = ...) -> int:
        """Delete tombstoned relation edges past the retention window."""
        ...

    async def prune_duplicate_edges(self) -> int:
        """Delete low-confidence duplicate tombstoned edges."""
        ...

    # --- reads ---
    async def full_text_search_chunks(
        self,
        terms: list[str],
        session_id: str | None = ...,
        top_k: int = ...,
        require_all: bool = ...,
    ) -> list[dict[str, Any]]:
        """Return LTM chunks whose text contains the given terms."""
        ...

    async def query_frame_chunks(
        self, session_id: str, frame_types: list[str]
    ) -> list[dict[str, Any]]:
        """Return FRAME rows for a session matching the given frame types."""
        ...


__all__ = ["IEmbedder", "ILTMStore", "ISTMStore", "IVectorStore"]
