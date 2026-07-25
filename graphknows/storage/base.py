"""Storage port interfaces (Protocols) for pluggable persistence.

Dependency-free by design: domain code depends on these abstractions, and the
concrete adapters in :mod:`graphknows.storage.arcadedb` implement them.

**Implementing a backend.** graphknows ships one storage backend (ArcadeDB).
The contract that matters is :class:`IGraphStore`: the golden layer keeps ONE
graph database per namespace, and ingestion, retrieval, and :class:`~graphknows.memory.Memory`
talk to nothing else. Implement it for your datastore to plug in an alternative.
Embedding is swappable via :class:`IEmbedder`.

The older two-database (STM/LTM) split has been fully retired: its store
classes and protocols are deleted, and everything now talks to the single
per-namespace :class:`IGraphStore`.
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
class IGraphStore(Protocol):
    """The single per-namespace graph store the live system runs on.

    One database holds sessions, files, chunks, entities, relations, temporal
    nodes, frames, and topics; lifecycle ``state`` (raw -> consolidated) replaces
    the old STM/LTM database split. Method set mirrors
    :class:`~graphknows.storage.arcadedb.graph_store.GraphStore`.
    """

    # --- lifecycle ---
    async def connect(self) -> None:
        """Open the store connection."""
        ...

    async def close(self) -> None:
        """Close the store connection."""
        ...

    async def ensure_schema(self, dims: int) -> None:
        """Create the database and graph schema idempotently."""
        ...

    async def drop_database(self) -> None:
        """Drop this store's entire database (destructive)."""
        ...

    async def query(self, cypher: str, **params: Any) -> list[dict]:
        """Run a read-only statement against this store's database."""
        ...

    async def command(self, cypher: str, **params: Any) -> list[dict]:
        """Run a writing statement against this store's database."""
        ...

    # --- writes ---
    async def ensure_session(self, session_id: str, started_at: str = ...) -> None:
        """Upsert a SESSION vertex; state starts ``raw``."""
        ...

    async def ensure_file(
        self, file_id: str, session_id: str, name: str = ..., media_type: str = ...
    ) -> None:
        """Upsert a FILE vertex linked to its session."""
        ...

    async def write_chunk(self, chunk: dict[str, Any]) -> None:
        """Upsert one CHUNK vertex with its embedding."""
        ...

    async def write_chunks_bulk(self, chunks: list[dict[str, Any]]) -> None:
        """Upsert CHUNK vertices in batches."""
        ...

    async def write_entities_bulk(
        self, session_id: str, chunk_id: str, entities: list[dict[str, Any]]
    ) -> None:
        """Upsert ENTITY vertices + MENTIONS edges for one chunk."""
        ...

    async def write_relation(
        self,
        src_name: str,
        predicate: str,
        dst_name: str,
        fact: str = ...,
        chunk_id: str = ...,
        confidence: float = ...,
        valid_at: str | None = ...,
        source: str = ...,
        src_type: str = ...,
        dst_type: str = ...,
    ) -> None:
        """Merge a typed entity->entity relation edge."""
        ...

    async def write_temporal(self, chunk_id: str, iso_dates: list[str]) -> None:
        """Merge TEMPORAL nodes + MENTIONS_DATE edges for a chunk."""
        ...

    async def write_frames(self, session_id: str, frames: list[dict[str, Any]]) -> int:
        """Bulk-write FRAME vertices + EVOKED_BY evidence edges. Returns count."""
        ...

    async def write_frame_instances(
        self, session_id: str, chunk_id: str, instances: list[dict[str, Any]]
    ) -> int:
        """Merge role-filled FRAME_INSTANCE vertices + PLAYS_ROLE edges."""
        ...

    async def write_topics(self, topics: list[dict[str, Any]]) -> int:
        """Upsert TOPIC vertices + HAS_TOPIC edges. Returns count."""
        ...

    # --- reads / retrieval collectors ---
    async def search_chunks_ann(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = ...,
        state: str | None = ...,
    ) -> list[dict[str, Any]]:
        """ANN search over CHUNK embeddings."""
        ...

    async def query_entity_chunks(
        self,
        entity_names: list[str],
        session_id: str | None = ...,
        top_k: int = ...,
        state: str | None = ...,
    ) -> list[dict[str, Any]]:
        """Chunks linked (via MENTIONS) to any of the given entity names."""
        ...

    async def full_text_search_chunks(
        self,
        terms: list[str],
        session_id: str | None = ...,
        top_k: int = ...,
        require_all: bool = ...,
        state: str | None = ...,
    ) -> list[dict[str, Any]]:
        """Chunks whose text contains the given terms."""
        ...

    async def query_temporal_chunks(
        self,
        iso_dates: list[str],
        session_id: str | None = ...,
        top_k: int = ...,
        state: str | None = ...,
    ) -> list[dict[str, Any]]:
        """Chunks linked to TEMPORAL nodes matching the given ISO dates."""
        ...

    async def query_frame_chunks(
        self, frame_types: list[str], session_id: str | None = ..., state: str | None = ...
    ) -> list[dict[str, Any]]:
        """FRAME rows (with evidence chunk) matching the given frame types."""
        ...

    async def query_frames_by_embedding(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = ...,
        state: str | None = ...,
    ) -> list[dict[str, Any]]:
        """ANN over FRAME embeddings, joined to their evidence chunks."""
        ...

    async def search_topics(self, embedding: list[float], top_k: int = ...) -> list[dict[str, Any]]:
        """ANN search over TOPIC embeddings."""
        ...

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch full chunk records keyed by ID."""
        ...

    async def resolve_query_entities(self, words: list[str], cap: int = ...) -> list[str]:
        """Resolve query tokens to ENTITY surface names."""
        ...

    async def entity_fact_context(self, entity_names: list[str], max_facts: int = ...) -> list[str]:
        """Compact structured fact-sheet for the given entities."""
        ...

    async def entity_relation_graph(self, session_id: str | None = ...) -> list[dict[str, Any]]:
        """Semantic entity->entity relation rows."""
        ...

    async def list_entities(self) -> list[dict[str, Any]]:
        """List every ENTITY with its incoming-MENTIONS count."""
        ...

    async def distinct_speakers(self) -> list[str]:
        """Casefolded distinct chunk speakers."""
        ...

    async def similar_entities_by_embedding(
        self, name: str, top_k: int = ...
    ) -> list[dict[str, Any]]:
        """Entities most similar to *name* by graph-embedding cosine."""
        ...

    # --- consolidation / analytics ---
    async def consolidate_session(self, session_id: str) -> int:
        """Flip a session's lifecycle state raw -> consolidated."""
        ...

    async def merge_entities(self, winner_norm: str, loser_norm: str) -> bool:
        """Merge the loser ENTITY into the winner. Idempotent."""
        ...

    async def drop_entities(self, name_norms: list[str]) -> int:
        """Delete the given ENTITY nodes and their edges. Returns count."""
        ...

    async def compute_and_persist_pagerank(self, skip_view_ensure: bool = ...) -> int:
        """Compute PageRank over the entity graph and persist it."""
        ...

    async def compute_graph_embeddings(
        self, dimensions: int = ..., skip_view_ensure: bool = ...
    ) -> int:
        """Compute node2vec embeddings over the entity graph and persist them."""
        ...

    async def compute_communities(self, skip_view_ensure: bool = ...) -> int:
        """Detect entity communities and persist ``community_id``."""
        ...

    # --- deletion ---
    async def delete_session(self, session_id: str) -> None:
        """Delete a session's frames, chunks, files, and SESSION node."""
        ...

    async def clear_all(self) -> None:
        """Delete every vertex and edge in this database."""
        ...


__all__ = ["IEmbedder", "IGraphStore"]
