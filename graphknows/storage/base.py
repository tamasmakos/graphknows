"""Storage port interfaces (Protocols) for pluggable persistence.

Dependency-free by design: domain code depends on these abstractions, and the
concrete adapters in :mod:`graphknows.storage.arcadedb` implement them.

**Implementing a backend.** graphknows ships one storage backend (ArcadeDB).
The contract that matters is :class:`IGraphStore`: the golden layer keeps ONE
graph database per namespace, and ingestion, retrieval, and :class:`~graphknows.memory.Memory`
talk to nothing else. Implement it for your datastore to plug in an alternative.
Embedding is swappable via :class:`IEmbedder`.

:class:`ISTMStore` and :class:`ILTMStore` describe the older two-database split.
That path is retired for the golden layer — implementing them does NOT make a
backend usable by ``Memory``; they are documented here only because
``ArcadeDBSTMStore``/``ArcadeDBLTMStore`` still ship to back the read-only
browsing views in the web server and the ``ltm_*`` helpers on ``Memory``.
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

    async def chunk_community_weights(self, chunk_ids: list[str]) -> dict[str, float]:
        """Quality-adjusted community boost weight per chunk."""
        ...

    async def resolve_query_entities(self, words: list[str], cap: int = ...) -> list[str]:
        """Resolve query tokens to ENTITY surface names."""
        ...

    async def entity_fact_context(
        self, entity_names: list[str], max_facts: int = ...
    ) -> list[str]:
        """Compact structured fact-sheet for the given entities."""
        ...

    async def entity_relation_graph(
        self, session_id: str | None = ...
    ) -> list[dict[str, Any]]:
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

    async def compute_community_quality(self, skip_view_ensure: bool = ...) -> dict[str, Any]:
        """Score community structure (modularity + conductance)."""
        ...

    # --- deletion ---
    async def delete_session(self, session_id: str) -> None:
        """Delete a session's frames, chunks, files, and SESSION node."""
        ...

    async def clear_all(self) -> None:
        """Delete every vertex and edge in this database."""
        ...


@runtime_checkable
class ISTMStore(Protocol):
    """Session-scoped graph store of the retired three-database split."""

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
class ILTMStore(Protocol):
    """Consolidated corpus graph store of the retired three-database split."""

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


__all__ = ["IEmbedder", "IGraphStore", "ILTMStore", "ISTMStore"]
