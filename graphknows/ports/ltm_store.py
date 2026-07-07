"""ILTMStore — Protocol for the long-term-memory graph store.

A backend implements this to serve the consolidated corpus graph: the promoted
session subgraph, entity centrality, community/topic structure, graph pruning,
and the deterministic frame layer. ``ArcadeDBLTMStore`` is the bundled
implementation; a third party plugs another by satisfying this contract.

Only methods consumed outside ``graphknows.memory.stores`` are part of the
contract. ``runtime_checkable`` verifies method presence (not signatures).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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
