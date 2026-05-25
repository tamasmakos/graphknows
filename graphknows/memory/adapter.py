"""Host-agnostic facade over the graphknows memory backend.

Composes MemoryIngestor, DialogWriter, SessionFlusher, and recall
functions behind a stable generic contract. Host-specific parsing
and normalization stay outside this class.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from graphknows.memory.adapter_types import (
    AdapterCapabilities,
    ArtifactRecord,
    ParticipantRecord,
    PromoteOptions,
    RecallRequest,
    TurnRecord,
)
from graphknows.memory.dialog import DialogWriter, Participant, ToolCall, Turn
from graphknows.memory.flush import FlushOptions, SessionFlusher
from graphknows.memory.ingestor import MemoryIngestor

# recall_short_term is imported lazily inside recall() to avoid a circular
# import: recall_stm → memory.ingestor → memory.__init__ → adapter → recall_stm


@dataclass
class GraphknowsAdapter:
    """Host-agnostic facade over the graphknows memory backend.

    Composes MemoryIngestor, DialogWriter, SessionFlusher, and recall
    functions behind a stable generic contract. Host-specific parsing
    and normalization stay outside this class.

    Args:
        mg_store: Memgraph store instance.
        pg_store: PgVector store instance.
        neo_store: Neo4j store instance.
        extractor: Entity extractor (GLiNER2EntityExtractor or compatible).
        embed_fn: Optional embedding function (text -> list[float]).
    """

    mg_store: Any
    pg_store: Any
    neo_store: Any
    extractor: Any
    embed_fn: Any | None = None

    _active_session_id: str | None = field(default=None, repr=False)
    _active_dialog_id: str | None = field(default=None, repr=False)
    _ingestor: MemoryIngestor | None = field(default=None, repr=False)
    _dialog_writer: DialogWriter | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize composed ingestor and dialog writer."""
        self._ingestor = MemoryIngestor(
            self.mg_store,
            self.pg_store,
            self.extractor,
            self.embed_fn,
        )
        self._dialog_writer = DialogWriter(
            self.mg_store,
            self.pg_store,
            self.extractor,
            self.embed_fn,
        )

    def open_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record that a session has started.

        Calls any setup needed on stores. No-op if stores don't require it.
        Stores ``session_id`` internally as ``self._active_session_id``.

        Args:
            session_id: Unique session identifier.
            metadata: Optional session metadata.
        """
        self._active_session_id = session_id
        if metadata and hasattr(self.mg_store, "write_session_metadata"):
            from graphknows.memory.ingestor import _run

            _run(self.mg_store.write_session_metadata(session_id, metadata))

    def start_dialog(
        self,
        session_id: str,
        dialog_id: str,
        participants: list[ParticipantRecord],
    ) -> None:
        """Initialize a dialog and register participants.

        Args:
            session_id: Parent session namespace.
            dialog_id: Unique dialog identifier.
            participants: List of participants in the dialog.
        """
        self._active_dialog_id = dialog_id
        mapped_participants = [
            Participant(
                id=p.participant_id,
                role=p.role,
                display_name=p.display_name or "",
            )
            for p in participants
        ]
        assert self._dialog_writer is not None  # set in __post_init__
        self._dialog_writer.start_dialog(
            session_id,
            dialog_id,
            mapped_participants,
        )

    def write_turn(self, turn: TurnRecord) -> None:
        """Write a single conversation turn into the interaction graph.

        Args:
            turn: Turn data including text, role, and tool calls.
        """
        dialog_id = self._active_dialog_id or ""
        mapped_tool_calls = [
            ToolCall(
                name=tc.tool_name,
                result_summary=tc.output_summary,
            )
            for tc in turn.tool_calls
        ]
        mapped_turn = Turn(
            id=turn.turn_id,
            index=0,
            role=turn.role,
            text=turn.content,
            timestamp=time.time(),
            tool_calls=mapped_tool_calls,
        )
        assert self._dialog_writer is not None  # set in __post_init__
        self._dialog_writer.write_turn(
            turn.session_id,
            dialog_id,
            mapped_turn,
        )

    def ingest_artifact(
        self,
        artifact: ArtifactRecord,
        session_id: str,
    ) -> None:
        """Ingest an artifact into short-term memory.

        Args:
            artifact: Artifact record to ingest.
            session_id: Session namespace.
        """
        metadata = {
            **artifact.metadata,
            "memory_class": artifact.memory_class.value,
            "artifact_id": artifact.artifact_id,
        }
        assert self._ingestor is not None  # set in __post_init__
        self._ingestor.ingest_text(
            artifact.content,
            session_id=session_id,
            title=artifact.path,
            source_hint=artifact.mime_type,
            metadata=metadata,
        )

    def recall(self, request: RecallRequest) -> list[dict[str, Any]]:
        """Recall memories from short-term memory.

        Args:
            request: Recall query parameters.

        Returns:
            List of recall hits as dictionaries.
        """
        from graphknows.retrieval.recall_stm import recall_short_term  # noqa: PLC0415

        session_id: str = request.session_id or ""
        hits = recall_short_term(
            self.mg_store,
            self.pg_store,
            query=request.query,
            session_id=session_id,
            top_k=request.top_k,
            embed_fn=self.embed_fn,
        )
        hit_dicts = [asdict(hit) for hit in hits]
        if not request.include_raw:
            hit_dicts = [
                hit
                for hit in hit_dicts
                if hit.get("provenance", {}).get("memory_class") != "raw_evidence"
            ]
        return hit_dicts

    async def promote(self, options: PromoteOptions) -> dict[str, Any]:
        """Promote short-term memory to long-term memory.

        Args:
            options: Promotion configuration.

        Returns:
            Summary dictionary with promotion results.
        """
        flusher = SessionFlusher(
            self.mg_store,
            self.pg_store,
            self.neo_store,
        )
        flush_options = FlushOptions(
            summarize_topics=options.run_topics,
            run_gds=options.run_gds,
        )
        result = await flusher.flush(
            options.session_id,
            options=flush_options,
        )
        summary = {
            "session_id": options.session_id,
            "run_gds": options.run_gds,
            "run_topics": options.run_topics,
        }
        if hasattr(result, "nodes_written"):
            summary["nodes_written"] = result.nodes_written
        if hasattr(result, "edges_written"):
            summary["edges_written"] = result.edges_written
        if hasattr(result, "chunks_promoted"):
            summary["chunks_promoted"] = result.chunks_promoted
        if hasattr(result, "turns_promoted"):
            summary["turns_promoted"] = result.turns_promoted
        if hasattr(result, "topics_created"):
            summary["topics_created"] = result.topics_created
        if hasattr(result, "merged_entities"):
            summary["merged_entities"] = result.merged_entities
        if hasattr(result, "elapsed_s"):
            summary["elapsed_s"] = result.elapsed_s
        if hasattr(result, "errors"):
            summary["errors"] = result.errors
        return summary

    def capabilities(self) -> AdapterCapabilities:
        """Return the adapter capability flags.

        Returns:
            AdapterCapabilities indicating supported features.
        """
        return AdapterCapabilities(
            supports_stm=True,
            supports_ltm=True,
            supports_dialog_graph=True,
            supports_artifact_ingest=True,
            supports_recall=True,
            supports_promote=True,
        )


# ---------------------------------------------------------------------------
# ADAPT-001: Host-agnostic protocol + concrete GraphknowsMemoryAdapter
# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Result of a memory ingest operation."""

    file_id: str
    chunks: int
    entities: int
    elapsed_s: float
    errors: list[str] = field(default_factory=list)


@dataclass
class RecallHit:
    """A single hit returned by a memory recall query."""

    score: float
    text: str
    session_id: str
    entities: list[str] = field(default_factory=list)
    source_type: str = ""


@dataclass
class FlushResult:
    """Result of flushing a session from STM to LTM."""

    nodes: int
    edges: int
    chunks: int
    turns: int
    elapsed_s: float
    errors: list[str] = field(default_factory=list)


@dataclass
class MemoryStats:
    """Aggregate node/chunk counts for STM and LTM."""

    stm: dict[str, Any] = field(default_factory=dict)
    ltm: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentMemoryAdapter(Protocol):
    """Stable interface between an agent runtime and Graphknows memory.

    Implementations must be async-safe and session-scoped.
    All methods are async so they can be used in any async agent loop.
    """

    async def ingest(
        self,
        text: str,
        session_id: str,
        title: str = "",
    ) -> IngestResult:
        """Ingest text into STM for the given session."""
        ...

    async def recall(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        scope: str = "stm",
    ) -> list[RecallHit]:
        """Recall relevant chunks from memory for the given query."""
        ...

    async def flush(self, session_id: str) -> FlushResult:
        """Flush a session from STM to LTM."""
        ...

    async def stats(self, session_id: str = "") -> MemoryStats:
        """Return node/chunk counts for STM and LTM."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class GraphknowsMemoryAdapter:
    """Concrete AgentMemoryAdapter backed by the Graphknows memory stack.

    Delegates to MemoryIngestor, STM/LTM recall, SessionFlusher, and
    pgvector/Neo4j queries for stats.
    """

    def __init__(self, container: Any = None) -> None:
        self._container = container

    # ------------------------------------------------------------------
    # Lazy helpers
    # ------------------------------------------------------------------

    def _stores_from_container(self) -> tuple[Any, Any, Any]:
        c = self._container
        return c.memgraph, c.pg, c.neo  # type: ignore[no-any-return]

    def _lazy_stores(self) -> tuple[Any, Any, Any, Any]:
        import os

        from graphknows.retrieval.stores.embedder import embed_one
        from graphknows.retrieval.stores.memgraph import MemgraphStore
        from graphknows.retrieval.stores.neo4j import Neo4jStore
        from graphknows.retrieval.stores.pgvector import PgVectorStore

        mg = MemgraphStore(os.environ.get("MEMGRAPH_URI", "bolt://172.18.0.1:17687"))
        pg = PgVectorStore(
            os.environ.get(
                "PGVECTOR_DSN",
                "postgresql://postgres:***@172.18.0.1:5432/graphknows",
            )
        )
        neo = Neo4jStore(
            uri=os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "testpassword123"),
        )
        return mg, pg, neo, embed_one

    def _get_extractor(self) -> Any:
        from graphknows.memory.extractor import GLiNER2EntityExtractor

        return GLiNER2EntityExtractor()

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------

    async def ingest(
        self,
        text: str,
        session_id: str,
        title: str = "",
    ) -> IngestResult:
        """Ingest text into STM and return ingest statistics."""
        if self._container is not None:
            mg, pg, _ = self._stores_from_container()
            embed_fn = getattr(self._container.settings, "embed_model", None)
            from graphknows.retrieval.stores.embedder import embed_one

            embed_fn = embed_one
        else:
            mg, pg, _, embed_fn = self._lazy_stores()

        from graphknows.memory.ingestor import MemoryIngestor

        extractor = self._get_extractor()
        ingestor = MemoryIngestor(mg, pg, extractor, embed_fn=embed_fn)

        import asyncio

        internal = await asyncio.to_thread(
            ingestor.ingest_text,
            text,
            session_id=session_id,
            title=title,
        )
        return IngestResult(
            file_id=internal.file_id,
            chunks=internal.chunk_count,
            entities=internal.entity_count,
            elapsed_s=internal.elapsed_s,
            errors=list(internal.errors),
        )

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        scope: str = "stm",
    ) -> list[RecallHit]:
        """Recall relevant chunks from STM, LTM, or both."""
        if self._container is not None:
            mg, pg, neo = self._stores_from_container()
            from graphknows.retrieval.stores.embedder import embed_one

            embed_fn = embed_one
        else:
            mg, pg, neo, embed_fn = self._lazy_stores()

        from graphknows.retrieval.recall_stm import recall_short_term

        stm_hits: list[Any] = []
        ltm_hits: list[Any] = []

        if scope in ("stm", "both"):
            stm_hits = await asyncio.to_thread(
                recall_short_term,
                mg,
                pg,
                query=query,
                session_id=session_id,
                top_k=top_k,
                embed_fn=embed_fn,
            )

        if scope in ("ltm", "both"):
            from graphknows.retrieval.recall_ltm import recall_long_term

            ltm_hits = await asyncio.to_thread(
                recall_long_term,
                neo,
                pg,
                query,
                top_k=top_k,
                session_filter=session_id,
                embed_fn=embed_fn,
            )

        all_hits = stm_hits + ltm_hits
        result: list[RecallHit] = []
        for h in all_hits:
            result.append(
                RecallHit(
                    score=h.score,
                    text=h.text,
                    session_id=session_id,
                    entities=list(h.entity_names or []),
                    source_type=h.source_type,
                )
            )
        result.sort(key=lambda r: r.score, reverse=True)
        return result[:top_k]

    # ------------------------------------------------------------------
    # flush
    # ------------------------------------------------------------------

    async def flush(self, session_id: str) -> FlushResult:
        """Flush a session from Memgraph STM to Neo4j LTM."""
        if self._container is not None:
            mg, pg, neo = self._stores_from_container()
        else:
            mg, pg, neo, _ = self._lazy_stores()
            import asyncio as _asyncio

            await _asyncio.to_thread(pg.connect)

        from graphknows.memory.flush import SessionFlusher

        flusher = SessionFlusher(mg, pg, neo)
        internal = await flusher.flush(session_id)
        return FlushResult(
            nodes=internal.nodes_written,
            edges=internal.edges_written,
            chunks=internal.chunks_promoted,
            turns=internal.turns_promoted,
            elapsed_s=internal.elapsed_s,
            errors=list(internal.errors),
        )

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    async def stats(self, session_id: str = "") -> MemoryStats:
        """Return aggregate node/chunk counts for STM and LTM."""
        import asyncpg

        if self._container is not None:
            _, pg, neo = self._stores_from_container()
            pg_dsn = getattr(pg, "_dsn", "")
        else:
            _, pg, neo, _ = self._lazy_stores()
            pg_dsn = ""

        if not pg_dsn:
            import os

            pg_dsn = os.environ.get(
                "PGVECTOR_DSN",
                "postgresql://postgres:testpassword123@172.18.0.1:5432/graphknows",
            )

        stm: dict[str, Any] = {}
        ltm: dict[str, Any] = {}

        conn = await asyncpg.connect(pg_dsn)
        try:
            stm["chunks"] = int(await conn.fetchval("SELECT count(*) FROM stm_chunks"))
            stm["turns"] = int(await conn.fetchval("SELECT count(*) FROM stm_turns"))
            ltm["chunks"] = int(await conn.fetchval("SELECT count(*) FROM ltm_chunks"))
            ltm["turns"] = int(await conn.fetchval("SELECT count(*) FROM ltm_turns"))
            ltm["topics"] = int(await conn.fetchval("SELECT count(*) FROM ltm_topics"))
        finally:
            await conn.close()

        from neo4j import GraphDatabase

        if self._container is not None:
            uri = getattr(neo, "_uri", "")
            user = getattr(neo, "_user", "")
            pw = getattr(neo, "_password", "")
        else:
            import os

            uri = os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687")
            user = os.environ.get("NEO4J_USER", "neo4j")
            pw = os.environ.get("NEO4J_PASSWORD", "testpassword123")

        drv = GraphDatabase.driver(uri, auth=(user, pw))
        try:
            with drv.session() as s:
                row_e = s.run("MATCH (e:ENTITY) RETURN count(e) as c").single()
                row_s = s.run("MATCH (s:SESSION) RETURN count(s) as c").single()
                ltm["entities"] = int(row_e["c"]) if row_e else 0
                ltm["sessions"] = int(row_s["c"]) if row_s else 0
        finally:
            drv.close()

        return MemoryStats(stm=stm, ltm=ltm)
