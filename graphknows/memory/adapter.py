"""Host-agnostic facade over the graphknows memory backend.

Composes STMService, LTMPromotionPipeline, and recall functions behind a stable
generic contract. Host-specific parsing and normalization stay outside this class.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from graphknows.memory.adapter_types import (
    AdapterCapabilities,
    ArtifactRecord,
    PromoteOptions,
    RecallRequest,
    TurnRecord,
)
from graphknows.memory.flush import LTMPromotionPipeline
from graphknows.memory.models.ingest import IngestResult
from graphknows.memory.models.ingest_options import IngestOptions
from graphknows.memory.models.message import MessageInput, normalize_messages
from graphknows.memory.models.promotion import PromotionRequest
from graphknows.memory.models.scope import MemoryScope
from graphknows.memory.stm import STMService
from graphknows.memory.stm.dialog import DialogCapture as DialogCapture
from graphknows.memory.stm.dialog import Turn


def _build_stm_service(
    mg_store: Any,
    pg_store: Any,
    extractor: Any,
    embed_fn: Any | None = None,
    rel_extractor: Any | None = None,
) -> STMService:
    """Factory helper — construct STMService from raw stores.

    Args:
        mg_store: ArcadeDBSTMStore instance.
        pg_store: ArcadeDBVectorStore instance.
        extractor: Entity extractor.
        embed_fn: Optional embedding callable.
        rel_extractor: Optional relation extractor.

    Returns:
        Fully configured STMService.
    """
    return STMService(
        mg_store=mg_store,
        pg_store=pg_store,
        extractor=extractor,
        embed_fn=embed_fn,
        rel_extractor=rel_extractor,
    )


@dataclass
class GraphknowsAdapter:
    """Host-agnostic facade over the graphknows memory backend.

    Composes STMService, LTMPromotionPipeline, and recall functions behind a
    stable generic contract. Host-specific parsing and normalization stay
    outside this class.

    Args:
        mg_store: ArcadeDBSTMStore instance.
        pg_store: ArcadeDBVectorStore instance.
        neo_store: ArcadeDBLTMStore instance.
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
    _stm_service: STMService | None = field(default=None, repr=False)
    _dialog_writer: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize STMService."""
        self._stm_service = _build_stm_service(
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
            from graphknows.memory.stm._async_bridge import _run

            _run(self.mg_store.write_session_metadata(session_id, metadata))

    def write_turn(self, turn: TurnRecord) -> None:
        """Write a single conversation turn into the interaction graph.

        Args:
            turn: Turn data including text and role.
        """
        dialog_id = self._active_dialog_id or ""
        mapped_turn = Turn(
            id=turn.turn_id,
            index=0,
            role=turn.role,
            text=turn.content,
            timestamp=time.time(),
        )
        extra_metadata = {"memory_class": turn.memory_class.value, **turn.metadata}
        dw = getattr(self, "_dialog_writer", None) or DialogCapture(  # type: ignore[no-untyped-call]
            mg_store=self.mg_store,
            pg_store=self.pg_store,
            extractor=self.extractor,
            embed_fn=self.embed_fn,
        )
        dw.write_turn(turn.session_id, dialog_id, mapped_turn, extra_metadata=extra_metadata)

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
        if self._stm_service is None:
            raise RuntimeError("GraphknowsAdapter is not initialized")
        self._stm_service.ingest(
            artifact.content,
            session_id=session_id,
            title=artifact.path,
            source_hint=artifact.mime_type,
            metadata=metadata,
        )

    def recall(self, request: RecallRequest) -> list[dict[str, Any]]:
        """Recall memories from short-term memory via DETRetriever.

        Args:
            request: Recall query parameters.

        Returns:
            List of recall hits as dictionaries.
        """
        from graphknows.memory.query.retriever import DETRetriever
        from graphknows.memory.stm._async_bridge import _run

        retriever = DETRetriever(pg=self.pg_store, neo=self.neo_store, mg=self.mg_store)
        ctx = _run(
            retriever.retrieve(
                request.query,
                session_id=request.session_id or "",
                top_k=request.top_k,
                scope="stm",
            )
        )
        hit_dicts = [
            {
                "text": r.get("text", ""),
                "source_type": "chunk",
                "scope": "short_term",
                "score": r.get("rrf_score", 0.0),
                "cosine_score": r.get("rrf_score", 0.0),
                "entity_names": r.get("entities", []),
                "heading_path": "",
                "provenance": {
                    "session_id": request.session_id or "",
                    "memory_class": r.get("metadata", {}).get("memory_class", "raw_evidence"),
                },
                "metadata": r.get("metadata", {}),
            }
            for r in ctx.fused_results
        ]
        if not request.include_raw:
            hit_dicts = [
                h
                for h in hit_dicts
                if h.get("provenance", {}).get("memory_class") != "raw_evidence"
            ]
        return hit_dicts

    async def promote(self, options: PromoteOptions) -> dict[str, Any]:
        """Promote short-term memory to long-term memory.

        Args:
            options: Promotion configuration.

        Returns:
            Summary dictionary with promotion results.
        """
        pipeline = LTMPromotionPipeline(
            self.mg_store,
            self.pg_store,
            self.neo_store,
        )
        result = await pipeline.promote(
            PromotionRequest(
                session_id=options.session_id,
                run_gds=options.run_gds,
            )
        )
        return {
            "session_id": options.session_id,
            "run_gds": options.run_gds,
            "topic_mode": result.topic_mode,
            "nodes_written": result.nodes_written,
            "edges_written": result.edges_written,
            "chunks_promoted": result.chunks_promoted,
            "turns_promoted": result.turns_promoted,
            "topics_created": result.topics_created,
            "merged_entities": result.merged_entities,
            "elapsed_s": result.elapsed_s,
            "errors": result.errors,
        }

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
# IngestResult is defined in graphknows.memory.models.ingest (imported above).


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
        messages: MessageInput,
        options: IngestOptions,
    ) -> IngestResult:
        """Ingest messages into STM.

        messages accepts str, dict, or list[str | dict | Message].
        All inputs are normalized via normalize_messages() before storage.
        """
        ...

    async def recall(
        self,
        query: str,
        scope: MemoryScope,
        top_k: int = 5,
        recall_scope: str = "stm",
    ) -> list[RecallHit]:
        """Recall relevant chunks from memory for the given query."""
        ...

    async def flush(self, scope: MemoryScope) -> FlushResult:
        """Flush a session from STM to LTM."""
        ...

    async def stats(self, scope: MemoryScope | None = None) -> MemoryStats:
        """Return node/chunk counts for STM and LTM."""
        ...


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class GraphknowsMemoryAdapter:
    """Concrete AgentMemoryAdapter backed by the Graphknows memory stack."""

    def __init__(self, container: Any = None) -> None:
        self._container = container
        self._svc_cache: tuple[Any, Any, STMService] | None = None

    # ------------------------------------------------------------------
    # Lazy helpers
    # ------------------------------------------------------------------

    def _stores_from_container(self) -> tuple[Any, Any, Any]:
        c = self._container
        return c.stm, c.vector, c.ltm  # type: ignore[no-any-return]

    def _lazy_stores(self) -> tuple[Any, Any, Any, Any]:
        from graphknows.memory.stm._async_bridge import _run
        from graphknows.memory.stores.arcadedb.client import ArcadeDBClient
        from graphknows.memory.stores.arcadedb.ltm_store import ArcadeDBLTMStore
        from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore
        from graphknows.memory.stores.arcadedb.vector_store import ArcadeDBVectorStore
        from graphknows.memory.stores.embedder import embed_one

        arcade_client = ArcadeDBClient(
            base_url=os.environ.get("ARCADEDB_URL", "http://172.18.0.1:2480"),
            user=os.environ.get("ARCADEDB_USER", "root"),
            password=os.environ.get("ARCADEDB_PASSWORD", "changeme"),
        )
        _run(arcade_client.connect())
        mg = ArcadeDBSTMStore(arcade_client)
        _run(mg.ensure_schema())

        vec_client = ArcadeDBClient(
            base_url=os.environ.get("ARCADEDB_URL", "http://172.18.0.1:2480"),
            user=os.environ.get("ARCADEDB_USER", "root"),
            password=os.environ.get("ARCADEDB_PASSWORD", "changeme"),
        )
        _run(vec_client.connect())
        pg = ArcadeDBVectorStore(vec_client)
        _run(pg.ensure_schema())

        ltm_client = ArcadeDBClient(
            base_url=os.environ.get("ARCADEDB_URL", "http://172.18.0.1:2480"),
            user=os.environ.get("ARCADEDB_USER", "root"),
            password=os.environ.get("ARCADEDB_PASSWORD", "changeme"),
        )
        _run(ltm_client.connect())
        neo = ArcadeDBLTMStore(ltm_client)
        _run(neo.ensure_schema())

        return mg, pg, neo, embed_one

    def _get_extractor(self) -> Any:
        from graphknows.memory.extractor import GLiNER2EntityExtractor

        return GLiNER2EntityExtractor()

    def _build_svc(self) -> tuple[Any, Any, STMService]:
        if self._svc_cache is not None:
            return self._svc_cache

        from graphknows.memory.stm._async_bridge import _run as _stm_run

        if self._container is not None:
            mg, pg, _ = self._stores_from_container()
            from graphknows.memory.stores.embedder import embed_one

            embed_fn = embed_one
        else:
            mg, pg, _, embed_fn = self._lazy_stores()
        extractor = self._get_extractor()
        _stm_run(pg.connect())
        self._svc_cache = mg, pg, STMService(mg, pg, extractor, embed_fn=embed_fn)
        return self._svc_cache

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------

    async def ingest(
        self,
        messages: MessageInput,
        options: IngestOptions,
    ) -> IngestResult:
        """Ingest messages into STM.

        messages accepts str, dict, or list[str | dict | Message].
        options carries scope (user_id/agent_id/run_id), infer flag, and metadata.
        """
        _, _, svc = self._build_svc()
        msgs = normalize_messages(messages)
        return await asyncio.to_thread(svc.ingest_messages, msgs, options)

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        scope: MemoryScope | str,
        top_k: int = 5,
        recall_scope: str = "stm",
    ) -> list[RecallHit]:
        """Recall relevant chunks from STM, LTM, or both via DETRetriever."""
        from graphknows.memory.query.retriever import DETRetriever

        session_id = scope.to_session_id() if isinstance(scope, MemoryScope) else scope

        if self._container is not None:
            mg, pg, neo = self._stores_from_container()
            embed_model = getattr(
                getattr(self._container, "settings", None),
                "embed_model",
                "openai/text-embedding-3-small",
            )
            pg_owned = False
        else:
            mg, pg, neo, _ = self._lazy_stores()
            await pg.connect()
            embed_model = "openai/text-embedding-3-small"
            pg_owned = True

        try:
            retriever = DETRetriever(pg=pg, neo=neo, mg=mg, embed_model=embed_model)
            ctx = await retriever.retrieve(
                query, session_id=session_id, top_k=top_k, scope=recall_scope
            )
        finally:
            if pg_owned:
                await pg.close()
                neo.close()

        return [
            RecallHit(
                score=r.get("rrf_score", 0.0),
                text=r.get("text", ""),
                session_id=session_id,
                entities=r.get("entities", []),
                source_type=r.get("sources", ""),
            )
            for r in ctx.fused_results
        ]

    # ------------------------------------------------------------------
    # flush
    # ------------------------------------------------------------------

    async def flush(self, scope: MemoryScope | str) -> FlushResult:
        """Flush a session from STM to LTM."""
        if self._container is not None:
            mg, pg, neo = self._stores_from_container()
        else:
            mg, pg, neo, _ = self._lazy_stores()
            await asyncio.to_thread(pg.connect)

        from graphknows.memory.flush import LTMPromotionPipeline
        from graphknows.memory.models.promotion import PromotionRequest

        session_id = scope.to_session_id() if isinstance(scope, MemoryScope) else scope
        pipeline = LTMPromotionPipeline(mg, pg, neo)
        internal = await pipeline.promote(PromotionRequest(session_id=session_id))
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

    async def stats(self, scope: MemoryScope | str | None = None) -> MemoryStats:
        """Return aggregate node/chunk counts for STM and LTM via ArcadeDB."""
        if self._container is not None:
            _, pg, neo = self._stores_from_container()
        else:
            _, pg, neo, _ = self._lazy_stores()

        stm: dict[str, Any] = {}
        ltm: dict[str, Any] = {}

        async def _count(store: Any, db: str, label: str) -> int:
            rows = await store._c.query(db, f"MATCH (n:{label}) RETURN count(n) AS c")
            return int(rows[0].get("c") or 0) if rows else 0

        stm["chunks"] = await _count(pg, pg._db if hasattr(pg, "_db") else "stm", "CHUNK")
        stm["turns"] = await _count(pg, pg._db if hasattr(pg, "_db") else "stm", "TURN")
        ltm["chunks"] = await _count(neo, "ltm", "CHUNK")
        ltm["turns"] = await _count(neo, "ltm", "TURN")
        ltm["topics"] = await _count(neo, "ltm", "TOPIC")
        ltm["entities"] = await _count(neo, "ltm", "ENTITY")
        ltm["sessions"] = await _count(neo, "ltm", "SESSION")

        return MemoryStats(stm=stm, ltm=ltm)
