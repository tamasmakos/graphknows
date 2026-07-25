"""STMService: first-class service layer for memory ingestion.

Owns text ingest and conversation (turn) capture against the single per-namespace
:class:`~graphknows.storage.arcadedb.graph_store.GraphStore` (golden layer). For
retrieval, use DETRetriever.retrieve().
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from graphknows.ingestion.stm.config import STMConfig

if TYPE_CHECKING:
    from graphknows.models.ingest import IngestResult
    from graphknows.models.ingest_options import IngestOptions
    from graphknows.models.message import Message

log = logging.getLogger("graphknows.ingestion.stm.service")


class STMService:
    """First-class service layer for memory ingestion.

    Owns text ingest and conversation (turn) capture, delegating writes to
    :class:`_STMIngestHandler` over the single ``GraphStore``.

    Attributes:
        _store: GraphStore for the single per-namespace database.
        _extractor: Entity extractor exposing extract(text) -> ExtractionResult.
        _embed_fn: Embedding callable (text -> list[float]).
        _rel_extractor: Optional relation extractor.
        _embed_dims: Embedding dimension for ensure_schema on connect.
        _config: STMConfig with tuning parameters.
        _ontology_index: Optional OntologyIndex steering extraction labels.
    """

    def __init__(
        self,
        store: Any,
        extractor: Any,
        embed_fn: Any | None = None,
        rel_extractor: Any | None = None,
        embed_dims: int = 0,
        config: STMConfig | None = None,
        enable_frames: bool = False,
        ontology_index: Any | None = None,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._rel_extractor = rel_extractor
        self._embed_dims = embed_dims
        self._config = config or STMConfig()
        self._enable_frames = enable_frames
        self._ontology_index = ontology_index

        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            from graphknows.storage.embedder import embed_one

            self._embed_fn = embed_one

        # Lazily constructed ingest handler (avoid circular init).
        self._ingest_handler: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect the store client, ensure the schema, and persist the ontology."""
        await self._store.connect()
        dims = self._embed_dims
        if not dims:
            from graphknows.storage.embedder import embed_dim

            dims = embed_dim()
        await self._store.ensure_schema(dims)
        # Reference data, written once per namespace rather than per chunk. The
        # store call is count-guarded, so a restart is a no-op.
        if self._ontology_index is not None:
            await self._store.write_ontology(self._ontology_index)

    async def close(self) -> None:
        """Close the store client. Idempotent."""
        await self._store.close()

    def _handler(self) -> Any:
        from graphknows.ingestion.stm.ingest import _STMIngestHandler

        if self._ingest_handler is None:
            self._ingest_handler = _STMIngestHandler(
                store=self._store,
                extractor=self._extractor,
                embed_fn=self._embed_fn,
                rel_extractor=self._rel_extractor,
                enable_frames=self._enable_frames,
                ontology_index=self._ontology_index,
            )
        return self._ingest_handler

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    async def ingest(
        self,
        text: str,
        session_id: str,
        title: str = "",
        source_hint: str = "",
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> IngestResult:
        """Ingest raw text into the GraphStore."""
        return await self._handler().ingest_text(  # type: ignore[no-any-return]
            text,
            session_id=session_id,
            title=title,
            source_hint=source_hint,
            metadata=metadata or {},
            infer=infer,
        )

    # ------------------------------------------------------------------
    # Conversation ingestion (Phase-1 bridge: 1 turn = 1 CHUNK(kind='turn'))
    # ------------------------------------------------------------------

    async def ingest_messages(
        self,
        messages: list[Message],
        options: IngestOptions,
    ) -> IngestResult:
        """Ingest normalized Messages as CHUNK(kind='turn') rows."""
        from graphknows.models.ingest import IngestResult

        session_id = options.scope.to_session_id()
        dialog_id = options.scope.to_dialog_id()

        await self._store.ensure_session(session_id, started_at=str(time.time()))
        await self._store.ensure_file(
            file_id=dialog_id, session_id=session_id, name="", media_type="conversation"
        )

        t0 = time.perf_counter()
        total_entities = 0
        errors: list[str] = []
        extra_metadata = {
            "memory_class": options.memory_class.value,
            **options.metadata,
        }

        handler = self._handler()
        for i, msg in enumerate(messages):
            try:
                total_entities += await handler.ingest_turn(
                    session_id=session_id,
                    file_id=dialog_id,
                    seq=options.turn_index_offset + i,
                    role=msg.role.value,
                    text=msg.content,
                    metadata=extra_metadata,
                    infer=options.infer,
                )
            except Exception as exc:
                errors.append(f"turn {i}: {exc}")

        return IngestResult(
            session_id=session_id,
            file_id=dialog_id,
            title="",
            chunk_count=len(messages),
            entity_count=total_entities,
            elapsed_s=round(time.perf_counter() - t0, 3),
            errors=errors,
        )
