"""STMService: first-class service layer for Short-Term Memory operations.

Owns STM ingest and dialog capture. Delegates to _STMIngestHandler and
DialogCapture. For retrieval, use DETRetriever.retrieve(scope="stm").
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from graphknows.ingestion.stm.config import STMConfig

if TYPE_CHECKING:
    from graphknows.models.ingest import IngestResult
    from graphknows.models.ingest_options import IngestOptions
    from graphknows.models.message import Message

log = logging.getLogger("graphknows.ingestion.stm.service")


class STMService:
    """First-class service layer for Short-Term Memory (STM) operations.

    Owns STM ingest, dialog turn capture, and STM recall. All logic is
    delegated to the sub-modules in this package; this class is the
    single public surface that callers (MemoryAdapter, MCP handlers, etc.)
    should reference.

    # TODO(Wave 3): replace direct store references with IGraphDb adapter.

    Attributes:
        _stm: ArcadeDBSTMStore for session evidence graph writes/queries.
        _vec: ArcadeDBVectorStore for stm_chunks / stm_turns writes/queries.
        _extractor: Entity extractor exposing extract(text) -> ExtractionResult.
        _embed_fn: Embedding callable (text -> list[float]).
        _rel_extractor: Optional relation extractor.
        _config: STMConfig with tuning parameters.
    """

    def __init__(
        self,
        stm_store: Any,
        vec_store: Any,
        extractor: Any,
        embed_fn: Any | None = None,
        rel_extractor: Any | None = None,
        config: STMConfig | None = None,
    ) -> None:
        self._stm = stm_store
        self._vec = vec_store
        self._extractor = extractor
        self._rel_extractor = rel_extractor
        self._config = config or STMConfig()

        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            from graphknows.storage.embedder import embed_one

            self._embed_fn = embed_one

        # Lazily constructed sub-handlers (avoid circular init)
        self._ingest_handler: Any = None
        self._dialog_capture: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect the shared store client on the caller's event loop.

        Both stores share one ArcadeDBClient (see build_stm_service), so the
        client is connected once and each store ensures its own schema.
        """
        await self._stm.connect()
        await self._stm.ensure_schema()
        await self._vec.connect()  # idempotent client connect + vector schema

    async def close(self) -> None:
        """Close the shared store client. Idempotent."""
        await self._stm.close()
        await self._vec.close()

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
    ) -> IngestResult:
        """Ingest raw text into STM.

        Args:
            text: Raw source text.
            session_id: Session namespace.
            title: Optional human title.
            source_hint: Original path/URL for provenance.
            metadata: Extra key/value pairs stored on the FILE node.

        Returns:
            IngestResult with counts and timing.
        """
        from graphknows.ingestion.stm.ingest import _STMIngestHandler

        if self._ingest_handler is None:
            self._ingest_handler = _STMIngestHandler(
                stm_store=self._stm,
                vec_store=self._vec,
                extractor=self._extractor,
                embed_fn=self._embed_fn,
                rel_extractor=self._rel_extractor,
            )
        return await self._ingest_handler.ingest_text(  # type: ignore[no-any-return]
            text,
            session_id=session_id,
            title=title,
            source_hint=source_hint,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Dialog
    # ------------------------------------------------------------------

    async def record_turn(
        self,
        session_id: str,
        dialog_id: str,
        turn: Any,
    ) -> Any:
        """Record a single dialog turn into the STM interaction graph.

        Args:
            session_id: Parent session namespace.
            dialog_id: Parent dialog identifier.
            turn: Turn dataclass from stm.dialog.

        Returns:
            DialogResult with entity_count and any errors.
        """
        from graphknows.ingestion.stm.dialog import DialogCapture

        if self._dialog_capture is None:
            self._dialog_capture = DialogCapture(
                stm_store=self._stm,
                vec_store=self._vec,
                extractor=self._extractor,
                embed_fn=self._embed_fn,
            )
        return await self._dialog_capture.write_turn(session_id, dialog_id, turn)

    # ------------------------------------------------------------------
    # Conversation ingestion
    # ------------------------------------------------------------------

    async def ingest_messages(
        self,
        messages: list[Message],
        options: IngestOptions,
    ) -> IngestResult:
        """Ingest a list of normalized Messages into STM as dialog turns.

        Args:
            messages: Pre-normalized list from normalize_messages().
            options: Scope, infer flag, memory_class, and metadata.

        Returns:
            IngestResult with turn count, entity count, and any errors.
        """
        from graphknows.models.ingest import IngestResult
        from graphknows.ingestion.stm.dialog import DialogCapture, Turn

        session_id = options.scope.to_session_id()
        dialog_id = options.scope.to_dialog_id()

        if not options.infer:
            # Embed-only path — create a transient capture with no extractor
            dc: DialogCapture = DialogCapture(
                stm_store=self._stm,
                vec_store=self._vec,
                extractor=None,
                embed_fn=self._embed_fn,
            )
        else:
            if self._dialog_capture is None:
                self._dialog_capture = DialogCapture(
                    stm_store=self._stm,
                    vec_store=self._vec,
                    extractor=self._extractor,
                    embed_fn=self._embed_fn,
                )
            dc = self._dialog_capture

        # Ensure DIALOG node exists
        await self._stm.write_dialog_node(
            session_id=session_id,
            dialog_id=dialog_id,
            started_at=str(time.time()),
        )

        t0 = time.perf_counter()
        total_entities = 0
        errors: list[str] = []
        extra_metadata = {
            "memory_class": options.memory_class.value,
            **options.metadata,
        }

        for i, msg in enumerate(messages):
            turn = Turn(
                id=str(uuid.uuid4()),
                index=options.turn_index_offset + i,
                role=msg.role.value,
                text=msg.content,
            )
            result = await dc.write_turn(
                session_id=session_id,
                dialog_id=dialog_id,
                turn=turn,
                extra_metadata=extra_metadata,
            )
            total_entities += result.entity_count
            errors.extend(result.errors)

        return IngestResult(
            session_id=session_id,
            file_id=dialog_id,
            title="",
            chunk_count=len(messages),
            entity_count=total_entities,
            elapsed_s=round(time.perf_counter() - t0, 3),
            errors=errors,
        )
