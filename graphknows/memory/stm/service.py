"""STMService: first-class service layer for Short-Term Memory operations.

Owns STM ingest, dialog capture, and STM recall. Delegates to
_STMIngestHandler, DialogCapture, and recall_short_term respectively.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from graphknows.memory.stm.config import STMConfig

if TYPE_CHECKING:
    from graphknows.memory.models.ingest import IngestResult
    from graphknows.memory.models.recall import RecallResult

log = logging.getLogger("graphknows.memory.stm.service")


class STMService:
    """First-class service layer for Short-Term Memory (STM) operations.

    Owns STM ingest, dialog turn capture, and STM recall. All logic is
    delegated to the sub-modules in this package; this class is the
    single public surface that callers (MemoryAdapter, MCP handlers, etc.)
    should reference.

    # TODO(Wave 3): replace direct store references with IGraphDb adapter.

    Attributes:
        _mg: MemgraphStore for session evidence graph writes/queries.
        _pg: PgVectorStore for stm_chunks / stm_turns writes/queries.
        _extractor: Entity extractor exposing extract(text) -> ExtractionResult.
        _embed_fn: Embedding callable (text -> list[float]).
        _rel_extractor: Optional relation extractor.
        _config: STMConfig with tuning parameters.
    """

    def __init__(
        self,
        mg_store: Any,
        pg_store: Any,
        extractor: Any,
        embed_fn: Any | None = None,
        rel_extractor: Any | None = None,
        config: STMConfig | None = None,
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._extractor = extractor
        self._rel_extractor = rel_extractor
        self._config = config or STMConfig()

        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            from graphknows.retrieval.stores.embedder import embed_one

            self._embed_fn = embed_one

        # Lazily constructed sub-handlers (avoid circular init)
        self._ingest_handler: Any = None
        self._dialog_capture: Any = None

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
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
        from graphknows.memory.stm.ingest import _STMIngestHandler

        if self._ingest_handler is None:
            self._ingest_handler = _STMIngestHandler(
                mg_store=self._mg,
                pg_store=self._pg,
                extractor=self._extractor,
                embed_fn=self._embed_fn,
                rel_extractor=self._rel_extractor,
            )
        return self._ingest_handler.ingest_text(  # type: ignore[return-value]
            text,
            session_id=session_id,
            title=title,
            source_hint=source_hint,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Dialog
    # ------------------------------------------------------------------

    def record_turn(
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
        from graphknows.memory.stm.dialog import DialogCapture

        if self._dialog_capture is None:
            self._dialog_capture = DialogCapture(
                mg_store=self._mg,
                pg_store=self._pg,
                extractor=self._extractor,
                embed_fn=self._embed_fn,
            )
        return self._dialog_capture.write_turn(session_id, dialog_id, turn)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        embed_fn: Any | None = None,
    ) -> list[RecallResult]:
        """Recall from short-term memory using hybrid RRF search.

        Args:
            query: Natural-language query string.
            session_id: Session namespace.
            top_k: Maximum results to return.
            embed_fn: Override embedding callable (uses self._embed_fn if None).

        Returns:
            List of RecallResult sorted by score descending.
        """
        from graphknows.memory.stm.recall import recall_short_term

        return recall_short_term(
            mg_store=self._mg,
            pg_store=self._pg,
            query=query,
            session_id=session_id,
            top_k=top_k,
            embed_fn=embed_fn or self._embed_fn,
        )
