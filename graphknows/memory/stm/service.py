"""STMService: first-class service layer for Short-Term Memory operations.

Owns STM ingest, dialog capture, and STM recall. Delegates to
_STMIngestHandler, DialogCapture, and recall_short_term respectively.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from graphknows.memory.config import STMConfig

if TYPE_CHECKING:
    from graphknows.memory.models.ingest import IngestResult
    from graphknows.memory.models.ingest_options import IngestOptions
    from graphknows.memory.models.message import Message
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
            from graphknows.memory.stores.embedder import embed_one

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
        return self._ingest_handler.ingest_text(  # type: ignore[no-any-return]
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
    # Conversation ingestion
    # ------------------------------------------------------------------

    def ingest_messages(
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
        from graphknows.memory.models.ingest import IngestResult
        from graphknows.memory.stm._async_bridge import _run
        from graphknows.memory.stm.dialog import DialogCapture, Turn

        session_id = options.scope.to_session_id()
        dialog_id = options.scope.to_dialog_id()

        if not options.infer:
            # Embed-only path — create a transient capture with no extractor
            dc: DialogCapture = DialogCapture(
                mg_store=self._mg,
                pg_store=self._pg,
                extractor=None,
                embed_fn=self._embed_fn,
            )
        else:
            if self._dialog_capture is None:
                self._dialog_capture = DialogCapture(
                    mg_store=self._mg,
                    pg_store=self._pg,
                    extractor=self._extractor,
                    embed_fn=self._embed_fn,
                )
            dc = self._dialog_capture

        # Ensure DIALOG node exists
        _run(
            self._mg.write_dialog_node(
                session_id=session_id,
                dialog_id=dialog_id,
                started_at=str(time.time()),
            )
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
            result = dc.write_turn(
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
        from graphknows.memory.models.recall import RecallResult
        from graphknows.memory.query._rrf_common import rrf_score
        from graphknows.memory.query._rrf_common import shape_score as _shape_score
        from graphknows.memory.stm._async_bridge import _run

        fn = embed_fn or self._embed_fn
        if fn is None:
            from graphknows.memory.stores.embedder import embed_one
            fn = embed_one

        words = [w for w in query.lower().split() if len(w) > 3]
        embedding = fn(query)
        results: dict[str, tuple[float, RecallResult]] = {}

        for rows, stype in (
            (_run(self._pg.search_stm_chunks(session_id, embedding, top_k=top_k * 2)), "chunk"),
            (_run(self._pg.search_stm_turns(session_id, embedding, top_k=top_k * 2)), "turn"),
        ):
            for rank, r in enumerate(rows or []):
                rrf = rrf_score(rank)
                cid = r["id"]
                if cid not in results or results[cid][0] < rrf:
                    results[cid] = (rrf, RecallResult(
                        text=r["text"], source_type=stype, scope="short_term",
                        score=rrf, cosine_score=r.get("cosine_score", 0.0),
                        entity_names=r.get("metadata", {}).get("entity_names", []),
                        heading_path=r.get("heading_path", ""),
                        provenance={"session_id": session_id, "file_id": r.get("file_id", ""),
                                    "memory_class": r.get("metadata", {}).get("memory_class", "raw_evidence")},
                        metadata=r.get("metadata", {}),
                    ))

        if words:
            for rank, ec in enumerate(_run(self._mg.query_entity_chunks(
                session_id=session_id, entity_names=words, top_k=top_k,
            )) or []):
                rrf = rrf_score(rank)
                cid = ec["chunk_id"]
                if cid in results:
                    old_score, old_result = results[cid]
                    old_result.score = old_score + rrf
                    old_result.entity_names = ec.get("entity_names", [])
                    results[cid] = (old_score + rrf, old_result)
                else:
                    results[cid] = (rrf, RecallResult(
                        text=ec.get("text", ""), source_type="chunk", scope="short_term",
                        score=rrf, cosine_score=0.0,
                        entity_names=ec.get("entity_names", []),
                        heading_path=ec.get("heading_path", ""),
                        provenance={"session_id": session_id, "file_id": ec.get("file_id", ""),
                                    "memory_class": ec.get("memory_class", "raw_evidence")},
                        metadata=ec.get("metadata", {}),
                    ))

        shaped = sorted(
            ((_shape_score(s, r.metadata.get("memory_class")), r) for s, r in results.values()),
            key=lambda x: x[0], reverse=True,
        )
        return [r for _, r in shaped[:top_k]]
