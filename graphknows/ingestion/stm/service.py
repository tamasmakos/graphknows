"""STMService: first-class service layer for memory ingestion.

Owns text ingest and conversation (turn) capture against the single per-namespace
:class:`~graphknows.storage.arcadedb.graph_store.GraphStore` (golden layer). For
retrieval, use DETRetriever.retrieve().
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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
        _embed_dims: Embedding dimension for ensure_schema on connect.
        _ontology_index: Optional OntologyIndex steering extraction labels.
    """

    def __init__(
        self,
        store: Any,
        extractor: Any,
        embed_fn: Any | None = None,
        embed_dims: int = 0,
        enable_frames: bool = False,
        ontology_index: Any | None = None,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._embed_dims = embed_dims
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
        # store call is guarded by IDENTITY (the ontology's own content digest
        # keys its ONTOLOGY vertex), so a restart is a no-op but an in-place
        # upgrade re-imports under its own version. Each write also mints a
        # SOURCE row so the (channel, version) it produced is enumerable.
        if self._ontology_index is not None:
            await self._store.write_ontology(self._ontology_index)
            await self._store.write_source(
                {
                    "channel": "ontology",
                    "version": self._ontology_index.digest,
                    "source_hash": self._ontology_index.digest,
                    "imported": len(self._ontology_index.terms),
                    "dropped_expressions": self._ontology_index.dropped_expressions,
                    "dropped_fe_mappings": 0,
                }
            )
        # FrameNet's FE plane, same story: reference data, once per namespace,
        # but guarded by IDENTITY now rather than presence — a corpus upgrade
        # (same FE count, different content) re-imports under a new version
        # instead of keeping the old plane forever (issue #145). Only
        # loaded/paid for when frames are enabled at all.
        if self._enable_frames:
            import asyncio
            import hashlib

            from graphknows.symbolic.framenet import fe

            def _build_fe() -> tuple[tuple[Any, ...], tuple[Any, ...], str]:
                # fe_rows()/semtype_rows() walk the full FrameNet corpus
                # (~6.6s) -- off the event loop so connect() never blocks
                # other coroutines on it. The corpus carries no version of
                # its own, so one is minted here the same way the ontology's
                # digest is: a short content hash, computed on this same
                # worker thread so connect() still does not block on it.
                rows = fe.fe_rows()
                payload = "\n".join(f"{r.id}\x1f{r.definition}" for r in rows)
                version = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
                return rows, fe.semtype_rows(), version

            fe_rows, semtype_rows, fe_version = await asyncio.to_thread(_build_fe)
            await self._store.write_frame_elements(fe_rows, semtype_rows, version=fe_version)
            await self._store.write_source(
                {
                    "channel": "framenet",
                    "version": fe_version,
                    "source_hash": fe_version,
                    "imported": len(fe_rows),
                    "dropped_expressions": 0,
                    "dropped_fe_mappings": fe.fe_mapping_drops(),
                }
            )

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
                enable_frames=self._enable_frames,
                ontology_index=self._ontology_index,
            )
        return self._ingest_handler

    async def rebind(self, channel: str = "ontology") -> dict[str, Any]:
        """Re-ground ``channel``'s stored mentions against the currently loaded index.

        Dual-write, in this order: read the namespace's existing (not yet
        superseded) INSTANCE_OF groundings for ``channel``, re-resolve each
        through the handler's own :meth:`_map_to_class` against the index
        this service was built with, write the results under the CURRENT
        version, and only then flag the old version's edges superseded. New
        groundings land first; old ones are retained and flagged, never
        deleted — an evaluated graph is not silently rewritten (issue #145).
        """
        handler = self._handler()
        version = getattr(self._ontology_index, "digest", "") or handler._ontology_source
        rows = await self._store.query(
            "MATCH (e:ENTITY)-[i:INSTANCE_OF]->(:ONTOLOGY_CLASS) "
            "WHERE i.channel = $channel AND coalesce(i.superseded, false) = false "
            "RETURN e.name AS name, e.type AS type, i.chunk_id AS chunk_id, "
            "i.span AS span",
            channel=channel,
        )
        mappings: list[dict[str, Any]] = []
        for row in rows:
            uri, _label, score, matcher = handler._map_to_class(str(row.get("type") or ""))
            if not uri:
                continue
            mappings.append(
                {
                    "name": row.get("name"),
                    "uri": uri,
                    "channel": channel,
                    "version": version,
                    "matcher": matcher,
                    "score": score,
                    "rank": 0,
                    "chunk_id": row.get("chunk_id"),
                    "span": row.get("span"),
                }
            )
        if mappings:
            await self._store.write_groundings(mappings)
        superseded = await self._store.supersede_groundings(channel, keep_version=version)
        return {"rebound": len(mappings), "superseded": superseded, "version": version}

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
        infer_flags: Sequence[bool] | None = None,
    ) -> IngestResult:
        """Ingest normalized Messages as CHUNK(kind='turn') rows.

        ``infer_flags``, when given, is parallel to ``messages`` and overrides
        ``options.infer`` per turn — the caller's own judgement about which
        turns are worth extracting from. This layer applies no content policy of
        its own; a turn with ``infer=False`` is still written and embedded and
        stays retrievable, it simply mints no entities or relations.
        """
        from graphknows.models.ingest import IngestResult

        session_id = options.scope.to_session_id()
        dialog_id = options.scope.to_dialog_id()

        await self._store.ensure_session(session_id, started_at=str(time.time()))
        await self._store.ensure_file(file_id=dialog_id, session_id=session_id)

        t0 = time.perf_counter()
        total_entities = 0
        errors: list[str] = []
        extra_metadata = {
            "memory_class": options.memory_class.value,
            **options.metadata,
        }

        handler = self._handler()
        chunks_before = handler._chunks_written
        for i, msg in enumerate(messages):
            try:
                # The turn's own timestamp is the anchor ingest_turn resolves
                # relative dates against. Passing it per message, not once for
                # the batch: a conversation spans days, and anchoring every turn
                # to one date is only marginally better than anchoring to now.
                turn_metadata = (
                    {**extra_metadata, "timestamp": msg.timestamp}
                    if msg.timestamp
                    else extra_metadata
                )
                total_entities += await handler.ingest_turn(
                    session_id=session_id,
                    file_id=dialog_id,
                    seq=options.turn_index_offset + i,
                    role=msg.role.value,
                    text=msg.content,
                    metadata=turn_metadata,
                    infer=(
                        options.infer
                        and (
                            infer_flags[i]
                            if infer_flags is not None and i < len(infer_flags)
                            else True
                        )
                    ),
                    # Message.name is the speaker alias — "Gina", not "user".
                    # It was accepted by the model and then dropped here, so a
                    # two-human conversation recorded every chunk's speaker as
                    # its coarse role and the real participants never reached
                    # CHUNK.speaker at all.
                    speaker=msg.name or "",
                    # Forces the pending window out on the final turn, so the
                    # tail of a conversation is never left buffered.
                    last_in_batch=(i == len(messages) - 1),
                )
            except Exception as exc:
                errors.append(f"turn {i}: {exc}")

        return IngestResult(
            session_id=session_id,
            file_id=dialog_id,
            title="",
            # Chunks actually written, which is NOT the message count once turns
            # are grouped into sliding windows.
            chunk_count=handler._chunks_written - chunks_before,
            entity_count=total_entities,
            elapsed_s=round(time.perf_counter() - t0, 3),
            errors=errors + handler._drain_decode_errors(),
            abstentions=handler._report_abstentions(),
        )
