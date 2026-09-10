"""Runtime facade for graphknows memory operations.

This module is the transport-neutral seam for MCP and in-process callers that
need memory operations without learning store lifecycles, background-loop
details, or promotion ordering.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from hashlib import sha256
from typing import Any

from graphknows.bounds import (
    check_limit,
    check_metadata_size,
    check_scope_id,
    check_text_length,
    check_top_k,
)
from graphknows.exceptions import ConfigurationError
from graphknows.ingestion.parsers.text import PlainTextParser, strip_content_heading
from graphknows.models.fact import FactState
from graphknows.models.hit import Hit
from graphknows.models.message import Message
from graphknows.models.report import Counters, IngestReport, RecallBudget, RecallResult
from graphknows.models.segment import Segment, SegmentKind
from graphknows.models.source import Source, session_uri
from graphknows.packs import DomainPack, load_packs
from graphknows.settings import Decoder, GraphKnowsSettings, MemoryMode, get_settings
from graphknows.storage.arcadedb._sql import not_forgotten
from graphknows.storage.arcadedb.writers import WRITTEN_TYPES

logger = logging.getLogger(__name__)

_INGEST_LOCKS: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
"""One ingest lock per namespace, shared by every ``Memory`` in this process.

A namespace is one physical database, so two facade instances writing the same
namespace must queue behind the same lock (FR-047).

ponytail: in-process only — a second service process needs a store-side lease.
"""


def _with_mode(settings: GraphKnowsSettings, mode: MemoryMode | str | None) -> GraphKnowsSettings:
    """Return ``settings`` with an explicit ``mode`` argument folded in.

    The argument wins over the settings value, exactly as ``namespace``'s does.
    ``model_copy`` leaves the caller's object untouched — and skips validators,
    so the production-secret check is re-run by hand.
    """
    if mode is None:
        return settings
    try:
        resolved = settings.model_copy(update={"mode": MemoryMode(mode)})
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    resolved.check_production_secrets()
    return resolved


def _require_decoder_key(settings: GraphKnowsSettings) -> None:
    """Fail here, not at the first chunk of a long ingest, on a keyless decoder."""
    if settings.decoder is Decoder.llm and not settings.llm_api_key:
        raise ConfigurationError(
            f"mode={settings.mode} decodes with the LLM and no API key resolves; "
            "set GRAPHKNOWS_LLM_API_KEY in the environment or .env."
        )


def _build_extractor(settings: GraphKnowsSettings) -> Any:
    """The extractor the configured decoder puts behind the write path (FR-043)."""
    from graphknows.ingestion.extraction.protocol import LLMExtractor, LocalExtractor

    return LLMExtractor() if settings.decoder is Decoder.llm else LocalExtractor()


def _with_queue_wait(report: IngestReport, waited_ms: int) -> IngestReport:
    """Return *report* with the time it spent queueing folded into its counters."""
    counters = report.counters.model_copy(update={"ingest_queue_wait_ms": waited_ms})
    return report.model_copy(update={"counters": counters})


def _redact_store_error(exc: Exception, *, where: str) -> str:
    """Log ``exc`` in full and return a caller-safe summary with a correlation id.

    ArcadeDB failures carry the request host and the raw server response body
    (see ``ArcadeDBClient._raise_for_status``), and ``doctor``, ``purge_memory``
    and ``flush`` hand their errors straight to an MCP caller — that
    detail must not cross the boundary verbatim (FR-031). The correlation id
    ties the redacted message back to the full one in the log, so redacting
    for the caller does not blind the operator (Constitution V).
    """
    correlation_id = uuid.uuid4().hex
    logger.error("%s failed [correlation_id=%s]: %s", where, correlation_id, exc)
    return f"correlation_id={correlation_id}; see server logs for details"


def _format_fused_hits(fused_results: list[dict[str, Any]], session_id: str) -> list[Hit]:
    return [
        Hit(
            text=strip_content_heading(r.get("text", "")),
            speaker=str(r.get("speaker", "") or ""),
            ts=str(r.get("ts", "") or ""),
            score=float(r.get("rrf_score", 0.0) or 0.0),
            sources=str(r.get("sources", "") or ""),
            session_id=session_id,
            chunk_id=str(r.get("chunk_id", "") or ""),
            doc_id=str(r.get("doc_id", "") or ""),
            entities=list(r.get("entities", []) or []),
            metadata=dict(r.get("metadata", {}) or {}),
        )
        for r in fused_results
    ]


def _turn_anchor(ts: object) -> datetime | None:
    """A turn's own time anchor, or ``None`` so ``Segment`` marks it inferred."""
    try:
        return datetime.fromisoformat(str(ts))
    except ValueError:
        return None


class Memory:
    """The graphknows memory facade — the primary developer-facing API.

    Transport-neutral: the MCP server is a thin wrapper over this class, so
    every transport sees identical behaviour. One ``Memory`` instance is safe
    for concurrent asyncio use.
    """

    def __init__(
        self,
        settings: GraphKnowsSettings | None = None,
        namespace: str | None = None,
        mode: MemoryMode | str | None = None,
        channels: list[Any] | None = None,
        packs: Sequence[DomainPack] = (),
    ) -> None:
        # The mode picks the decoder. Explicit arg wins over the settings value;
        # nothing implicit sets it, so a caller who named it meant it — hence the
        # keyless LLM decoder failing here rather than mid-ingest.
        self.settings = _with_mode(settings or get_settings(), mode)
        _require_decoder_key(self.settings)
        # The namespace selects one physical ArcadeDB database (mem_<ns>, "" -> mem,
        # the golden layer). Explicit arg wins over the settings default.
        self.namespace = self.settings.namespace if namespace is None else namespace
        # Caller-supplied channels: the extension seam for an application that
        # wants its own structure in the graph. One object carries both halves —
        # `populate` writes it during flush, `collect` reads it during recall —
        # so a custom schema cannot end up written but unread, or read but
        # never written. Appended after the settings-gated defaults.
        self._channels: list[Any] = list(channels or [])
        # The domain knowledge this facade reads with. Loaded here, not at first
        # ingest, so two packs contesting a symbol fail the caller who supplied
        # them rather than a write months later (FR-025).
        self._packs = tuple(packs)
        load_packs(self._packs)
        # One retriever over one connected store, built lazily on first use.
        self._retriever: Any = None
        self._pipeline: Any = None
        self._store: Any = None
        self._connect_lock = asyncio.Lock()

    async def __aenter__(self) -> Memory:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the store connection held by this runtime instance."""
        if self._store is not None:
            try:
                await self._store.close()
            except Exception as exc:
                # Close is best-effort — a failure here must not mask the real
                # error that is usually already unwinding. But an unlogged
                # swallow makes a leaked pool indistinguishable from a clean
                # shutdown from outside, so name it.
                logger.warning("Memory.close: store close failed (non-fatal): %s", exc)
        self._retriever = None
        self._pipeline = None
        self._store = None

    async def _get_retriever(self) -> Any:
        """Return the cached retriever, connecting the store lazily on first call.

        Guarded by an ``asyncio.Lock`` so concurrent first-use callers connect
        the store exactly once.
        """
        if self._retriever is not None:
            return self._retriever
        async with self._connect_lock:
            if self._retriever is not None:  # another coroutine won
                return self._retriever
            return await self._build_retriever()

    async def _build_retriever(self) -> Any:
        from graphknows.retrieval import build_retriever
        from graphknows.storage import build_graph_store
        from graphknows.storage.embedder import embed_dim

        s = self.settings
        store = build_graph_store(s, self.namespace)
        # Record the store before connecting so a failure during connect or
        # ensure_schema is cleaned up by close() instead of orphaning pools.
        self._store = store
        try:
            await store.connect()
            dims = s.embed_dimensions or embed_dim()
            await store.ensure_schema(dims)
        except Exception:
            await self.close()
            raise
        self._retriever = build_retriever(s, store=store, extra_channels=self._channels)
        return self._retriever

    async def _get_pipeline(self) -> Any:
        """The write path over the connected store, built once per instance."""
        if self._pipeline is None:
            from graphknows.ingestion.pipeline import IngestPipeline

            await self._get_retriever()  # connects the store + ensures schema
            pipeline = IngestPipeline(self._store, _build_extractor(self.settings), self._packs)
            # The symbolic index the packs bring, written before the first
            # source: labelling edges MATCH these vertices.
            await pipeline.write_symbols()
            self._pipeline = pipeline
        return self._pipeline

    async def ingest(self, source: Source, segments: Sequence[Segment]) -> IngestReport:
        """Write *source* and its *segments*, serialised per namespace (FR-047).

        A concurrent ingest of the same namespace queues here rather than
        failing, and how long it waited comes back as ``ingest_queue_wait_ms``
        — a queue that has become the bottleneck is visible in the report, not
        only in a latency graph. Recall takes no lock and stays available.
        """
        queued_at = time.monotonic()
        async with _INGEST_LOCKS[self.namespace]:
            waited_ms = round((time.monotonic() - queued_at) * 1000)
            pipeline = await self._get_pipeline()
            report = await pipeline.ingest(source, segments)
        return _with_queue_wait(report, waited_ms)

    async def recall(self, query: str, budget: RecallBudget | None = None) -> RecallResult:
        """The facts *query*'s symbols activate, with the evidence asserting them."""
        from graphknows.retrieval.retriever import SymbolRecall

        await self._get_retriever()  # connects the store + ensures schema
        return await SymbolRecall(self._store).recall(query, budget)

    async def forget(self, record_id: str) -> Counters:
        """Tombstone the fact, entity or segment *record_id* (FR-037).

        A tombstone, never a delete: the record stays readable to a merge-log
        replay while every retrieval path filters it out. Forgetting a segment
        cascades to every fact left without live evidence (edge case 8), and
        how many that was comes back as ``facts_excluded_tombstoned``.
        """
        await self._get_retriever()  # connects the store + ensures schema
        forgotten = str(FactState.FORGOTTEN)
        for label in ("FACT", "ENTITY", "SEGMENT"):
            await self._store.command(
                f"MATCH (v:{label} {{id: $id}}) SET v.state = $state",
                id=record_id,
                state=forgotten,
            )
        return Counters(facts_excluded_tombstoned=await self._cascade_forget(record_id))

    async def _cascade_forget(self, segment_id: str) -> int:
        """Tombstone the facts whose only evidence was segment *segment_id*.

        A no-op unless *segment_id* named a segment: a fact still asserted in a
        live segment keeps its evidence and its state.
        """
        rows = await self._store.command(
            f"MATCH (f:FACT)-[:ASSERTED_IN]->(:SEGMENT {{id: $id}}) "
            f"OPTIONAL MATCH (f)-[:ASSERTED_IN]->(e:SEGMENT) WHERE {not_forgotten('e')} "
            "WITH f, count(e) AS live WHERE live = 0 "
            "SET f.state = $state RETURN count(f) AS facts",
            id=segment_id,
            state=str(FactState.FORGOTTEN),
        )
        return int(rows[0]["facts"]) if rows else 0

    async def doctor(self) -> dict[str, Any]:
        """Check configured backend connectivity for this runtime's namespace."""
        from graphknows._version import get_version
        from graphknows.storage import build_arcadedb_client
        from graphknows.storage.namespace import db_name

        db = db_name(self.namespace)
        result: dict[str, Any] = {"version": get_version()}
        try:
            c = build_arcadedb_client(self.settings)
            await c.connect()
            result["arcadedb"] = await c.database_exists(db)
            await c.close()
        except Exception as exc:
            result["arcadedb"] = _redact_store_error(exc, where="doctor")
        result["pid"] = os.getpid()
        return result

    async def stats(self, session_id: str = "") -> dict[str, Any]:
        """Return golden-layer memory statistics for this runtime's namespace.

        Counts come from the single database (``mem``/``mem_<ns>``); the
        ``raw``/``consolidated`` split is reported per lifecycle ``state``.
        """
        from graphknows.storage import build_arcadedb_client
        from graphknows.storage.arcadedb.graph_store import GraphStore
        from graphknows.storage.namespace import db_name

        db = db_name(self.namespace)
        c = build_arcadedb_client(self.settings)
        await c.connect()
        try:
            if session_id:
                sid = session_id
                raw_chunks = await c.query(
                    db,
                    "MATCH (n:CHUNK {session_id: $sid}) WHERE n.state = 'raw' RETURN count(n) AS c",
                    params={"sid": sid},
                )
                cons_chunks = await c.query(
                    db,
                    "MATCH (n:CHUNK {session_id: $sid}) WHERE n.state = 'consolidated' "
                    "RETURN count(n) AS c",
                    params={"sid": sid},
                )
            else:
                raw_chunks = await c.query(
                    db, "MATCH (n:CHUNK) WHERE n.state = 'raw' RETURN count(n) AS c"
                )
                cons_chunks = await c.query(
                    db, "MATCH (n:CHUNK) WHERE n.state = 'consolidated' RETURN count(n) AS c"
                )

            entities = await c.query(db, "MATCH (n:ENTITY) RETURN count(n) AS c")
            topics = await c.query(db, "MATCH (n:TOPIC) RETURN count(n) AS c")
            sessions = await c.query(db, "MATCH (n:SESSION) RETURN count(n) AS c")
            # ENTITY-[:REL]->ENTITY is the extracted relation layer. Reported
            # so a caller can see whether ingest produced relations at all —
            # an entity count alone hides an empty relation graph.
            relations = await c.query(db, "MATCH (:ENTITY)-[r:REL]->(:ENTITY) RETURN count(r) AS c")

            def _n(rows: list[dict[str, Any]]) -> int:
                return int(rows[0].get("c") or 0) if rows else 0

            return {
                "sessions": _n(sessions),
                "entities": _n(entities),
                "relations": _n(relations),
                "topics": _n(topics),
                "chunks": {"raw": _n(raw_chunks), "consolidated": _n(cons_chunks)},
                # SOURCE rows: which (channel, version) this namespace holds —
                # the reader that keeps SOURCE from being write-only (#145).
                # Wraps the already-connected client rather than opening a
                # second connection just to reuse GraphStore.sources().
                "sources": await GraphStore(client=c, db=db).sources(),
            }
        finally:
            await c.close()

    async def ingest_memory(
        self,
        messages: str | list[dict[str, Any]] | dict[str, Any] = "",
        *,
        session_id: str = "",
        user_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        title: str = "",
        infer: bool = True,
        metadata: dict[str, Any] | None = None,
        text: str = "",
    ) -> dict[str, Any]:
        """Ingest text or messages into memory — one lifecycle, two shapes.

        Both shapes run the same pipeline eagerly and are graph-searchable when
        the call returns (FR-045): a conversation is one ``Source`` whose turns
        are ``turn`` segments, a document is one ``Source`` whose chunks are
        ``prose`` segments. There is no buffer to drain any more.
        """
        if text and not messages:
            messages = text

        check_scope_id(session_id, "session_id")
        check_scope_id(user_id, "user_id")
        check_scope_id(agent_id, "agent_id")
        check_scope_id(run_id, "run_id")
        check_metadata_size(metadata)
        if isinstance(messages, str):
            check_text_length(messages, "text")

        from graphknows.models.message import normalize_messages
        from graphknows.models.scope import MemoryScope

        scope = MemoryScope(
            user_id=user_id or None,
            agent_id=agent_id or None,
            run_id=run_id or session_id or None,
        )
        is_conversation = isinstance(messages, list) or (
            isinstance(messages, dict) and "role" in messages
        )

        started = time.monotonic()
        if is_conversation:
            msgs = normalize_messages(messages)
            if not msgs:
                return {
                    "file_id": "",
                    "chunks": 0,
                    "entities": 0,
                    "turns": 0,
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "errors": [],
                }
            source, segments = self._session_as_source(scope.to_session_id(), msgs)
            report = await self.ingest(source, segments)
            return {
                "file_id": source.id,
                "chunks": report.segments_written,
                "entities": report.entities_touched,
                "turns": len(msgs),
                "elapsed_s": round(time.monotonic() - started, 3),
                "errors": [],
            }

        source, segments = self._document_as_source(str(messages), title)
        report = await self.ingest(source, segments)
        return {
            "file_id": source.id,
            "chunks": report.segments_written,
            "entities": report.entities_touched,
            "elapsed_s": round(time.monotonic() - started, 3),
            "errors": [],
            "warnings": [],
            "abstentions": {},
        }

    async def recall_memory(
        self,
        query: str,
        *,
        session_id: str = "",
        top_k: int = 5,
        scope: str = "both",
        cross_session: bool = False,
    ) -> dict[str, Any]:
        """Recall ranked hits: ``{"hits": list[Hit], "facts": [...]}``.

        Hits are :class:`~graphknows.models.hit.Hit` objects, not dicts — one
        shape for every caller, carrying the speaker and timestamp an answerer
        needs to attribute and date a memory. Render them with
        :func:`~graphknows.models.hit.render_memories`; transports that cannot
        carry the type call :meth:`~graphknows.models.hit.Hit.to_dict`.

        ``scope`` selects the segment lifecycle the retriever filters on:
        ``"ltm"`` reads consolidated segments only, ``"both"`` (the default)
        applies no state filter. The buffer that ``"stm"`` used to name is
        gone — every ingest is eager now — so it normalises to ``"both"``.

        ``cross_session``, when True, makes the search namespace-wide
        (``session_id=""`` to the retriever, which treats that as "no session
        filter"). Defaults to False so the MCP server and web UI keep today's
        session-scoped behaviour unless they opt in.
        """
        # Bounded at the trust boundary (FR-030), not silently clamped: a
        # top_k above MAX_TOP_K is rejected. It used to have no ceiling at all
        # — measured discarding evidence the retriever had already ranked (a
        # recall-vs-depth sweep on conv-30 plateaued at 0.827 under an old cap
        # of 100 and read as gold chunks being UNREACHABLE) — but an unbounded
        # top_k also multiplies straight into the retriever's pool_k and the
        # store's k_pre = top_k * 20, so an oversized request is a resourcing
        # problem before it is a ranking one.
        check_scope_id(session_id, "session_id")
        check_top_k(top_k)
        top_k = max(1, top_k)
        if scope not in {"ltm", "both"}:
            scope = "both"

        retriever = await self._get_retriever()
        ctx = await retriever.retrieve(
            query,
            session_id="" if cross_session else session_id,
            top_k=top_k,
        )
        hits = _format_fused_hits(ctx.fused_results, session_id)
        facts = list(getattr(ctx, "facts", []) or [])

        # ``scope`` is echoed back POST-normalisation so a caller reports what
        # was actually searched rather than what it asked for — and so no
        # transport has to reimplement the fallback rule to say so.
        return {"hits": hits[:top_k], "facts": facts, "scope": scope}

    def _document_as_source(self, text: str, title: str) -> tuple[Source, list[Segment]]:
        """One document as a ``Source``, each heading-aligned chunk a ``prose`` ``Segment``."""
        parsed = PlainTextParser().parse(text, title=title)
        source = Source(
            uri=f"document:{parsed.source_id}",
            content_hash=sha256(text.encode()).hexdigest(),
            mime="text/plain",
            namespace=self.namespace,
            meta={"title": parsed.title},
        )
        segments = [
            Segment(
                source_id=source.id,
                text=chunk.text,
                kind=SegmentKind.prose,
                path=chunk.heading_path,
                byte_range=(chunk.char_offset, chunk.char_offset + len(chunk.text.encode())),
            )
            for chunk in parsed.segments
        ]
        return source, segments

    def _session_as_source(
        self, session_id: str, turns: Sequence[Message]
    ) -> tuple[Source, list[Segment]]:
        """One session as a ``Source``, each of its turns as a ``turn`` ``Segment``.

        The joined transcript is the source's content and every turn addresses
        its own byte range of it, so both identities are derived: re-ingesting a
        session rewrites the same segments in place instead of forking a set.
        """
        transcript = "\n".join(turn.content for turn in turns)
        source = Source(
            uri=session_uri(session_id),
            content_hash=sha256(transcript.encode()).hexdigest(),
            mime="text/plain",
            namespace=self.namespace,
        )
        segments: list[Segment] = []
        start = 0
        for index, turn in enumerate(turns):
            end = start + len(turn.content.encode())
            segments.append(
                Segment(
                    source_id=source.id,
                    text=turn.content,
                    kind=SegmentKind.turn,
                    path=str(index),
                    byte_range=(start, end),
                    # Free string, not a closed role vocabulary: a session may
                    # name speakers no four-value enum has a member for.
                    role=turn.role,
                    # The turn's own time anchor, carried through so relative
                    # dates in it resolve against when it was said rather than
                    # against ingestion time.
                    observed_at=_turn_anchor(turn.timestamp or ""),
                )
            )
            start = end + 1  # the newline this join put after the turn
        return source, segments

    async def flush(self) -> dict[str, Any]:
        """Report how identity resolved and what this run wrote but never read.

        The write path is eager — a source is extracted, labelled and asserted
        as it is ingested — so there is nothing left to promote (FR-045). What
        flush still answers is the two questions no single ingest can:

        1. **Identity**: exact-key only until the identity story lands, so no
           merge is committed and no candidate is proposed (FR-048).
        2. **Dead weight**: the golden types the write path writes, minus the
           ones this connection has actually read back (FR-040). A plane that
           is written on every ingest and read by nothing is cost with no
           retrieval to show for it.
        """
        errors: list[str] = []
        unread: tuple[str, ...] = ()
        try:
            await self._get_retriever()  # connects the store + ensures schema
            unread = tuple(sorted(WRITTEN_TYPES - self._store.read_types))
        except Exception as exc:
            errors.append(f"connect: {_redact_store_error(exc, where='flush:connect')}")
        return {
            "identity": "exact-key",
            "merges": 0,
            "unread_types": list(unread),
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Namespace-first SDK surface. ``user_id``/``agent_id``/``run_id`` are the
    # memory scope (mem0-style); at least one is required. These are the methods
    # an application or agent framework calls; the ``*_memory`` verbs above back
    # the MCP + web transports.
    # ------------------------------------------------------------------

    @staticmethod
    def _scoped_session_id(user_id: str, agent_id: str, run_id: str, session_id: str) -> str:
        """Resolve an explicit session_id, else derive one from the memory scope."""
        if session_id:
            return session_id
        from graphknows.models.scope import MemoryScope

        return MemoryScope(
            user_id=user_id or None,
            agent_id=agent_id or None,
            run_id=run_id or None,
        ).to_session_id()

    async def add(
        self,
        messages: str | list[dict[str, Any]] | dict[str, Any] = "",
        *,
        user_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        title: str = "",
        infer: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add messages (or raw text) to memory under a user/agent/run scope."""
        return await self.ingest_memory(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            title=title,
            infer=infer,
            metadata=metadata,
        )

    async def search(
        self,
        query: str,
        *,
        user_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        session_id: str = "",
        top_k: int = 5,
        scope: str = "both",
    ) -> dict[str, Any]:
        """Search memory scoped to a user/agent/run (or an explicit session).

        Always passes ``cross_session=True`` to :meth:`recall_memory`: the GRAPH
        half of the search is namespace-wide, so a later session recalls facts
        consolidated from earlier sessions in the same namespace. The STM
        buffer half stays scoped to this session regardless — unflushed turns
        from another session are never visible here.
        """
        sid = self._scoped_session_id(user_id, agent_id, run_id, session_id)
        return await self.recall_memory(
            query, session_id=sid, top_k=top_k, scope=scope, cross_session=True
        )

    async def purge_memory(self, session_id: str = "") -> dict[str, Any]:
        """Purge memory for this namespace (golden layer, single database).

        ``session_id`` deletes just that session's subgraph; otherwise all data
        is cleared (schema and indexes are preserved).
        """
        from graphknows.storage import build_graph_store

        errors: list[str] = []
        deleted = False
        try:
            store = build_graph_store(self.settings, self.namespace)
            await store.connect()
            try:
                if session_id:
                    await store.delete_source(session_uri(session_id))
                else:
                    await store.clear_all()
                deleted = True
            finally:
                await store.close()
        except Exception as exc:
            errors.append(f"arcadedb: {_redact_store_error(exc, where='arcadedb')}")
        return {
            "session_id": session_id or "(all)",
            "deleted": deleted,
            "errors": errors,
        }

    async def drop_namespace(self) -> dict[str, Any]:
        """Drop this runtime's entire namespace database (``mem_<ns>``).

        An O(1) reset — it removes the database outright instead of walking and
        DETACH-DELETE-ing every node. Refuses to drop the default (``""``)
        namespace, whose database (``mem``) is shared and must be cleared with
        :meth:`purge_memory` instead of dropped.
        """
        from graphknows.storage import build_arcadedb_client
        from graphknows.storage.namespace import db_name

        if not self.namespace:
            return {
                "dropped": False,
                "namespace": "",
                "error": "refusing to drop the default namespace; use purge_memory instead",
            }
        db = db_name(self.namespace)
        await self.close()  # release any cached retriever store bound to this DB
        client = build_arcadedb_client(self.settings)
        await client.connect()
        try:
            await client.drop_database(db)
        finally:
            await client.close()
        return {"dropped": True, "namespace": self.namespace, "databases": [db]}

    async def corpus_ingest(
        self,
        documents: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest a batch of documents into one session, then report on the run.

        Runs the same golden-layer path as ``ingest_memory`` + ``flush``.
        It previously delegated to ``CorpusIngestWorkflow``, which was still
        wired for the retired three-store (stm/ltm/vector) split: it called
        ``STMService`` with removed keyword arguments, so every invocation
        returned ``{"error": ...}`` with zero documents processed — and even
        past that, it wrote to databases the golden retrieval path never reads.

        A document is ``{"text": ..., "id": ..., "title": ...}``; a document
        without text is recorded in ``errors`` and skipped.
        """
        import time

        if not documents:
            return {"documents_processed": 0, "error": "no documents provided"}

        t0 = time.monotonic()
        sid = session_id or f"corpus-{uuid.uuid4().hex}"
        errors: list[str] = []
        processed = chunks = entities = 0

        for doc in documents:
            text = str(doc.get("text", "") or "")
            doc_id = doc.get("id", "<unknown>")
            if not text.strip():
                errors.append(f"Skipped document missing 'text': id={doc_id}")
                continue
            try:
                res = await self.ingest_memory(
                    text, session_id=sid, title=str(doc.get("title", "") or "")
                )
            except Exception as exc:
                errors.append(f"ingest failed for doc id={doc_id}: {exc}")
                continue
            processed += 1
            chunks += int(res.get("chunks", 0) or 0)
            entities += int(res.get("entities", 0) or 0)
            errors.extend(res.get("errors", []) or [])

        flushed = await self.flush()
        errors.extend(flushed.get("errors", []) or [])

        return {
            "session_id": sid,
            "documents_processed": processed,
            "entities_extracted": entities,
            "chunks_written": chunks,
            "unread_types": flushed["unread_types"],
            "elapsed_s": round(time.monotonic() - t0, 3),
            "errors": errors,
        }

    async def ltm_entity(self, name: str) -> dict[str, Any]:
        """Return one golden-layer ENTITY by case-insensitive name."""
        from graphknows.storage import build_graph_store

        store = build_graph_store(self.settings, self.namespace)
        await store.connect()
        try:
            rows = await store.query(
                "MATCH (e:ENTITY) WHERE toLower(e.name) = toLower($name) "
                "RETURN e.name AS name, e.type AS type, e.confidence AS confidence, "
                "e.pagerank AS pagerank, e.community_id AS community_id LIMIT 1",
                name=name,
            )
        finally:
            await store.close()
        if not rows:
            return {"error": f"Entity not found: {name}"}
        return {k: v for k, v in rows[0].items() if not k.startswith("@")}

    async def ltm_entities(self, session_id: str = "", limit: int = 200) -> dict[str, Any]:
        """List ENTITY nodes from the golden graph, optionally scoped to one session."""
        check_scope_id(session_id, "session_id")
        check_limit(limit)
        from graphknows.storage import build_graph_store

        store = build_graph_store(self.settings, self.namespace)
        await store.connect()
        try:
            if session_id:
                rows = await store.query(
                    "MATCH (c:CHUNK)-[:MENTIONS]->(e:ENTITY) WHERE c.session_id = $sid "
                    "RETURN DISTINCT e.name AS name, coalesce(e.type,'ENTITY') AS type, "
                    "coalesce(e.confidence, 1.0) AS confidence, "
                    "coalesce(e.pagerank, 0.0) AS pagerank, "
                    "coalesce(e.community_id, -1) AS community_id "
                    "LIMIT $limit",
                    sid=session_id,
                    limit=limit,
                )
            else:
                rows = await store.query(
                    "MATCH (e:ENTITY) "
                    "RETURN e.name AS name, coalesce(e.type,'ENTITY') AS type, "
                    "coalesce(e.confidence, 1.0) AS confidence, "
                    "coalesce(e.pagerank, 0.0) AS pagerank, "
                    "coalesce(e.community_id, -1) AS community_id "
                    "ORDER BY e.name LIMIT $limit",
                    limit=limit,
                )
        finally:
            await store.close()
        entities = [{k: v for k, v in r.items() if not k.startswith("@")} for r in rows]
        return {"entities": entities, "count": len(entities), "session_id": session_id}
