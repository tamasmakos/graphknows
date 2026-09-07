"""Runtime facade for graphknows memory operations.

This module is the transport-neutral seam for MCP and in-process callers that
need memory operations without learning store lifecycles, background-loop
details, or promotion ordering.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from collections import defaultdict
from typing import Any

from graphknows.bounds import (
    bounded,
    check_limit,
    check_metadata_size,
    check_scope_id,
    check_text_length,
    check_top_k,
)
from graphknows.exceptions import ConfigurationError
from graphknows.ingestion.parsers.text import strip_content_heading
from graphknows.models.hit import Hit
from graphknows.settings import Decoder, GraphKnowsSettings, MemoryMode, get_settings

logger = logging.getLogger(__name__)


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


def _redact_store_error(exc: Exception, *, where: str) -> str:
    """Log ``exc`` in full and return a caller-safe summary with a correlation id.

    ArcadeDB failures carry the request host and the raw server response body
    (see ``ArcadeDBClient._raise_for_status``), and ``doctor``, ``purge_memory``
    and ``flush_memory`` hand their errors straight to an MCP caller — that
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


_WORD_RE = re.compile(r"[a-z0-9']+")


def _format_stm_hits(
    turns: list[dict[str, Any]], query: str, top_k: int, session_id: str
) -> list[Hit]:
    """Rank buffered TURN rows by cheap word-overlap against ``query``.

    Deliberately no embeddings: the STM buffer is meant to stay free (no
    extraction, no vectors) until ``flush`` runs it through the real pipeline.
    Ranked by (overlap desc, seq desc) so ties favour the most recent turn;
    zero-overlap turns are dropped entirely.
    """
    q_terms = set(_WORD_RE.findall(query.lower()))
    if not q_terms:
        return []
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for t in turns:
        words = set(_WORD_RE.findall(str(t.get("text", "")).lower()))
        overlap = len(q_terms & words)
        if overlap:
            scored.append((overlap, int(t.get("seq", 0) or 0), t))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [
        Hit(
            text=strip_content_heading(str(t.get("text", ""))),
            speaker=str(t.get("speaker", "") or ""),
            ts=str(t.get("ts", "") or ""),
            score=float(overlap),
            sources="stm",
            session_id=session_id,
            chunk_id=str(t.get("turn_id", "") or ""),
            metadata={"role": t.get("role", ""), "seq": t.get("seq", 0)},
        )
        for overlap, _seq, t in scored[:top_k]
    ]


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
        # One retriever over one connected store, built lazily on first use.
        self._retriever: Any = None
        self._store: Any = None
        self._connect_lock = asyncio.Lock()
        # One lock per session: allocating the next TURN sequence and writing at
        # it is a read-then-write, so two concurrent add() calls on one session
        # would otherwise read the same max(seq) and overwrite each other's
        # turn_id. Per-session, not global, so unrelated sessions still ingest
        # in parallel.
        # ponytail: in-process only — a second worker on the same session needs
        # a store-side sequence.
        self._turn_seq_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

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
                raw_turns = await c.query(
                    db,
                    "MATCH (n:TURN {session_id: $sid}) WHERE n.state = 'raw' RETURN count(n) AS c",
                    params={"sid": sid},
                )
                cons_turns = await c.query(
                    db,
                    "MATCH (n:TURN {session_id: $sid}) WHERE n.state = 'consolidated' "
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
                raw_turns = await c.query(
                    db, "MATCH (n:TURN) WHERE n.state = 'raw' RETURN count(n) AS c"
                )
                cons_turns = await c.query(
                    db, "MATCH (n:TURN) WHERE n.state = 'consolidated' RETURN count(n) AS c"
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
                "turns": {"raw": _n(raw_turns), "consolidated": _n(cons_turns)},
                # SOURCE rows: which (channel, version) this namespace holds —
                # the reader that keeps SOURCE from being write-only (#145).
                # Wraps the already-connected client rather than opening a
                # second connection just to reuse GraphStore.sources().
                "sources": await GraphStore(client=c, db=db).sources(),
            }
        finally:
            await c.close()

    async def redecode(self, session_id: str) -> dict[str, Any]:
        """Re-run the decoder over this session's decode-failed chunks (FR-024).

        Returns ``{"session_id", "attempted", "recovered", "still_failed",
        "abstentions", "errors"}``. Only chunks marked ``decode_state='failed'``
        are touched, and the text already in the graph is the input — the source
        is never re-ingested. A chunk that decodes loses the marker, one that
        fails again keeps it, so a second run finds nothing and reports
        ``attempted: 0``.

        Deliberately not folded into :meth:`flush_memory`: flush is the
        raw -> consolidated lifecycle, and a re-drive is not a lifecycle
        transition.
        """
        from graphknows.ingestion import build_stm_service

        service = build_stm_service(self.settings, self.namespace)
        try:
            await service.connect()
            return await service.redecode(session_id)  # type: ignore[no-any-return]
        finally:
            await service.close()

    @bounded
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
        """Ingest text or messages into memory — two DIFFERENT lifecycles, same method.

        - **Conversation turns** (a list, or a dict with a ``"role"`` key): land as
          cheap ``TURN`` vertices — no NER, no embedding, no CHUNK nodes. They are
          NOT retrievable via LTM/graph search and NOT counted in ``entities``
          until :meth:`flush_memory` drains the buffer through the real pipeline.
          Only :meth:`recall_memory` with ``scope="stm"``/``"both"`` sees them.
        - **A plain string / document** (no ``"role"`` key): runs the full
          extraction pipeline (GLiNER + embeddings) immediately and is graph-
          searchable right away. Unchanged from before — web upload,
          ``corpus_ingest``, and the eval harness depend on this staying eager.
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

        if is_conversation:
            import time

            t0 = time.monotonic()
            msgs = normalize_messages(messages)
            if not msgs:
                return {
                    "file_id": "",
                    "chunks": 0,
                    "entities": 0,
                    "turns": 0,
                    "elapsed_s": round(time.monotonic() - t0, 3),
                    "errors": [],
                }
            await self._get_retriever()  # connects the store + ensures schema
            store = self._store
            sid = scope.to_session_id()
            # Held across the writes, not just the read: next_turn_seq derives
            # the sequence from the turns already written, so releasing before
            # they land would hand the same number to the next caller.
            async with self._turn_seq_locks[sid]:
                await store.ensure_session(sid)
                seq = await store.next_turn_seq(sid)
                for m in msgs:
                    # The message's own timestamp, when it carries one, becomes the
                    # TURN's ts — and flush() feeds it back as the temporal anchor.
                    # Defaulting to write time would date every relative expression
                    # in the turn to whenever the buffer happened to be written.
                    await store.write_turn(
                        sid,
                        m.role.value,
                        m.content,
                        seq,
                        m.timestamp or "",
                        speaker=m.name or "",
                        infer=infer,
                    )
                    seq += 1
            return {
                "file_id": "",
                "chunks": 0,
                "entities": 0,
                "turns": len(msgs),
                "elapsed_s": round(time.monotonic() - t0, 3),
                "errors": [],
            }

        from graphknows.ingestion import build_stm_service

        service = build_stm_service(self.settings, self.namespace)
        try:
            # connect() inside the try so a failure after the pool opens (schema
            # DDL, transient ArcadeDB error) still runs close() in the finally.
            await service.connect()
            result = await service.ingest(
                messages,  # type: ignore[arg-type]
                session_id=scope.to_session_id(),
                title=title,
                metadata=metadata or {},
                infer=infer,
            )
            return {
                "file_id": result.file_id,
                "chunks": result.chunk_count,
                "entities": result.entity_count,
                "elapsed_s": round(result.elapsed_s, 3),
                "errors": list(result.errors),
                "warnings": [],
                "abstentions": result.abstentions,
            }
        finally:
            await service.close()

    @bounded
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

        ``scope`` has real meaning: ``"stm"`` searches ONLY the session's
        buffered ``TURN`` rows (cheap word-overlap, no embeddings — see
        :func:`_format_stm_hits`); ``"ltm"`` searches ONLY the graph via the
        configured retriever; ``"both"`` returns STM hits first, then graph
        hits, truncated to ``top_k`` total.

        It defaults to ``"both"`` — the same default as :meth:`search`, the
        LangGraph nodes and the MCP tool. It used to default to ``"stm"`` here
        alone, which means a caller who omitted it searched only the unflushed
        buffer: empty immediately after a flush, and empty for every session
        that ingested through any other one. Four surfaces, one default.

        ``cross_session``, when True, makes the GRAPH half namespace-wide
        (``session_id=""`` to the retriever, which treats that as "no session
        filter"). The STM buffer stays session-scoped regardless — a caller
        cannot cross-session-read another session's unflushed turns. Defaults
        to False so the MCP server and web UI keep today's session-scoped graph
        behaviour unless they opt in.
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
        if scope not in {"stm", "ltm", "both"}:
            scope = "both"

        stm_hits: list[Hit] = []
        graph_hits: list[Hit] = []
        facts: list[Any] = []

        if scope in ("stm", "both"):
            await self._get_retriever()  # connects the store + ensures schema
            store = self._store
            turns = await store.list_turns(session_id, state="raw", limit=30)
            stm_hits = _format_stm_hits(turns, query, top_k, session_id)

        if scope in ("ltm", "both"):
            retriever = await self._get_retriever()
            graph_session_id = "" if cross_session else session_id
            # "both" here means "no chunk-state filter" inside the retriever —
            # the raw/consolidated split is orthogonal now that it no longer
            # doubles as the stm/ltm switch (the TURN buffer is that switch).
            graph_scope = "ltm" if scope == "ltm" else "both"
            ctx = await retriever.retrieve(
                query, session_id=graph_session_id, top_k=top_k, scope=graph_scope
            )
            graph_hits = _format_fused_hits(ctx.fused_results, session_id)
            facts = list(getattr(ctx, "facts", []) or [])

        # ``scope`` is echoed back POST-normalisation so a caller reports what
        # was actually searched rather than what it asked for — and so no
        # transport has to reimplement the fallback rule to say so.
        return {"hits": (stm_hits + graph_hits)[:top_k], "facts": facts, "scope": scope}

    def _topic_summarizer_for(self, mode: Any) -> Any:
        """The LLM summarizer for ``mode``; ``None`` for every non-llm mode.

        Returning ``None`` is what keeps the extractive topic layer free of any
        network dependency, so it is valid under ``llm_free``.
        """
        from graphknows.settings import TopicMode

        if mode is not TopicMode.llm:
            return None
        from graphknows.topics.summarize import TopicSummarizer

        s = self.settings
        return TopicSummarizer(
            model=s.summarization_model or s.llm_model,
            api_key=s.llm_api_key.get_secret_value(),
            api_base=s.llm_api_base,
        )

    async def _flush_topics(self, store: Any) -> dict[str, Any]:
        """The topic stage as invoked by a flush — opt-in only (FR-045).

        Returns ``{"topics_created": int, "modularity": float | None}``. The
        modularity is the topic stage's own Newman modularity over the entity
        graph (``TopicSet.metrics["newman_modularity"]``) — the real, trusted
        number, now that ArcadeDB's ``modularityScore``-derived reporting
        (``compute_community_quality``) has been retired.

        Topics have exactly one reader, the browse tool (``ltm_topics``), so a
        flush no longer pays clustering + description just because
        ``GRAPHKNOWS_MODE=llm_assisted`` prefers ``TopicMode.llm`` as a
        preset — that read ``self.settings.topic_mode``, the preset-resolved
        value. Here it reads the raw ``topics`` field instead: unset means no
        one asked, same as an explicit ``none``. A caller who wants topics
        sets ``GRAPHKNOWS_TOPICS`` explicitly, or rebuilds on demand through
        :meth:`build_topics` / the ``memory_topics`` tool, which do keep
        following the preset — calling either one is itself the opt-in.
        """
        from graphknows.settings import TopicMode
        from graphknows.topics import TopicConfig, build_topics, persist_topics

        mode = self.settings.topics
        if mode is None or mode is TopicMode.none:
            return {"topics_created": 0, "modularity": None}
        cfg = TopicConfig(mode=mode)
        topic_set = await build_topics(
            store, cfg, self._topic_summarizer_for(mode), namespace=self.namespace
        )
        persisted = await persist_topics(store, topic_set)
        return {
            "topics_created": persisted,
            "modularity": topic_set.metrics.get("newman_modularity"),
        }

    async def build_topics(
        self,
        *,
        config: Any = None,
        persist: bool = True,
        export_path: str = "",
        include_export: bool = False,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Rebuild the topic layer over the existing graph. Returns the metrics.

        Standalone by design: this re-runs clustering, description and grading
        **without** touching ingestion, entity resolution, PageRank or
        consolidation, so topics can be swept at different settings over an
        unchanged graph.

        Args:
            config: A :class:`~graphknows.topics.TopicConfig`; built from the
                runtime defaults when omitted.
            persist: Write the resulting TOPIC layer (replacing the previous one).
            export_path: Optional path for a read-only JSON artifact.
            include_export: Return the artifact payload in-band under
                ``export`` — the same dict ``export_path`` would have written —
                for callers that cannot be handed a file. Off by default: the
                CLI prints every key it gets back.
            **overrides: Field overrides applied to ``config`` (e.g.
                ``min_size=3``).

        Returns:
            The build's metrics, plus ``persisted``, and ``export`` or
            ``export_path`` when either was asked for.
        """
        from dataclasses import replace

        from graphknows.topics import TopicConfig, build_topics, export_json, persist_topics

        cfg = config or TopicConfig(mode=self.settings.topic_mode)
        if overrides:
            cfg = replace(cfg, **overrides)

        await self._get_retriever()  # connects the store + ensures schema
        store = self._store
        summarizer = self._topic_summarizer_for(cfg.mode)
        topic_set = await build_topics(store, cfg, summarizer, namespace=self.namespace)

        persisted = await persist_topics(store, topic_set) if persist else 0
        out: dict[str, Any] = {
            **topic_set.metrics,
            "persisted": persisted,
            "config_fingerprint": cfg.fingerprint(),
        }
        if include_export:
            out["export"] = topic_set.to_dict()
        if export_path:
            out["export_path"] = str(export_json(topic_set, export_path))
        return out

    async def _ingest_buffered_turns(
        self, store: Any, session_id: str, raw_turns: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Run the session's buffered ``TURN`` rows through the real STM pipeline.

        This is the expensive step (GLiNER extraction + embeddings) that the
        buffer exists to defer. ``raw_turns`` is newest-first (as returned by
        ``list_turns``); it is reversed to oldest-first so seq order in the
        graph matches the order the turns actually happened.
        """
        from graphknows.ingestion import build_stm_service
        from graphknows.models.ingest_options import IngestOptions
        from graphknows.models.message import Message, Role
        from graphknows.models.scope import MemoryScope

        ordered = list(reversed(raw_turns))
        # Carry each turn's ts through as the Message timestamp: it is the
        # anchor the extractor resolves that turn's relative dates against.
        # Dropping it here is what dated a Jan-Jul 2023 conversation's TEMPORAL
        # nodes to 2026 — 42 of 62 wrong — because every unanchored expression
        # fell back to ingestion time.
        messages = [
            Message(
                role=Role(t["role"]),
                content=t["text"],
                name=t.get("speaker") or None,
                timestamp=t.get("ts") or None,
            )
            for t in ordered
        ]
        # Each turn's own extract/skip decision, made by the caller at add()
        # time. A turn stored with infer=False is still written and embedded —
        # it stays retrievable as text — it just mints no entities.
        infer_flags = [bool(t.get("infer", True)) for t in ordered]
        scope = MemoryScope(run_id=session_id)
        # Continue this dialog file's chunk sequence instead of restarting at 0.
        # Every flush ingests into the same file_id, and neighbor_chunks expands
        # a hit to its siblings by (file_id, seq ± radius) — a second flush
        # numbering from 0 again hands retrieval the first flush's chunks as
        # this one's neighbours. Forward-only: seq values already colliding on
        # disk stay as they are.
        last_seq = await store.max_chunk_seq(scope.to_dialog_id())
        options = IngestOptions(
            scope=scope,
            turn_index_offset=0 if last_seq is None else last_seq + 1,
        )

        service = build_stm_service(self.settings, self.namespace)
        try:
            await service.connect()
            result = await service.ingest_messages(messages, options, infer_flags=infer_flags)
        finally:
            await service.close()
        return {
            "turns": len(messages),
            "chunks": result.chunk_count,
            "abstentions": result.abstentions,
            "errors": list(result.errors),
        }

    async def flush_memory(self, session_id: str) -> dict[str, Any]:
        """Consolidate a session in place.

        Flush now does two things where it used to do one, in order:

        1. **Drains the TURN buffer**: every buffered turn for this session is
           run through the real ingestion pipeline (GLiNER extraction +
           embeddings — this is the materially slower call now, not the
           analytics below), producing CHUNK/ENTITY nodes, then the buffer is
           marked ``consolidated`` so it is not drained again.
        2. **Consolidates the graph**: resolves duplicate/junk ENTITY nodes
           (layered resolver: junk-hub filter, name-standardization merge,
           person-nickname merge), then recomputes PageRank/node2vec
           embeddings/communities over the *merged* entity graph (native
           ArcadeDB algos, each with its own soft-fail — see
           ``GraphStore.compute_and_persist_pagerank`` et al.), builds the
           TOPIC layer over those communities when ``GRAPHKNOWS_TOPICS`` asks
           for it, flips the session's CHUNK lifecycle ``state`` raw ->
           consolidated in place, and runs ``GraphStore.consolidate_relations``
           (issue #140) over the merged graph: duplicate REL edges under
           different predicate surfaces collapse with a raised ``support``
           counter, polarity contradictions are linked (not deleted), and a
           functional-property collision tombstones every edge but the newest.

        The result's ``verifier_coverage`` reports, per relation tier, how many
        REL edges carry a relation-verifier score and their mean: the verifier
        only ever sees the tiers routed through its gate, so per-tier coverage
        is what makes an unverified tier visible instead of silent.

        The result's ``entity_coverage`` reports the whole-namespace count of
        ENTITY nodes with no semantic edge (issue #160). Isolation is a
        property of the finished graph, not of one chunk — a relation written
        by a later chunk can connect an entity an earlier chunk minted — so
        flush, not per-chunk ingest, is the only place this can be measured.
        """
        import time

        from graphknows.ingestion.consolidation import resolve_entities

        t0 = time.monotonic()
        errors: list[str] = []
        pagerank_updated = 0
        graph_embeddings_computed = 0
        communities_computed = 0
        chunks_flipped = 0
        topics_created = 0
        turns_ingested = 0
        turn_chunks = 0
        abstentions: dict[str, int] = {}
        modularity: float | None = None
        resolution = {"dropped": 0, "merged": 0, "considered": 0}
        rel_consolidation = {"deduped": 0, "contradictions": 0, "invalidated": 0}
        verifier_coverage: dict[str, dict[str, Any]] = {}
        entity_coverage: dict[str, int] = {}
        # Seconds per named stage, so the slow one is identifiable without a profiler.
        stage_s: dict[str, float] = {}

        async def step(name: str, coro: Any, default: Any) -> Any:
            """Await one pipeline step, recording a named error instead of aborting.

            Each step is independently recoverable: an analytics failure must not
            skip ``consolidate_session``, which is the state flip that makes the
            flush meaningful. The step name is kept in the error so a caller can
            tell which stage degraded.

            Timed, because ``flush_elapsed_s`` covers drain, resolution, pagerank,
            node2vec, louvain and topics together — a 50-minute flush said nothing
            about WHICH of those took the 50 minutes, so every conversation about
            ingest cost started by guessing. A failed step is timed too: how long
            a stage took before it fell over is exactly what you want to know.
            """
            started = time.monotonic()
            try:
                return await coro
            except Exception as exc:
                errors.append(f"{name}: {_redact_store_error(exc, where=f'flush_memory:{name}')}")
                return default
            finally:
                stage_s[name] = round(time.monotonic() - started, 2)

        try:
            await self._get_retriever()  # ensures the store is connected + schema'd
            store = self._store
        except Exception as exc:
            # Without a connected store no step below can run.
            errors.append(f"connect: {_redact_store_error(exc, where='flush_memory:connect')}")
            store = None

        if store is not None:
            raw_turns = await step(
                "list_turns", store.list_turns(session_id, state="raw", limit=10000), []
            )
            if raw_turns:
                drained = await step(
                    "drain_turns",
                    self._ingest_buffered_turns(store, session_id, raw_turns),
                    {"turns": 0, "chunks": 0, "abstentions": {}, "errors": []},
                )
                turns_ingested = drained["turns"]
                turn_chunks = drained["chunks"]
                abstentions = drained.get("abstentions") or {}
                # Per-turn extraction failures are caught inside STMService and
                # never raise, so `step()` above sees a clean return and would
                # otherwise hide them — surface them here so a graph that failed
                # on every write is not indistinguishable from an empty one.
                for turn_error in drained.get("errors") or []:
                    errors.append(f"drain_turns: {turn_error}")
            await step("mark_turns_consolidated", store.mark_turns_consolidated(session_id), 0)

            # Caller-supplied structure goes in HERE: the buffer has drained, so
            # this session's CHUNK vertices exist and custom edges have
            # something to attach to; the analytics below have not run, so
            # entity resolution and the topic build see the finished graph.
            # Each channel soft-fails under its own name — one broken extension
            # must not cost the consolidation the rest of the flush does.
            #
            # Settings-gated defaults (frames/ontology) go through the same
            # loop as caller-supplied channels: default_channels() is also
            # what build_retriever() uses for the read side (see
            # ``_build_retriever``), so a channel registered there is active
            # on both paths from one registration. None of the defaults
            # implements ``populate`` today; they inherit the no-op default,
            # so this loop adds no write work for them yet.
            from graphknows.channels.registry import default_channels

            for channel in [*default_channels(self.settings), *self._channels]:
                await step(
                    f"populate:{getattr(channel, 'name', type(channel).__name__)}",
                    channel.populate(store, session_id),
                    None,
                )

            resolution = await step(
                "resolve_entities", resolve_entities(store, session_id), resolution
            )
            # Entity merges above replay REL edges through write_relation and
            # can themselves mint duplicates, so consolidation must run after
            # resolution has settled, over the already-merged graph.
            rel_consolidation = await step(
                "consolidate_relations",
                store.consolidate_relations(),
                rel_consolidation,
            )
            # Rebuild the analytical view once and share it across all three
            # algos below (each is skip_view_ensure=True) — rebuilding is not
            # free. A failure here is non-fatal and deliberately not recorded in
            # errors: each algo call still runs and handles its own failure
            # (all three soft-fail to 0 with a log line), it
            # just retries the (already-failed) view.
            try:
                await store._ensure_analytical_view()
            except Exception as exc:
                logger.warning("flush_memory: analytical view ensure failed: %s", exc)
            pagerank_updated = await step(
                "pagerank", store.compute_and_persist_pagerank(skip_view_ensure=True), 0
            )
            graph_embeddings_computed = await step(
                "graph_embeddings", store.compute_graph_embeddings(skip_view_ensure=True), 0
            )
            communities_computed = await step(
                "communities", store.compute_communities(skip_view_ensure=True), 0
            )
            # Topics run last of the analytics: the stage clusters over the
            # node2vec embeddings and pagerank computed above. It is the SAME
            # stage as Memory.build_topics — one implementation, so a rebuild at
            # new settings is never a different code path from the flush. The
            # topic build's own Newman modularity is the flush's reported
            # "modularity" — ArcadeDB's untrustworthy modularityScore is gone.
            topic_result = await step(
                "topics", self._flush_topics(store), {"topics_created": 0, "modularity": None}
            )
            topics_created = topic_result["topics_created"]
            modularity = topic_result["modularity"]
            chunks_flipped = await step(
                "consolidate_session", store.consolidate_session(session_id), 0
            )
            verifier_coverage = await step("verifier_coverage", store.verifier_coverage(), {})
            entity_coverage = await step("entity_coverage", store.entity_coverage(), {})

        return {
            "nodes": pagerank_updated,
            "edges": 0,
            "chunks": chunks_flipped,
            "turns": turns_ingested,
            "turn_chunks": turn_chunks,
            # What the write path declined to write, per gate. Reported so an
            # empty graph can be told apart from a graph whose gates ate it.
            "abstentions": abstentions,
            # Seconds per stage. `elapsed_s` is their sum plus the untimed glue.
            "stage_s": stage_s,
            "topics_created": topics_created,
            "entities_dropped": resolution["dropped"],
            "merged_entities": resolution["merged"],
            # merges / entities considered for resolution, so a bare count is
            # never read without the population it's a fraction of.
            "merge_rate": (
                round(resolution["merged"] / considered, 4)
                if (considered := resolution.get("considered"))
                else 0.0
            ),
            "graph_embeddings_computed": graph_embeddings_computed,
            "communities_computed": communities_computed,
            "modularity": modularity,
            "communities_scored": topics_created,
            "rel_consolidation": rel_consolidation,
            "verifier_coverage": verifier_coverage,
            # A graph whose entities are all isolated and a graph with no
            # entities look identical to every other number in this report.
            "entity_coverage": entity_coverage,
            "elapsed_s": round(time.monotonic() - t0, 3),
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

    async def flush(
        self,
        *,
        user_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """Promote a user/agent/run scope (or an explicit session) from STM to LTM.

        Ingests the session's buffered ``TURN`` rows into the graph (real
        extraction + embeddings), marks them consolidated, then runs the usual
        entity-resolution/analytics consolidation pass. See
        :meth:`flush_memory` for the full step order.
        """
        sid = self._scoped_session_id(user_id, agent_id, run_id, session_id)
        return await self.flush_memory(sid)

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
                    await store.delete_session(session_id)
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
        """Ingest a batch of documents into one session, then consolidate it.

        Runs the same golden-layer path as ``ingest_memory`` + ``flush_memory``.
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

        flushed = await self.flush_memory(sid)
        errors.extend(flushed.get("errors", []) or [])

        return {
            "session_id": sid,
            "documents_processed": processed,
            "entities_extracted": entities,
            "chunks_written": chunks,
            "merged_entities": flushed.get("merged_entities", 0),
            "chunks_consolidated": flushed.get("chunks", 0),
            "elapsed_s": round(time.monotonic() - t0, 3),
            "errors": errors,
        }

    async def ltm_topics(self, min_entities: int = 3) -> dict[str, Any]:
        """List golden-layer TOPIC nodes with their entity membership.

        Entity membership is the direct ``ENTITY-[:IN_TOPIC]->TOPIC`` edge — the
        actual output of the topic clustering, written by ``write_topics``.
        """
        from graphknows.storage import build_graph_store

        store = build_graph_store(self.settings, self.namespace)
        await store.connect()
        try:
            rows = await store.query(
                "MATCH (e:ENTITY)-[:IN_TOPIC]->(t:TOPIC) "
                "WITH t, count(DISTINCT e) AS entity_count, "
                "collect(DISTINCT e.name) AS entity_names "
                "WHERE entity_count >= $min "
                "RETURN t.topic_id AS topic_id, t.title AS title, t.summary AS summary, "
                "t.keywords AS keywords, t.coherence AS coherence, "
                "entity_count AS entity_count, entity_names AS entity_names "
                "ORDER BY entity_count DESC",
                min=min_entities,
            )
        finally:
            await store.close()
        topics = []
        for r in rows:
            topic = {k: v for k, v in r.items() if not k.startswith("@")}
            names = topic.get("entity_names") or []
            topic["entity_names"] = list(names)[:20]
            topics.append(topic)
        return {"topics": topics}

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

    @bounded
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
