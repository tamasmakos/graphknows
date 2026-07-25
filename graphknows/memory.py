"""Runtime facade for graphknows memory operations.

This module is the transport-neutral seam for MCP, the Memory Mirror web UI,
and in-process callers that need memory operations without learning store
lifecycles, background-loop details, or promotion ordering.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from graphknows.ingestion.parsers.base import strip_content_heading
from graphknows.settings import GraphKnowsSettings

logger = logging.getLogger(__name__)


def _format_fused_hits(
    fused_results: list[dict[str, Any]], session_id: str
) -> list[dict[str, Any]]:
    return [
        {
            "score": r.get("rrf_score", 0.0),
            "cosine_score": r.get("rrf_score", 0.0),
            "type": r.get("sources", ""),
            "sources": r.get("sources", ""),
            "text": strip_content_heading(r.get("text", "")),
            "session_id": session_id,
            "entities": r.get("entities", []),
            "metadata": r.get("metadata", {}),
        }
        for r in fused_results
    ]


class Memory:
    """The graphknows memory facade — the primary developer-facing API.

    Transport-neutral: the MCP server and the Memory Mirror web UI are thin
    wrappers over this class, so every transport sees identical behaviour. One
    ``Memory`` instance is safe for concurrent asyncio use.
    """

    def __init__(
        self, settings: GraphKnowsSettings | None = None, namespace: str | None = None
    ) -> None:
        self.settings = settings or GraphKnowsSettings()
        # The namespace selects one physical ArcadeDB database (mem_<ns>, "" -> mem,
        # the golden layer). Explicit arg wins over the settings default.
        self.namespace = self.settings.namespace if namespace is None else namespace
        # One retriever per strategy, all sharing the single connected store, so
        # a caller can ask for deterministic or agentic retrieval per call
        # without a second connection or a server restart.
        self._retrievers: dict[str, Any] = {}
        self._retriever_stores: list[Any] = []
        self._connect_lock = asyncio.Lock()

    async def __aenter__(self) -> Memory:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close any store connections held by this runtime instance."""
        import contextlib

        for store in self._retriever_stores:
            with contextlib.suppress(Exception):
                await store.close()
        self._retrievers = {}
        self._retriever_stores = []

    async def connect(self) -> None:
        """Eagerly open store connections on the caller's event loop.

        Optional: recall/ingest connect lazily on first use. Call this to pay the
        connection cost up front (e.g. at app startup) or to fail fast.
        """
        await self._get_retriever()

    def _resolve_strategy(self, strategy: str = "") -> str:
        """Strategy name for the cache key: explicit arg, else the configured default."""
        from graphknows.settings import RetrievalStrategy

        return (RetrievalStrategy(strategy) if strategy else self.settings.retrieval_strategy).value

    async def _get_retriever(self, strategy: str = "") -> Any:
        """Return a cached retriever, connecting the store lazily on first call.

        Guarded by an ``asyncio.Lock`` so concurrent first-use callers connect
        the store exactly once. A second strategy reuses that same store.
        """
        key = self._resolve_strategy(strategy)
        if (cached := self._retrievers.get(key)) is not None:
            return cached
        async with self._connect_lock:
            if (cached := self._retrievers.get(key)) is not None:  # another coroutine won
                return cached
            return await self._build_retriever(key)

    async def _build_retriever(self, strategy: str) -> Any:
        from graphknows.retrieval import build_retriever
        from graphknows.storage import build_graph_store
        from graphknows.storage.embedder import embed_dim

        s = self.settings
        if self._retriever_stores:
            store = self._retriever_stores[0]  # already connected for another strategy
        else:
            store = build_graph_store(s, self.namespace)
            # Record the store before connecting so a failure during connect or
            # ensure_schema is cleaned up by close() instead of orphaning pools.
            self._retriever_stores = [store]
            try:
                await store.connect()
                dims = s.embed_dimensions or embed_dim()
                await store.ensure_schema(dims)
            except Exception:
                await self.close()
                raise
        # The strategy fork (deterministic vs agentic) lives in build_retriever.
        retriever = build_retriever(s, store=store, strategy=strategy)
        self._retrievers[strategy] = retriever
        return retriever

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
            result["arcadedb"] = str(exc)
        result["pid"] = os.getpid()
        return result

    async def stats(self, session_id: str = "") -> dict[str, Any]:
        """Return golden-layer memory statistics for this runtime's namespace.

        Counts come from the single database (``mem``/``mem_<ns>``); the
        ``raw``/``consolidated`` split is reported per lifecycle ``state``.
        """
        from graphknows.storage import build_arcadedb_client
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

            def _n(rows: list[dict[str, Any]]) -> int:
                return int(rows[0].get("c") or 0) if rows else 0

            return {
                "sessions": _n(sessions),
                "entities": _n(entities),
                "topics": _n(topics),
                "chunks": {"raw": _n(raw_chunks), "consolidated": _n(cons_chunks)},
            }
        finally:
            await c.close()

    def mirror(self, session_id: str = "", scope: str = "ltm") -> dict[str, Any]:
        """Return Memory Mirror URL metadata."""
        from graphknows.links import mirror_url

        selected_scope = scope if scope in {"stm", "ltm"} else "ltm"
        return {
            "url": mirror_url(session_id=session_id, scope=selected_scope),
            "session_id": session_id,
            "scope": selected_scope,
        }

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
        memory_class: str = "raw_evidence",
        metadata: dict[str, Any] | None = None,
        text: str = "",
    ) -> dict[str, Any]:
        """Ingest text or messages into short-term memory."""
        if text and not messages:
            messages = text

        from graphknows.ingestion import build_stm_service
        from graphknows.models.ingest_options import IngestOptions
        from graphknows.models.memory_class import MemoryClass
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

        service = build_stm_service(self.settings, self.namespace)
        try:
            # connect() inside the try so a failure after the pool opens (schema
            # DDL, transient ArcadeDB error) still runs close() in the finally.
            await service.connect()
            # Only structured conversations go through the turn-based path. A plain
            # string (even without a title) must go through the text/chunk path —
            # the turn path creates no CHUNK nodes, leaving retrieval-over-chunks
            # empty for bare-string ingests.
            if is_conversation:
                try:
                    resolved_class = MemoryClass(memory_class)
                except ValueError:
                    resolved_class = MemoryClass.raw_evidence
                options = IngestOptions(
                    scope=scope,
                    metadata=metadata or {},
                    infer=infer,
                    memory_class=resolved_class,
                )
                result = await service.ingest_messages(normalize_messages(messages), options)
                return {
                    "file_id": result.file_id,
                    "chunks": result.chunk_count,
                    "entities": result.entity_count,
                    "elapsed_s": round(result.elapsed_s, 3),
                    "errors": list(result.errors),
                }

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
            }
        finally:
            await service.close()

    async def recall_memory(
        self,
        query: str,
        *,
        session_id: str = "",
        top_k: int = 5,
        scope: str = "stm",
        strategy: str = "",
    ) -> dict[str, Any]:
        """Recall ranked STM/LTM hits using the compatibility result shape.

        ``strategy`` selects ``det`` or ``agentic`` for this call; empty uses the
        configured default. It is independent of how the graph was ingested.
        """
        top_k = max(1, min(top_k, 25))
        if scope not in {"stm", "ltm", "both"}:
            scope = "both"
        retriever = await self._get_retriever(strategy)
        ctx = await retriever.retrieve(query, session_id=session_id, top_k=top_k, scope=scope)
        return {
            "hits": _format_fused_hits(ctx.fused_results, session_id),
            "facts": list(getattr(ctx, "facts", []) or []),
        }

    def _topic_summarizer(self) -> Any:
        """The LLM summarizer for the configured topic mode."""
        return self._topic_summarizer_for(self.settings.topic_mode)

    def _topic_summarizer_for(self, mode: Any) -> Any:
        """The LLM summarizer for ``mode``; ``None`` for every non-llm mode.

        Returning ``None`` is what keeps the extractive topic layer free of any
        network dependency, so it is valid under ``llm_free``.
        """
        from graphknows.settings import TopicMode

        if mode is not TopicMode.llm:
            return None
        from graphknows.ingestion.summarize import TopicSummarizer

        s = self.settings
        return TopicSummarizer(
            model=s.summarization_model or s.llm_model,
            api_key=s.llm_api_key,
            api_base=s.llm_api_base,
        )

    async def _flush_topics(self, store: Any) -> dict[str, Any]:
        """The topic stage as invoked by a flush, using the configured defaults.

        Returns ``{"topics_created": int, "modularity": float | None}``. The
        modularity is the topic stage's own Newman modularity over the entity
        graph (``TopicSet.metrics["newman_modularity"]``) — the real, trusted
        number, now that ArcadeDB's ``modularityScore``-derived reporting
        (``compute_community_quality``) has been retired.
        """
        from graphknows.settings import TopicMode
        from graphknows.topics import TopicConfig, build_topics, persist_topics

        mode = self.settings.topic_mode
        if mode is TopicMode.none:
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
            **overrides: Field overrides applied to ``config`` (e.g.
                ``distance=0.25``, ``min_size=3``, ``n2v_walk_length=40``).

        Returns:
            The build's metrics, plus ``persisted`` and ``export_path``.
        """
        from dataclasses import replace

        from graphknows.topics import TopicConfig, build_topics, export_json, persist_topics

        cfg = config or TopicConfig(mode=self.settings.topic_mode)
        if overrides:
            cfg = replace(cfg, **overrides)

        await self._get_retriever()  # connects the store + ensures schema
        store = self._retriever_stores[0]
        summarizer = self._topic_summarizer_for(cfg.mode)
        topic_set = await build_topics(store, cfg, summarizer, namespace=self.namespace)

        persisted = await persist_topics(store, topic_set) if persist else 0
        out: dict[str, Any] = {
            **topic_set.metrics,
            "persisted": persisted,
            "config_fingerprint": cfg.fingerprint(),
        }
        if export_path:
            out["export_path"] = str(export_json(topic_set, export_path))
        return out

    async def flush_memory(self, session_id: str) -> dict[str, Any]:
        """Consolidate a session in place.

        The full STM->LTM promotion pipeline is retired in the golden layer.
        This resolves duplicate/junk ENTITY nodes (layered resolver: junk-hub
        filter, name-standardization merge, person-nickname merge), then
        recomputes PageRank/node2vec embeddings/communities over the *merged*
        entity graph (native ArcadeDB algos, each with its own soft-fail —
        see ``GraphStore.compute_and_persist_pagerank`` et al.), builds the
        TOPIC layer over those communities when ``GRAPHKNOWS_TOPICS`` asks for
        it, and flips the session's lifecycle ``state`` raw -> consolidated in
        place. Nothing else is deleted; relation canonicalize lands later.
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
        modularity: float | None = None
        resolution = {"dropped": 0, "merged": 0}

        async def step(name: str, coro: Any, default: Any) -> Any:
            """Await one pipeline step, recording a named error instead of aborting.

            Each step is independently recoverable: an analytics failure must not
            skip ``consolidate_session``, which is the state flip that makes the
            flush meaningful. The step name is kept in the error so a caller can
            tell which stage degraded.
            """
            try:
                return await coro
            except Exception as exc:
                logger.warning("flush_memory: %s failed: %s", name, exc)
                errors.append(f"{name}: {exc}")
                return default

        try:
            await self._get_retriever()  # ensures the store is connected + schema'd
            store = self._retriever_stores[0]
        except Exception as exc:
            # Without a connected store no step below can run.
            errors.append(f"connect: {exc}")
            store = None

        if store is not None:
            resolution = await step(
                "resolve_entities", resolve_entities(store, session_id), resolution
            )
            # Rebuild the analytical view once and share it across all three
            # algos below (each is skip_view_ensure=True) — rebuilding is not
            # free. A failure here is non-fatal and deliberately not recorded in
            # errors: each algo call still runs and handles its own failure
            # (pagerank falls back to numpy; the other two soft-fail to 0), it
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

        return {
            "nodes": pagerank_updated,
            "edges": 0,
            "chunks": chunks_flipped,
            "turns": 0,
            "topics_created": topics_created,
            "entities_dropped": resolution["dropped"],
            "merged_entities": resolution["merged"],
            "graph_embeddings_computed": graph_embeddings_computed,
            "communities_computed": communities_computed,
            "modularity": modularity,
            "communities_scored": topics_created,
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
        memory_class: str = "raw_evidence",
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
            memory_class=memory_class,
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
        strategy: str = "",
    ) -> dict[str, Any]:
        """Search memory scoped to a user/agent/run (or an explicit session)."""
        sid = self._scoped_session_id(user_id, agent_id, run_id, session_id)
        return await self.recall_memory(
            query, session_id=sid, top_k=top_k, scope=scope, strategy=strategy
        )

    async def flush(
        self,
        *,
        user_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        """Promote a user/agent/run scope (or an explicit session) from STM to LTM."""
        sid = self._scoped_session_id(user_id, agent_id, run_id, session_id)
        return await self.flush_memory(sid)

    async def purge_memory(self, scope: str = "both", session_id: str = "") -> dict[str, Any]:
        """Purge memory for this namespace (golden layer, single database).

        ``scope`` is retained for API/deprecation compatibility but no longer
        selects a physical store — there is one database. ``session_id`` deletes
        just that session's subgraph; otherwise all data is cleared (schema and
        indexes are preserved). ``stm_deleted``/``ltm_deleted`` are kept in the
        result for MCP-consumer compatibility and both mirror the single delete.
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
            errors.append(f"arcadedb: {exc}")
        return {
            "scope": scope,
            "session_id": session_id or "(all)",
            "stm_deleted": deleted,
            "ltm_deleted": deleted,
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

    def ltm_query(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Return topic vector-search matches for research evaluation callers."""
        from graphknows._async_bridge import _run as bg_run
        from graphknows.storage import build_graph_store
        from graphknows.storage.embedder import embed_one

        k = max(1, min(top_k, 20))

        async def _search() -> list[dict[str, Any]]:
            try:
                embedding = embed_one(query)
            except Exception:
                logger.debug("ltm_query: embedding failed for query=%r", query, exc_info=True)
                return []
            store = build_graph_store(self.settings, self.namespace)
            await store.connect()
            try:
                return await store.search_topics(embedding, top_k=k)
            finally:
                await store.close()

        try:
            results = bg_run(_search())
        except Exception:
            logger.debug("ltm_query: search failed for query=%r", query, exc_info=True)
            results = []
        return {"results": results[:k]}

    async def ltm_entities(self, session_id: str = "", limit: int = 200) -> dict[str, Any]:
        """List ENTITY nodes from the golden graph, optionally scoped to one session."""
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
