"""Runtime facade for graphknows memory operations.

This module is the transport-neutral seam for MCP, the Memory Mirror web UI,
and in-process callers that need memory operations without learning store
lifecycles, background-loop details, or promotion ordering.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from graphknows.settings import GraphKnowsSettings

_STM_RECALL_WEIGHT: float = 0.6
_LTM_RECALL_WEIGHT: float = 0.4
assert _STM_RECALL_WEIGHT + _LTM_RECALL_WEIGHT == 1.0


def normalise_and_weight(hits: list[Any], weight: float) -> None:
    """Min-max normalise scores in-place, then apply a scope weight."""
    if not hits:
        return
    scores = [h.score for h in hits]
    mn, mx = min(scores), max(scores)
    rng = mx - mn
    for hit in hits:
        hit.score = weight * ((hit.score - mn) / rng if rng else 1.0)


def _clean_recall_text(text: str) -> str:
    return re.sub(r"^##\s*Content\s*\n+", "", text, flags=re.IGNORECASE).strip()


def _format_fused_hits(
    fused_results: list[dict[str, Any]], session_id: str
) -> list[dict[str, Any]]:
    return [
        {
            "score": r.get("rrf_score", 0.0),
            "cosine_score": r.get("rrf_score", 0.0),
            "type": r.get("sources", ""),
            "sources": r.get("sources", ""),
            "text": _clean_recall_text(r.get("text", "")),
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
        # The namespace selects a physical ArcadeDB database pair (stm_<ns>/ltm_<ns>);
        # "" keeps the original stm/ltm. Explicit arg wins over the settings default.
        self.namespace = self.settings.namespace if namespace is None else namespace
        self._retriever: Any = None
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
        self._retriever = None
        self._retriever_stores = []

    async def connect(self) -> None:
        """Eagerly open store connections on the caller's event loop.

        Optional: recall/ingest connect lazily on first use. Call this to pay the
        connection cost up front (e.g. at app startup) or to fail fast.
        """
        await self._get_retriever()

    async def _get_retriever(self) -> Any:
        """Return a cached retriever, connecting stores lazily on first call.

        Guarded by an ``asyncio.Lock`` so concurrent first-use callers connect
        the stores exactly once.
        """
        if self._retriever is not None:
            return self._retriever
        async with self._connect_lock:
            if self._retriever is not None:  # another coroutine won the race
                return self._retriever
            return await self._build_retriever()

    async def _build_retriever(self) -> Any:
        from graphknows.retrieval import build_retriever
        from graphknows.storage import (
            build_arcadedb_ltm_store,
            build_arcadedb_stm_store,
            build_arcadedb_vector_store,
        )

        s = self.settings
        vec = build_arcadedb_vector_store(s, self.namespace)
        ltm = build_arcadedb_ltm_store(s, self.namespace)
        stm = build_arcadedb_stm_store(s, self.namespace)
        # Record the stores before connecting so a failure during connect or
        # ensure_schema is cleaned up by close() instead of orphaning open pools.
        self._retriever_stores = [vec, ltm, stm]
        try:
            await asyncio.gather(
                vec.connect(),  # vector: connect + ensure_schema
                ltm.connect(),
                stm.connect(),
            )
            await ltm.ensure_schema()
        except Exception:
            await self.close()
            raise
        # The mode fork (deterministic vs agentic) lives in build_retriever.
        self._retriever = build_retriever(s, vec=vec, ltm=ltm, stm=stm)
        return self._retriever

    async def doctor(self) -> dict[str, Any]:
        """Check configured backend connectivity for this runtime's namespace."""
        from graphknows.storage import build_arcadedb_client
        from graphknows._version import get_version
        from graphknows.storage.namespace import db_names

        stm_db, ltm_db = db_names(self.namespace)
        result: dict[str, Any] = {"version": get_version()}
        try:
            c = build_arcadedb_client(self.settings)
            await c.connect()
            result["arcadedb_stm"] = await c.database_exists(stm_db)
            result["arcadedb_ltm"] = await c.database_exists(ltm_db)
            await c.close()
        except Exception as exc:
            result["arcadedb_stm"] = str(exc)
            result["arcadedb_ltm"] = str(exc)
        result["pid"] = os.getpid()
        return result

    async def stats(self, session_id: str = "") -> dict[str, Any]:
        """Return STM and LTM memory statistics for this runtime's namespace."""
        from graphknows.storage import build_arcadedb_client
        from graphknows.storage.namespace import db_names

        stm_db, ltm_db = db_names(self.namespace)
        c = build_arcadedb_client(self.settings)
        await c.connect()
        try:
            if session_id:
                sid = session_id
                stm_chunks = await c.query(
                    stm_db,
                    "MATCH (n:CHUNK {session_id: $sid}) RETURN count(n) AS c",
                    params={"sid": sid},
                )
                stm_turns = await c.query(
                    stm_db,
                    "MATCH (n:TURN {session_id: $sid}) RETURN count(n) AS c",
                    params={"sid": sid},
                )
                ltm_chunks = await c.query(
                    ltm_db,
                    "MATCH (n:CHUNK {session_id: $sid}) RETURN count(n) AS c",
                    params={"sid": sid},
                )
                ltm_turns = await c.query(
                    ltm_db,
                    "MATCH (n:TURN {session_id: $sid}) RETURN count(n) AS c",
                    params={"sid": sid},
                )
            else:
                stm_chunks = await c.query(stm_db, "MATCH (n:CHUNK) RETURN count(n) AS c")
                stm_turns = await c.query(stm_db, "MATCH (n:TURN) RETURN count(n) AS c")
                ltm_chunks = await c.query(ltm_db, "MATCH (n:CHUNK) RETURN count(n) AS c")
                ltm_turns = await c.query(ltm_db, "MATCH (n:TURN) RETURN count(n) AS c")

            ltm_entities = await c.query(ltm_db, "MATCH (n:ENTITY) RETURN count(n) AS c")
            ltm_topics = await c.query(ltm_db, "MATCH (n:TOPIC) RETURN count(n) AS c")
            ltm_sessions = await c.query(ltm_db, "MATCH (n:SESSION) RETURN count(n) AS c")

            def _n(rows: list[dict[str, Any]]) -> int:
                return int(rows[0].get("c") or 0) if rows else 0

            return {
                "stm": {"chunks": _n(stm_chunks), "turns": _n(stm_turns)},
                "ltm": {
                    "sessions": _n(ltm_sessions),
                    "entities": _n(ltm_entities),
                    "topics": _n(ltm_topics),
                    "chunks": _n(ltm_chunks),
                    "turns": _n(ltm_turns),
                },
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
    ) -> dict[str, Any]:
        """Recall ranked STM/LTM hits using the compatibility result shape."""
        top_k = max(1, min(top_k, 25))
        if scope not in {"stm", "ltm", "both"}:
            scope = "both"
        retriever = await self._get_retriever()
        ctx = await retriever.retrieve(query, session_id=session_id, top_k=top_k, scope=scope)
        return {"hits": _format_fused_hits(ctx.fused_results, session_id)}

    async def flush_memory(self, session_id: str) -> dict[str, Any]:
        """Promote a session from STM into LTM."""
        from graphknows.ingestion import build_promotion_pipeline
        from graphknows.ingestion.models.promotion import PromotionRequest
        from graphknows._async_bridge import _run as bg_run

        pipeline = build_promotion_pipeline(self.settings, self.namespace)
        request = PromotionRequest(
            session_id=session_id,
            run_gds=True,
            clear_stm=True,
            merge_entities=True,
        )
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: bg_run(pipeline.promote(request)))
        await loop.run_in_executor(None, lambda: bg_run(pipeline.close()))

        return {
            "nodes": result.nodes_written,
            "edges": result.edges_written,
            "chunks": result.chunks_promoted,
            "turns": result.turns_promoted,
            "topics_created": result.topics_created,
            "merged_entities": result.merged_entities,
            "elapsed_s": round(result.elapsed_s, 3),
            "errors": list(result.errors),
        }

    # ------------------------------------------------------------------
    # Namespace-first SDK surface. ``user_id``/``agent_id``/``run_id`` are the
    # memory scope (mem0-style); at least one is required. These are the methods
    # an application or agent framework calls; the ``*_memory`` verbs above back
    # the MCP + web transports.
    # ------------------------------------------------------------------

    @staticmethod
    def _scoped_session_id(
        user_id: str, agent_id: str, run_id: str, session_id: str
    ) -> str:
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
    ) -> dict[str, Any]:
        """Search memory scoped to a user/agent/run (or an explicit session)."""
        sid = self._scoped_session_id(user_id, agent_id, run_id, session_id)
        return await self.recall_memory(query, session_id=sid, top_k=top_k, scope=scope)

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
        """Purge STM and/or LTM memory."""
        errors: list[str] = []
        stm_deleted = False
        ltm_deleted = False
        if scope in ("stm", "both"):
            await self._purge_stm(session_id, errors)
            stm_deleted = True
        if scope in ("ltm", "both"):
            await self._purge_ltm(session_id, errors)
            ltm_deleted = True
        return {
            "scope": scope,
            "session_id": session_id or "(all)",
            "stm_deleted": stm_deleted,
            "ltm_deleted": ltm_deleted,
            "errors": errors,
        }

    async def drop_namespace(self) -> dict[str, Any]:
        """Drop this runtime's entire namespace (both physical databases).

        An O(1) reset — it removes the ``stm_<ns>``/``ltm_<ns>`` databases outright
        instead of walking and DETACH-DELETE-ing every node. Refuses to drop the
        default (``""``) namespace, whose databases (``stm``/``ltm``) are shared and
        must be cleared with :meth:`purge_memory` instead of dropped.
        """
        from graphknows.storage import build_arcadedb_client
        from graphknows.storage.namespace import db_names

        if not self.namespace:
            return {
                "dropped": False,
                "namespace": "",
                "error": "refusing to drop the default namespace; use purge_memory instead",
            }
        stm_db, ltm_db = db_names(self.namespace)
        await self.close()  # release any cached retriever stores bound to these DBs
        client = build_arcadedb_client(self.settings)
        await client.connect()
        try:
            await client.drop_database(stm_db)
            await client.drop_database(ltm_db)
        finally:
            await client.close()
        return {"dropped": True, "namespace": self.namespace, "databases": [stm_db, ltm_db]}

    async def _purge_stm(self, session_id: str, errors: list[str]) -> None:
        from graphknows.storage import build_arcadedb_stm_store

        try:
            stm_store = build_arcadedb_stm_store(self.settings, self.namespace)
            await stm_store.connect()
            try:
                if session_id:
                    await stm_store.delete_session(session_id)
                else:
                    await stm_store.drop_database()
            finally:
                await stm_store.close()
        except Exception as exc:
            errors.append(f"arcadedb stm: {exc}")

    async def _purge_ltm(self, session_id: str, errors: list[str]) -> None:
        from graphknows.storage import (
            build_arcadedb_ltm_store,
            build_arcadedb_vector_store,
        )

        try:
            ltm_store = build_arcadedb_ltm_store(self.settings, self.namespace)
            await ltm_store.connect()
            try:
                if session_id:
                    await self._delete_ltm_session(ltm_store, session_id)
                else:
                    # Do NOT drop+recreate the database. A freshly recreated DB
                    # leaves the CHUNK(id) UNIQUE index unbuilt for the very next
                    # flush, so MERGE stops deduping and the two flush write-paths
                    # (graph subgraph-write + vector promote) insert duplicate,
                    # half-un-embedded chunk nodes. Keep the schema/indexes and
                    # delete all data instead.
                    await ltm_store.ensure_schema()
                    vec_store = build_arcadedb_vector_store(self.settings, self.namespace)
                    await vec_store.connect()
                    await vec_store.close()
                    await self._clear_ltm_all(ltm_store)
            finally:
                await ltm_store.close()
        except Exception as exc:
            errors.append(f"arcadedb ltm: {exc}")

    @staticmethod
    async def _clear_ltm_all(ltm_store: Any) -> None:
        """Delete all LTM data while preserving schema and indexes."""
        import contextlib

        labels = (
            "CHUNK",
            "TURN",
            "ENTITY",
            "TOPIC",
            "DIALOG",
            "FILE",
            "SESSION",
            "TEMPORAL",
            "SEGMENT",
            "DOC",
            "SESSIONENTITY",
            "FRAME",
        )
        for label in labels:
            with contextlib.suppress(Exception):
                await ltm_store._cmd(f"MATCH (n:{label}) DETACH DELETE n")

    @staticmethod
    async def _delete_ltm_session(ltm_store: Any, session_id: str) -> None:
        await ltm_store._cmd(
            "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(:FILE)-[:HAS_CHUNK]->(c:CHUNK) "
            "DETACH DELETE c",
            sid=session_id,
        )
        await ltm_store._cmd(
            "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE) DETACH DELETE f",
            sid=session_id,
        )
        await ltm_store._cmd(
            "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(:DIALOG)-[:HAS_TURN]->(t:TURN) "
            "DETACH DELETE t",
            sid=session_id,
        )
        await ltm_store._cmd(
            "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(d:DIALOG) DETACH DELETE d",
            sid=session_id,
        )
        await ltm_store._cmd("MATCH (sess:SESSION {id: $sid}) DETACH DELETE sess", sid=session_id)
        await ltm_store._cmd("MATCH (e:ENTITY) WHERE NOT (e)--() DETACH DELETE e")
        await ltm_store._cmd("MATCH (n:FRAME {session_id: $sid}) DETACH DELETE n", sid=session_id)

    async def corpus_ingest(
        self,
        documents: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest a document batch through the corpus workflow."""
        if not documents:
            return {"documents_processed": 0, "error": "no documents provided"}
        try:
            from graphknows.ingestion import build_corpus_workflow
            from graphknows.ingestion.workflows.corpus_ingest import CorpusIngestRequest

            workflow = build_corpus_workflow(self.settings, self.namespace)
            result = await workflow.run(
                CorpusIngestRequest(
                    documents=documents,
                    session_id=session_id,
                    run_gds=True,
                )
            )
        except Exception as exc:
            return {"error": str(exc), "documents_processed": 0}
        return {
            "session_id": result.session_id,
            "documents_processed": result.documents_processed,
            "entities_extracted": result.entities_extracted,
            "chunks_written": result.chunks_written,
            "nodes_written": result.nodes_written,
            "edges_written": result.edges_written,
            "elapsed_s": result.elapsed_s,
            "errors": result.errors,
        }

    async def ltm_topics(self, min_entities: int = 3) -> dict[str, Any]:
        """List LTM topic nodes."""
        from graphknows.storage import build_arcadedb_ltm_store

        ltm = build_arcadedb_ltm_store(self.settings, self.namespace)
        await ltm.connect()
        try:
            rows = await ltm._qry(
                "MATCH (t:TOPIC) WHERE t.entity_count >= $min "
                "RETURN t.topic_id AS topic_id, t.label AS label, t.summary AS summary, "
                "t.community_id AS community_id, t.entity_names AS entity_names, "
                "t.entity_count AS entity_count "
                "ORDER BY t.entity_count DESC",
                min=min_entities,
            )
        finally:
            await ltm.close()
        return {"topics": [{k: v for k, v in r.items() if not k.startswith("@")} for r in rows]}

    async def ltm_entity(self, name: str) -> dict[str, Any]:
        """Return one LTM entity by name."""
        from graphknows.storage import build_arcadedb_ltm_store

        ltm = build_arcadedb_ltm_store(self.settings, self.namespace)
        await ltm.connect()
        try:
            rows = await ltm._qry(
                "MATCH (e:ENTITY) WHERE toLower(e.name) = toLower($name) "
                "RETURN e.name AS name, e.type AS type, e.confidence AS confidence, "
                "e.pagerank AS pagerank, e.community_id AS community_id LIMIT 1",
                name=name,
            )
        finally:
            await ltm.close()
        if not rows:
            return {"error": f"Entity not found: {name}"}
        return {k: v for k, v in rows[0].items() if not k.startswith("@")}

    def ltm_query(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Return topic-like LTM matches for research evaluation callers."""
        from graphknows._async_bridge import _run as bg_run

        topics = bg_run(self.ltm_topics(min_entities=1))["topics"]
        tokens = [token for token in re.findall(r"\b\w{3,}\b", query.lower())]
        if tokens:
            topics = [
                topic
                for topic in topics
                if any(
                    token in str(topic.get("summary") or topic.get("label") or "").lower()
                    or any(token in str(name).lower() for name in topic.get("entity_names") or [])
                    for token in tokens
                )
            ]
        return {"results": topics[: max(1, min(top_k, 20))]}

    async def ltm_entities(self, session_id: str = "", limit: int = 200) -> dict[str, Any]:
        """List ENTITY nodes from the LTM graph, optionally scoped to one session."""
        from graphknows.storage import build_arcadedb_ltm_store

        ltm = build_arcadedb_ltm_store(self.settings, self.namespace)
        await ltm.connect()
        try:
            if session_id:
                # Split into two MATCH clauses — ArcadeDB's openCypher rejects
                # 5-hop chains in a single MATCH pattern.
                rows = await ltm._qry(
                    "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK) "
                    "MATCH (c)-[:MENTIONS]->(e:ENTITY) "
                    "RETURN DISTINCT e.name AS name, coalesce(e.type,'ENTITY') AS type, "
                    "coalesce(e.confidence, 1.0) AS confidence, "
                    "coalesce(e.pagerank, 0.0) AS pagerank, "
                    "coalesce(e.community_id, -1) AS community_id "
                    "LIMIT $limit",
                    sid=session_id,
                    limit=limit,
                )
            else:
                rows = await ltm._qry(
                    "MATCH (e:ENTITY) "
                    "RETURN e.name AS name, coalesce(e.type,'ENTITY') AS type, "
                    "coalesce(e.confidence, 1.0) AS confidence, "
                    "coalesce(e.pagerank, 0.0) AS pagerank, "
                    "coalesce(e.community_id, -1) AS community_id "
                    "ORDER BY e.name LIMIT $limit",
                    limit=limit,
                )
        finally:
            await ltm.close()
        entities = [{k: v for k, v in r.items() if not k.startswith("@")} for r in rows]
        return {"entities": entities, "count": len(entities), "session_id": session_id}
