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

from graphknows.config import GraphKnowsSettings

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


def _format_recall_hits(results: list[Any], session_id: str) -> list[dict[str, Any]]:
    return [
        {
            "score": round(result.score, 6),
            "cosine_score": round(getattr(result, "cosine_score", result.score), 6),
            "type": result.source_type,
            "text": _clean_recall_text(result.text),
            "session_id": result.provenance.get("session_id", session_id),
            "entities": list(result.entity_names or []),
        }
        for result in results
    ]


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


def _format_entity_recall_hits(results: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "text": _clean_recall_text(result.text),
            "score": round(result.score, 6),
            "entities": list(result.entity_names or []),
            "provenance": dict(result.provenance or {}),
        }
        for result in results
    ]


class GraphKnowsRuntime:
    """Transport-neutral runtime for graphknows memory operations."""

    def __init__(self, settings: GraphKnowsSettings | None = None) -> None:
        self.settings = settings or GraphKnowsSettings()
        self._retriever: Any = None
        self._retriever_stores: list[Any] = []

    async def close(self) -> None:
        """Close any store connections held by this runtime instance."""
        import contextlib

        for store in self._retriever_stores:
            with contextlib.suppress(Exception):
                await store._c.close()
        self._retriever = None
        self._retriever_stores = []

    async def _get_retriever(self) -> Any:
        """Return a cached retriever, connecting stores lazily on first call."""
        if self._retriever is not None:
            return self._retriever
        from graphknows.config.builders.stores import (
            build_arcadedb_ltm_store,
            build_arcadedb_stm_store,
            build_arcadedb_vector_store,
        )

        s = self.settings
        pg = build_arcadedb_vector_store(s)
        neo = build_arcadedb_ltm_store(s)
        mg = build_arcadedb_stm_store(s)
        await asyncio.gather(
            pg.connect(),  # vector: connect + ensure_schema
            neo._c.connect(),
            mg._c.connect(),
        )
        await neo.ensure_schema()
        self._retriever_stores = [pg, neo, mg]
        if s.retrieval_mode == "agentic":
            from graphknows.memory.query.agentic import AgenticRetriever

            self._retriever = AgenticRetriever(settings=s, pg=pg, neo=neo, stm=mg)
        else:
            from graphknows.memory.query.retriever import DETRetriever

            self._retriever = DETRetriever(
                pg=pg, neo=neo, mg=mg, embed_model=s.embed_model, fusion_alpha=s.fusion_alpha
            )
        return self._retriever

    async def doctor(self) -> dict[str, Any]:
        """Check configured backend connectivity."""
        from graphknows.config.health import check_backends

        result = await check_backends(self.settings)
        result["pid"] = os.getpid()
        return result

    async def stats(self, session_id: str = "") -> dict[str, Any]:
        """Return STM and LTM memory statistics."""
        from graphknows.config.health import get_memory_stats

        return await get_memory_stats(self.settings, session_id=session_id)

    def mirror(self, session_id: str = "", scope: str = "ltm") -> dict[str, Any]:
        """Return Memory Mirror URL metadata."""
        from graphknows.web.links import mirror_url

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

        from graphknows.config.builders.stores import build_stm_service
        from graphknows.memory.models.memory_class import MemoryClass
        from graphknows.memory.models.ingest_options import IngestOptions
        from graphknows.memory.models.message import normalize_messages
        from graphknows.memory.models.scope import MemoryScope
        from graphknows.memory.stm._async_bridge import _run as _bg_run

        scope = MemoryScope(
            user_id=user_id or None,
            agent_id=agent_id or None,
            run_id=run_id or session_id or None,
        )
        is_conversation = isinstance(messages, list) or (
            isinstance(messages, dict) and "role" in messages
        )

        service = build_stm_service(self.settings)
        try:
            if is_conversation or (isinstance(messages, str) and not title):
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
                result = service.ingest_messages(normalize_messages(messages), options)
                return {
                    "file_id": result.file_id,
                    "chunks": result.chunk_count,
                    "entities": result.entity_count,
                    "elapsed_s": round(result.elapsed_s, 3),
                    "errors": list(result.errors),
                }

            result = service.ingest(
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
            _bg_run(service._mg._c.close())
            _bg_run(service._pg._c.close())

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
        from graphknows.config.builders.stores import build_promotion_pipeline
        from graphknows.memory.models.promotion import PromotionRequest
        from graphknows.memory.stm._async_bridge import _run as bg_run

        pipeline = build_promotion_pipeline(self.settings)
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

    async def _purge_stm(self, session_id: str, errors: list[str]) -> None:
        from graphknows.config.builders.stores import build_arcadedb_stm_store

        try:
            stm_store = build_arcadedb_stm_store(self.settings)
            await stm_store._c.connect()
            try:
                if session_id:
                    await stm_store.delete_session(session_id)
                else:
                    await stm_store._c.drop_database(stm_store._db)
            finally:
                await stm_store._c.close()
        except Exception as exc:
            errors.append(f"arcadedb stm: {exc}")

    async def _purge_ltm(self, session_id: str, errors: list[str]) -> None:
        from graphknows.config.builders.stores import (
            build_arcadedb_ltm_store,
            build_arcadedb_vector_store,
        )

        try:
            ltm_store = build_arcadedb_ltm_store(self.settings)
            await ltm_store._c.connect()
            try:
                if session_id:
                    await self._delete_ltm_session(ltm_store, session_id)
                else:
                    await ltm_store._c.drop_database("ltm")
                    # Recreate both schemas immediately so the next flush can
                    # write without a 503 (drop removes all DDL).
                    await ltm_store.ensure_schema()
                    # vec_store.connect() creates its own client connection,
                    # recreates the ltm DB (idempotent), and adds the vector DDL.
                    vec_store = build_arcadedb_vector_store(self.settings)
                    await vec_store.connect()
                    await vec_store.close()
            finally:
                await ltm_store._c.close()
        except Exception as exc:
            errors.append(f"arcadedb ltm: {exc}")

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

    async def corpus_ingest(
        self,
        documents: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest a document batch through the corpus workflow."""
        if not documents:
            return {"documents_processed": 0, "error": "no documents provided"}
        try:
            from graphknows.config.builders.stores import build_corpus_workflow
            from graphknows.memory.workflows.corpus_ingest import CorpusIngestRequest

            workflow = build_corpus_workflow(self.settings)
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
        from graphknows.config.builders.stores import build_arcadedb_ltm_store

        ltm = build_arcadedb_ltm_store(self.settings)
        await ltm._c.connect()
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
            await ltm._c.close()
        return {"topics": [{k: v for k, v in r.items() if not k.startswith("@")} for r in rows]}

    async def ltm_entity(self, name: str) -> dict[str, Any]:
        """Return one LTM entity by name."""
        from graphknows.config.builders.stores import build_arcadedb_ltm_store

        ltm = build_arcadedb_ltm_store(self.settings)
        await ltm._c.connect()
        try:
            rows = await ltm._qry(
                "MATCH (e:ENTITY) WHERE toLower(e.name) = toLower($name) "
                "RETURN e.name AS name, e.type AS type, e.confidence AS confidence, "
                "e.pagerank AS pagerank, e.community_id AS community_id LIMIT 1",
                name=name,
            )
        finally:
            await ltm._c.close()
        if not rows:
            return {"error": f"Entity not found: {name}"}
        return {k: v for k, v in rows[0].items() if not k.startswith("@")}

    def ltm_query(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Return topic-like LTM matches for research evaluation callers."""
        from graphknows.memory.stm._async_bridge import _run as bg_run

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
        from graphknows.config.builders.stores import build_arcadedb_ltm_store

        ltm = build_arcadedb_ltm_store(self.settings)
        await ltm._c.connect()
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
            await ltm._c.close()
        entities = [{k: v for k, v in r.items() if not k.startswith("@")} for r in rows]
        return {"entities": entities, "count": len(entities), "session_id": session_id}
