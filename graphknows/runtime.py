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


def _format_fused_hits(fused_results: list[dict], session_id: str) -> list[dict[str, Any]]:
    return [
        {
            "score": r.get("rrf_score", 0.0),
            "cosine_score": r.get("rrf_score", 0.0),
            "type": r.get("sources", ""),
            "text": _clean_recall_text(r.get("text", "")),
            "session_id": session_id,
            "entities": r.get("entities", []),
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
        from graphknows.memory.adapter_types import MemoryClass
        from graphknows.memory.models.ingest_options import IngestOptions
        from graphknows.memory.models.message import normalize_messages
        from graphknows.memory.models.scope import MemoryScope

        scope = MemoryScope(
            user_id=user_id or None,
            agent_id=agent_id or None,
            run_id=run_id or session_id or None,
        )
        is_conversation = isinstance(messages, list) or (
            isinstance(messages, dict) and "role" in messages
        )

        service = build_stm_service(self.settings)
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
        )
        return {
            "file_id": result.file_id,
            "chunks": result.chunk_count,
            "entities": result.entity_count,
            "elapsed_s": round(result.elapsed_s, 3),
            "errors": list(result.errors),
            "warnings": [],
        }

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
        return await asyncio.to_thread(self._recall_memory_sync, query, session_id, top_k, scope)

    def _connect_pg_on_bridge(self) -> Any:
        from graphknows.config.builders.stores import build_pgvector_store
        from graphknows.memory.stm._async_bridge import _run as bg_run

        pg = build_pgvector_store(self.settings)
        bg_run(pg.connect())
        return pg

    def _recall_memory_sync(
        self,
        query: str,
        session_id: str,
        top_k: int,
        scope: str,
    ) -> dict[str, Any]:
        from graphknows.config.builders.stores import (
            build_memgraph_store,
            build_neo4j_store,
        )
        from graphknows.memory.query.retriever import DETRetriever
        from graphknows.memory.stm._async_bridge import _run as bg_run

        mg = build_memgraph_store(self.settings)
        pg = self._connect_pg_on_bridge()
        neo = build_neo4j_store(self.settings)
        try:
            retriever = DETRetriever(
                pg=pg, neo=neo, mg=mg, embed_model=self.settings.embed_model,
            )
            ctx = bg_run(retriever.retrieve(query, session_id=session_id, top_k=top_k, scope=scope))
        finally:
            mg.close()
            neo.close()
        return {"hits": _format_fused_hits(ctx.fused_results, session_id)}

    async def flush_memory(self, session_id: str) -> dict[str, Any]:
        """Promote a session from STM into LTM."""
        from graphknows.config.builders.stores import (
            build_neo4j_store,
            build_pgvector_store,
            build_promotion_pipeline,
        )
        from graphknows.memory.models.promotion import PromotionRequest
        from graphknows.memory.stm._async_bridge import _run as bg_run

        neo_check = build_neo4j_store(self.settings)
        try:
            if await asyncio.to_thread(self._session_exists, neo_check, session_id):
                return {
                    "session_id": session_id,
                    "status": "already_flushed",
                    "nodes": 0,
                    "edges": 0,
                    "chunks": 0,
                    "turns": 0,
                    "topics_created": 0,
                    "merged_entities": 0,
                    "elapsed_s": 0.0,
                    "errors": [],
                }
        finally:
            neo_check.close()

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

        try:
            pg = build_pgvector_store(self.settings)
            try:
                await pg.connect()
                neo_sync = build_neo4j_store(self.settings)
                try:
                    await self._sync_topics_to_pgvector(neo_sync, pg)
                finally:
                    neo_sync.close()
            finally:
                await pg.close()
        except Exception as exc:
            result.errors.append(f"topic_sync: {exc}")

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

    @staticmethod
    def _session_exists(neo_store: Any, session_id: str) -> bool:
        with neo_store._driver.session(database=neo_store._db) as session:
            rows = list(
                session.run(
                    "MATCH (s:SESSION {id: $sid}) RETURN count(s) AS n",
                    sid=session_id,
                )
            )
        return bool(rows and rows[0]["n"] > 0)

    async def _sync_topics_to_pgvector(self, neo_store: Any, pg_store: Any) -> int:
        from graphknows.memory.stores.embedder import embed_one

        with neo_store._driver.session(database=neo_store._db) as session:
            rows = session.run(
                "MATCH (t:TOPIC) "
                "OPTIONAL MATCH (e:ENTITY)-[:IN_TOPIC]->(t) "
                "RETURN t.id AS topic_id, t.summary AS summary, t.community_id AS comm_id, "
                "collect(e.name) AS members"
            ).data()

        synced = 0
        loop = asyncio.get_running_loop()
        for row in rows:
            topic_id = row.get("topic_id") or f"TOPIC_{row.get('comm_id', synced)}"
            summary = (row.get("summary") or "").strip()
            members = row.get("members") or []
            embedding = await loop.run_in_executor(None, embed_one, summary or topic_id)
            await pg_store.write_ltm_topic(
                {
                    "topic_id": topic_id,
                    "summary": summary,
                    "embedding": embedding,
                    "entity_names": [member for member in members if member],
                    "session_ids": [],
                }
            )
            synced += 1
        return synced

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
        from graphknows.config.builders.stores import build_memgraph_store

        try:
            mg_store = build_memgraph_store(self.settings)
            if session_id:
                await mg_store.delete_session(session_id)
            else:
                await asyncio.to_thread(self._wipe_graph_store, mg_store)
        except Exception as exc:
            errors.append(f"memgraph stm: {exc}")

        try:
            from graphknows.config.builders.stores import build_pgvector_store

            pg_store = build_pgvector_store(self.settings)
            try:
                await pg_store.connect()
                if session_id:
                    await pg_store.clear_stm(session_id)
                else:
                    pool = pg_store._pool_or_raise()
                    async with pool.acquire() as conn:
                        await conn.execute("TRUNCATE TABLE stm_chunks, stm_turns")
            finally:
                await pg_store.close()
        except Exception as exc:
            errors.append(f"pgvector stm: {exc}")

    async def _purge_ltm(self, session_id: str, errors: list[str]) -> None:
        from graphknows.config.builders.stores import build_neo4j_store, build_pgvector_store

        try:
            neo_store = build_neo4j_store(self.settings)
            if session_id:
                await asyncio.to_thread(self._delete_ltm_session, neo_store, session_id)
            else:
                await asyncio.to_thread(self._wipe_graph_store, neo_store)
        except Exception as exc:
            errors.append(f"neo4j ltm: {exc}")

        try:
            pg_store = build_pgvector_store(self.settings)
            try:
                await pg_store.connect()
                pool = pg_store._pool_or_raise()
                async with pool.acquire() as conn:
                    if session_id:
                        await conn.execute("DELETE FROM ltm_chunks WHERE session_id = $1", session_id)
                        await conn.execute("DELETE FROM ltm_turns WHERE session_id = $1", session_id)
                        await conn.execute(
                            "DELETE FROM ltm_topics "
                            "WHERE $1 = ANY(COALESCE(session_ids, ARRAY[]::text[]))",
                            session_id,
                        )
                        if await conn.fetchval("SELECT to_regclass('public.chunks') IS NOT NULL"):
                            await conn.execute("DELETE FROM chunks WHERE session_id = $1", session_id)
                    else:
                        await conn.execute(
                            "TRUNCATE TABLE ltm_chunks, ltm_turns, ltm_topics, "
                            "chunks, chunk_entities, entities"
                        )
            finally:
                await pg_store.close()
        except Exception as exc:
            errors.append(f"pgvector ltm: {exc}")

    @staticmethod
    def _wipe_graph_store(store: Any) -> None:
        kwargs = {"database": store._db} if hasattr(store, "_db") else {}
        with store._driver.session(**kwargs) as session:
            session.run("MATCH (n) DETACH DELETE n")

    @staticmethod
    def _delete_ltm_session(neo_store: Any, session_id: str) -> None:
        with neo_store._driver.session(database=neo_store._db) as session:
            session.run(
                """
                MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)
                OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:CHUNK)
                WITH f, collect(DISTINCT c) AS chunks
                FOREACH (chunk IN chunks | DETACH DELETE chunk)
                DETACH DELETE f
                """,
                sid=session_id,
            )
            session.run(
                """
                MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(d:DIALOG)
                OPTIONAL MATCH (d)-[:HAS_TURN]->(t:TURN)
                WITH d, collect(DISTINCT t) AS turns
                FOREACH (turn IN turns | DETACH DELETE turn)
                DETACH DELETE d
                """,
                sid=session_id,
            )
            session.run("MATCH (sess:SESSION {id: $sid}) DETACH DELETE sess", sid=session_id)
            session.run(
                "MATCH (n:SessionEntity {session_id: $sid}) DETACH DELETE n",
                sid=session_id,
            )
            session.run("MATCH ()-[r {session_id: $sid}]-() DELETE r", sid=session_id)
            session.run("MATCH (e:ENTITY) WHERE NOT (e)--() DETACH DELETE e")
            session.run(
                """
                MATCH (t:TOPIC)
                WHERE NOT (t)<-[:IN_TOPIC]-(:ENTITY)
                  AND NOT (t)<-[:IN_COMMUNITY]-(:ENTITY)
                DETACH DELETE t
                """
            )

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
                    summarize_topics=True,
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

    def ltm_topics(self, min_entities: int = 3) -> dict[str, Any]:
        """List LTM topic nodes."""
        from graphknows.config.builders.stores import build_neo4j_store

        neo = build_neo4j_store(self.settings)
        try:
            return {"topics": neo.list_topic_nodes(min_entities)}
        finally:
            neo.close()

    def ltm_entity(self, name: str) -> dict[str, Any]:
        """Return one LTM entity by name."""
        from graphknows.config.builders.stores import build_neo4j_store

        neo = build_neo4j_store(self.settings)
        try:
            entity = neo.get_entity_by_name(name)
        finally:
            neo.close()
        if entity is None:
            return {"error": f"Entity not found: {name}"}
        return entity

    def ltm_query(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Return topic-like LTM matches for research evaluation callers."""
        topics = self.ltm_topics(min_entities=1)["topics"]
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

    def ltm_entities(self, session_id: str = "", limit: int = 200) -> dict[str, Any]:
        """List ENTITY nodes from the LTM graph, optionally scoped to one session."""
        from graphknows.config.builders.stores import build_neo4j_store

        neo = build_neo4j_store(self.settings)
        try:
            _db = getattr(neo, "_db", None)
            kwargs: dict[str, Any] = {"database": _db} if _db else {}
            with neo._driver.session(**kwargs) as s:
                if session_id:
                    cypher = """
                    MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(:FILE)-[:HAS_CHUNK]->(:CHUNK)
                          -[:MENTIONS]->(e:ENTITY)
                    RETURN DISTINCT e.name AS name, coalesce(e.type,'ENTITY') AS type,
                           coalesce(e.confidence, 1.0) AS confidence,
                           coalesce(e.pagerank, 0.0) AS pagerank,
                           coalesce(e.community_id, -1) AS community_id
                    ORDER BY e.name LIMIT $limit
                    """
                    rows = s.run(cypher, sid=session_id, limit=limit).data()
                else:
                    cypher = """
                    MATCH (e:ENTITY)
                    RETURN e.name AS name, coalesce(e.type,'ENTITY') AS type,
                           coalesce(e.confidence, 1.0) AS confidence,
                           coalesce(e.pagerank, 0.0) AS pagerank,
                           coalesce(e.community_id, -1) AS community_id
                    ORDER BY e.name LIMIT $limit
                    """
                    rows = s.run(cypher, limit=limit).data()
        finally:
            neo.close()
        return {"entities": [dict(r) for r in rows], "count": len(rows), "session_id": session_id}

