"""LTMPromotionPipeline: promote STM to LTM (ArcadeDB)."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import math
import time
from typing import Any, cast

from graphknows.config.adapters.ports.vector_db import IVectorDb
from graphknows.memory.ltm.entity_resolution import EntityResolution
from graphknows.memory.models.contracts import CommunityAssignment, SessionSnapshot
from graphknows.memory.models.promotion import PromotionRecord, PromotionRequest

log = logging.getLogger("graphknows.memory.flush")


class LTMPromotionPipeline:
    """Promote a session's STM to LTM and optionally clear STM.

    Promotion phases (in order):
    1. Read SESSION subgraph from ArcadeDB STM.
    2. Optional entity resolution: canonical name, highest confidence.
    3. MERGE into ArcadeDB LTM: SESSION → FILE → CHUNK → ENTITY, SESSION → DIALOG → TURN.
    4. Optional LLM relation extraction (llm_assisted mode only).
    5. Copy stm_chunks → ltm_chunks, stm_turns → ltm_turns.
    7. PageRank / centrality persistence onto ENTITY nodes (both modes).
    8. Graph pruning: remove tombstoned-beyond-window edges (both modes).
    9. Optional GDS Leiden community detection.
    10. Optional LLM topic summarization → ltm_topics.
    11. Optional community metadata write-back.
    12. Optional STM clear.
    """

    def __init__(
        self,
        mg_store: Any,
        pg_store: IVectorDb,
        neo_store: Any,
        llm_api_key: str = "",
        llm_api_base: str = "",
        llm_model: str = "",
        extraction_uses_llm: bool = False,
        channels: list[Any] | None = None,
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._neo = neo_store
        self._llm_api_key = llm_api_key
        self._llm_api_base = llm_api_base
        self._llm_model = llm_model
        self._extraction_uses_llm = extraction_uses_llm
        self._channels = channels or []
        self._entity_resolution = EntityResolution()

    async def close(self) -> None:
        """Release all store connections."""
        await self._pg.close()
        for store in (self._mg, self._neo):
            client = getattr(store, "_c", None)
            if client is not None and hasattr(client, "close"):
                with contextlib.suppress(Exception):
                    await client.close()

    async def promote(self, request: PromotionRequest) -> PromotionRecord:
        """Run the full STM→LTM promotion pipeline.

        Args:
            request: Promotion parameters.

        Returns:
            PromotionRecord with counts and any non-fatal errors.
        """
        t0 = time.monotonic()
        record = PromotionRecord(session_id=request.session_id)
        policy = request.resolve_policy(self._extraction_uses_llm)
        record.topic_mode = policy.topic_mode

        await self._ensure_stores_connected()
        stm_data = await self._read_stm(request.session_id)
        snapshot = SessionSnapshot.from_store(request.session_id, stm_data)
        stm_data = snapshot.to_store_dict()

        if policy.merge_entities:
            stm_data["entities"], record.audit = self._resolve_entities_with_audit(
                stm_data.get("entities", [])
            )
            record.merged_entities = record.audit.merged_entities
            # Remap alias names in edges to the canonical form so ArcadeDB doesn't
            # create split entity nodes (one with CHUNK-[:MENTIONS] edges from
            # chunk_entity_edges, another with relation edges from the canonical).
            if record.audit.entity_aliases:
                alias_to_canon: dict[str, str] = {}
                for canon, aliases in record.audit.entity_aliases.items():
                    for alias in aliases:
                        alias_to_canon[alias.casefold()] = canon
                for edge in stm_data.get("chunk_entity_edges", []):
                    name = edge.get("entity_name", "")
                    edge["entity_name"] = alias_to_canon.get(name.casefold(), name)
                for edge in stm_data.get("turn_entity_edges", []):
                    name = edge.get("entity_name", "")
                    edge["entity_name"] = alias_to_canon.get(name.casefold(), name)

        graph_counts = await self._write_ltm_graph(
            request.session_id, stm_data, errors=record.errors
        )
        record.nodes_written = graph_counts["nodes"]
        record.edges_written = graph_counts["edges"]

        await self._copy_vector_to_ltm(request.session_id, record)
        await self._index_entities(stm_data, record)

        # Channel promotions: each channel derives + writes its LTM structure now
        # that chunks are in LTM. Soft-fail per channel (matches GDS/topics phases).
        await self._run_channel_promotions(request.session_id, record)

        # Phase 7: PageRank / centrality persistence (both modes).
        await self._run_pagerank_persistence(record)

        # Phase 8: Graph pruning — remove stale tombstoned edges (both modes).
        await self._run_graph_pruning(record)

        gds_result = None
        if policy.run_gds:
            gds_result = await self._run_gds(record)
        if policy.run_topics:
            await self._run_topics(request.session_id, gds_result, record)
        if policy.run_gds:
            # Community metadata written after GDS+topics so that both the
            # Python-native Leiden write-back and any leidenalg fallback in
            # write_topic_summaries have already set community_id on ENTITY nodes.
            await self._write_community_metadata(request.session_id, record)

        if policy.clear_stm:
            try:
                await self._clear_stm(request.session_id)
            except Exception as exc:
                record.errors.append(f"clear_stm: {exc}")

        record.elapsed_s = time.monotonic() - t0
        log.info(
            "promote session=%s nodes=%d edges=%d chunks=%d turns=%d topics=%d elapsed=%.1fs",
            request.session_id,
            record.nodes_written,
            record.edges_written,
            record.chunks_promoted,
            record.turns_promoted,
            record.topics_created,
            record.elapsed_s,
        )
        return record

    async def _ensure_stores_connected(self) -> None:
        """Connect all store clients that are not yet open."""
        if hasattr(self._pg, "connect") and not getattr(self._pg, "_pool", None):
            await self._pg.connect()
        for store in (self._mg, self._neo):
            client = getattr(store, "_c", None)
            if (
                client is not None
                and getattr(client, "_client", None) is None
                and hasattr(client, "connect")
            ):
                await client.connect()

    # keep old name so callers outside this file still compile
    _ensure_pg_connected = _ensure_stores_connected

    async def _copy_vector_to_ltm(self, session_id: str, record: PromotionRecord) -> None:
        try:
            chunks_promoted, turns_promoted = await self._pg.promote_to_ltm(session_id)
            record.chunks_promoted = chunks_promoted
            record.turns_promoted = turns_promoted
        except Exception as exc:
            record.errors.append(f"promote_to_ltm: {exc}")

    async def _run_gds(self, record: PromotionRecord) -> Any:
        """Run community detection and persist community_id on entities.

        Gated to llm_assisted by ``PromotionPolicy`` (``run_gds``): only the
        llm_assisted retriever consumes ``community_id`` (topic summaries +
        community-weighted fusion), so llm_free skips this entirely.
        """
        detect = getattr(self._neo, "detect_communities", None)
        if not callable(detect):
            return None
        try:
            result = await detect()
        except Exception as exc:
            record.errors.append(f"gds: {exc}")
            log.warning("community detection failed: %s", exc)
            return None
        log.info(
            "community detection: %d communities (algo=%s, modularity=%.3f)",
            getattr(result, "community_count", 0),
            getattr(result, "algorithm_used", "?"),
            getattr(result, "modularity", 0.0),
        )
        return result

    async def _run_topics(self, session_id: str, gds_result: Any, record: PromotionRecord) -> None:
        try:
            summarizer = None
            if self._llm_api_key:
                from graphknows.memory.ltm.topic_summarizer import TopicSummarizer

                summarizer = TopicSummarizer(
                    self._llm_api_key, self._llm_model, self._llm_api_base
                )
            record.topics_created = await self._neo.write_topic_summaries(
                session_id=session_id,
                summarizer=summarizer,
                pg_store=self._pg,
                gds_modularity=getattr(gds_result, "modularity", None),
                community_algorithm=getattr(gds_result, "algorithm_used", None),
            )
        except Exception as exc:
            record.errors.append(f"topics: {exc}")
            log.warning("Topic summarization failed: %s", exc)

    async def _read_stm(self, session_id: str) -> dict[str, Any]:
        try:
            return cast(dict[str, Any], await self._mg.read_session_subgraph(session_id))
        except Exception as exc:
            log.warning("read_stm %s: %s", session_id, exc)
            return {"entities": [], "files": [], "chunks": [], "dialogs": [], "turns": []}

    def _resolve_entities_with_audit(
        self, entities: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], Any]:
        return self._entity_resolution.resolve(entities)

    async def _write_community_metadata(
        self,
        session_id: str,
        record: PromotionRecord,
    ) -> None:
        """Write community assignments back to Semantic Memory when available."""
        assignments = await self._read_community_assignments()
        record.audit.communities = assignments
        if not assignments:
            return
        writer = getattr(self._pg, "write_community_assignments", None)
        if not callable(writer):
            return
        try:
            result = writer(session_id, [item.model_dump() for item in assignments])
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            record.errors.append(f"community_metadata: {exc}")

    async def _read_community_assignments(self) -> list[CommunityAssignment]:
        try:
            rows = await self._neo._qry(
                "MATCH (e:ENTITY) WHERE e.community_id IS NOT NULL "
                "OPTIONAL MATCH (c:CHUNK)-[:MENTIONS]->(e) "
                "OPTIONAL MATCH (t:TURN)-[:HAS_ENTITY|MENTIONS]->(e) "
                "WITH e.community_id AS community_id, "
                "     collect(DISTINCT e.name) AS entity_names, "
                "     collect(DISTINCT c.id) AS chunk_ids, "
                "     collect(DISTINCT t.id) AS turn_ids "
                "RETURN community_id, entity_names, chunk_ids, turn_ids"
            )
            return [self._community_from_row(row) for row in rows]
        except Exception as exc:
            log.warning("community assignment read failed: %s", exc)
            return []

    @staticmethod
    def _community_from_row(row: dict[str, Any]) -> CommunityAssignment:
        entity_names = [name for name in row.get("entity_names", []) if name]
        chunk_ids = [chunk_id for chunk_id in row.get("chunk_ids", []) if chunk_id]
        turn_ids = [turn_id for turn_id in row.get("turn_ids", []) if turn_id]
        community_id = int(row.get("community_id") or row.get("cid") or 0)
        weight = max(1.0, 1.0 + math.log1p(len(entity_names)))
        return CommunityAssignment(
            community_id=community_id,
            entity_names=entity_names,
            chunk_ids=chunk_ids,
            turn_ids=turn_ids,
            weight=weight,
            algorithm=str(row.get("algorithm") or row.get("community_algorithm") or ""),
        )

    async def _write_ltm_graph(
        self,
        session_id: str,
        stm_data: dict[str, Any],
        errors: list[str] | None = None,
    ) -> dict[str, int]:
        """MERGE all STM nodes/edges into ArcadeDB LTM."""
        nodes = 0
        edges = 0
        try:
            result = await self._neo.write_session_subgraph(session_id, stm_data)
            nodes = result.get("nodes", 0)
            edges = result.get("edges", 0)
        except Exception as exc:
            log.warning("_write_ltm_graph %s: %s", session_id, exc)
            if errors is not None:
                errors.append(f"write_ltm_graph: {exc}")

        if not self._extraction_uses_llm:
            log.info(
                "_write_ltm_graph %s: extraction_uses_llm=False; skipping "
                "extract_and_write_relations (rel_edges=0)",
                session_id,
            )
            return {"nodes": nodes, "edges": edges}

        if not (self._llm_api_key or "").strip():
            log.warning(
                "_write_ltm_graph %s: llm_api_key is empty — skipping "
                "extract_and_write_relations (rel_edges=0)",
                session_id,
            )
            return {"nodes": nodes, "edges": edges}

        rel_edges = 0
        try:
            rel_edges = await self._neo.extract_and_write_relations(
                stm_data=stm_data,
                llm_api_key=self._llm_api_key,
                session_id=session_id,
            )
        except Exception as exc:
            log.warning(
                "_write_ltm_graph %s: extract_and_write_relations failed: %s",
                session_id,
                exc,
            )
            if errors is not None:
                errors.append(f"extract_and_write_relations: {exc}")
        edges += rel_edges
        return {"nodes": nodes, "edges": edges}

    async def _index_entities(
        self,
        stm_data: dict[str, Any],
        record: Any,
    ) -> None:
        """Embed entity names and upsert into the vector store.

        Called after promote_to_ltm so the entity-graph path in HybridRetriever
        has live data to search. Non-fatal: errors are logged and appended to record.
        """
        from graphknows.memory.stores.embedder import embed

        entities: list[dict[str, Any]] = stm_data.get("entities", [])
        if not entities:
            return

        # Build chunk_count per entity from chunk metadata.
        chunk_entity_counts: dict[str, int] = {}
        for chunk in stm_data.get("chunks", []):
            meta = chunk.get("metadata", {}) or {}
            for name in meta.get("entity_names", []):
                if isinstance(name, str):
                    chunk_entity_counts[name] = chunk_entity_counts.get(name, 0) + 1

        names = [e["name"] for e in entities]
        try:
            emb_array = await asyncio.to_thread(embed, names)
        except Exception as exc:
            log.warning("_index_entities: embed failed: %s", exc)
            record.errors.append(f"entity_embed: {exc}")
            return

        entity_records = [
            {
                "name": ent["name"],
                "label": ent.get("type") or ent.get("label") or "Entity",
                "embedding": emb_array[i].tolist(),
                "chunk_count": chunk_entity_counts.get(ent["name"], 0),
            }
            for i, ent in enumerate(entities)
        ]
        try:
            await self._pg.upsert_entities(entity_records)
            log.info("_index_entities: upserted %d entity embeddings", len(entity_records))
        except Exception as exc:
            log.warning("_index_entities: upsert failed: %s", exc)
            record.errors.append(f"entity_upsert: {exc}")

    async def _run_pagerank_persistence(self, record: PromotionRecord) -> None:
        """Compute and persist PageRank onto ENTITY nodes.

        Calls ltm_store.compute_and_persist_pagerank() if available.
        Non-fatal.
        """
        persist_fn = getattr(self._neo, "compute_and_persist_pagerank", None)
        if not callable(persist_fn):
            log.debug("_run_pagerank_persistence: ltm_store has no compute_and_persist_pagerank, skip")
            return
        try:
            updated = await persist_fn()
            log.info("_run_pagerank_persistence: updated pagerank on %d ENTITY nodes", updated)
        except Exception as exc:
            log.warning("_run_pagerank_persistence failed (non-fatal): %s", exc)
            record.errors.append(f"pagerank_persistence: {exc}")

    async def _run_graph_pruning(self, record: PromotionRecord) -> None:
        """Prune stale tombstoned relation edges from LTM.

        Calls ltm_store.prune_tombstoned_edges() and prune_duplicate_edges()
        if available.  Non-fatal.
        """
        for fn_name in ("prune_tombstoned_edges", "prune_duplicate_edges"):
            prune_fn = getattr(self._neo, fn_name, None)
            if not callable(prune_fn):
                log.debug("_run_graph_pruning: ltm_store has no %s, skip", fn_name)
                continue
            try:
                deleted = await prune_fn()
                log.info("_run_graph_pruning %s: deleted %d edges", fn_name, deleted)
            except Exception as exc:
                log.warning("_run_graph_pruning %s failed (non-fatal): %s", fn_name, exc)
                record.errors.append(f"{fn_name}: {exc}")

    async def _run_channel_promotions(self, session_id: str, record: Any) -> None:
        """Run each channel's promote() hook over the freshly-promoted LTM chunks.

        Soft-fail per channel (like _run_gds/_run_topics): a broken enrichment
        channel is logged and recorded, but never aborts the promotion.
        """
        if not self._channels:
            return
        from graphknows.memory.channels.base import ChannelStores

        stores = ChannelStores(mg=self._mg, pg=self._pg, neo=self._neo)
        for channel in self._channels:
            try:
                await channel.promote(stores, session_id, record)
            except Exception as exc:
                log.warning("channel %s promote failed (non-fatal): %s", channel.name, exc)
                record.errors.append(f"channel_{channel.name}: {exc}")

    async def _clear_stm(self, session_id: str) -> None:
        await self._mg.delete_session(session_id)
        await self._pg.clear_stm(session_id)
