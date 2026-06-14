"""LTMPromotionPipeline: promote STM (Memgraph + stm_* pgvector) to LTM (Neo4j + ltm_* pgvector)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, cast

from graphknows.memory.ltm.enrichment import LTMEnrichmentService
from graphknows.memory.models.promotion import PromotionRecord, PromotionRequest

log = logging.getLogger("graphknows.memory.flush")


class LTMPromotionPipeline:
    """Promote a session's STM to LTM and optionally clear STM.

    Promotion phases (in order):
    1. Read SESSION subgraph from Memgraph.
    2. Optional entity resolution: canonical name, highest confidence.
    3. MERGE into Neo4j: SESSION → FILE → CHUNK → ENTITY, SESSION → DIALOG → TURN.
    4. Optional LLM relation extraction (llm_assisted mode only).
    5. Copy stm_chunks → ltm_chunks, stm_turns → ltm_turns.
    6. Optional GDS PageRank + Leiden community detection.
    7. Optional LLM topic summarization → ltm_topics.
    8. Optional STM clear.
    """

    def __init__(
        self,
        mg_store: Any,
        pg_store: Any,
        neo_store: Any,
        openrouter_key: str = "",
        llm_model: str = "meta-llama/llama-3.1-8b-instruct",
        relation_extraction_mode: str = "gliner2",
        enrichment_service: LTMEnrichmentService | None = None,
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._neo = neo_store
        self._openrouter_key = openrouter_key
        self._llm_model = llm_model
        self._relation_extraction_mode = relation_extraction_mode
        self._enrichment = enrichment_service or LTMEnrichmentService()

    async def promote(self, request: PromotionRequest) -> PromotionRecord:
        """Run the full STM→LTM promotion pipeline.

        Args:
            request: Promotion parameters.

        Returns:
            PromotionRecord with counts and any non-fatal errors.
        """
        t0 = time.monotonic()
        record = PromotionRecord(session_id=request.session_id)

        if hasattr(self._pg, "connect") and not getattr(self._pg, "_pool", None):
            await self._pg.connect()

        stm_data = await self._read_stm(request.session_id)

        if request.merge_entities:
            stm_data["entities"], record.merged_entities = self._resolve_entities(
                stm_data.get("entities", [])
            )

        graph_counts = await self._write_ltm_graph(
            request.session_id, stm_data, errors=record.errors
        )
        record.nodes_written = graph_counts["nodes"]
        record.edges_written = graph_counts["edges"]

        try:
            chunks_promoted, turns_promoted = await self._pg.promote_to_ltm(request.session_id)
            record.chunks_promoted = chunks_promoted
            record.turns_promoted = turns_promoted
        except Exception as exc:
            record.errors.append(f"promote_to_ltm: {exc}")

        await self._index_entities(stm_data, record)

        gds_result = None

        if request.run_gds:
            try:
                gds_result = await self._enrichment.run_gds_async(self._neo)
                record.gds_result = gds_result
            except Exception as exc:
                record.errors.append(f"gds: {exc}")
                log.warning("GDS enrichment failed: %s", exc)

        if request.run_topics:
            try:
                record.topics_created = await self._enrichment.run_topic_summarization(
                    session_id=request.session_id,
                    driver=self._neo,
                    pg_store=self._pg,
                    llm_key=self._openrouter_key,
                    llm_model=self._llm_model,
                    gds_result=gds_result,
                )
            except Exception as exc:
                record.errors.append(f"topics: {exc}")
                log.warning("Topic summarization failed: %s", exc)

        if request.clear_stm:
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

    async def _read_stm(self, session_id: str) -> dict[str, Any]:
        try:
            return cast(dict[str, Any], await self._mg.read_session_subgraph(session_id))
        except Exception as exc:
            log.warning("read_stm %s: %s", session_id, exc)
            return {"entities": [], "files": [], "chunks": [], "dialogs": [], "turns": []}

    def _resolve_entities(self, entities: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in entities:
            groups[entity["name"].lower()].append(entity)

        resolved: list[dict[str, Any]] = []
        merge_count = 0
        for group in groups.values():
            best = max(group, key=lambda x: x.get("confidence", 1.0))
            if len(group) > 1:
                merge_count += len(group) - 1
            resolved.append(best)

        return resolved, merge_count

    async def _write_ltm_graph(
        self,
        session_id: str,
        stm_data: dict[str, Any],
        errors: list[str] | None = None,
    ) -> dict[str, int]:
        """MERGE all STM nodes/edges into Neo4j LTM."""
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

        if self._relation_extraction_mode != "llm_assisted":
            log.info(
                "_write_ltm_graph %s: relation_extraction_mode=%s; skipping "
                "extract_and_write_relations (rel_edges=0)",
                session_id,
                self._relation_extraction_mode,
            )
            return {"nodes": nodes, "edges": edges}

        if not (self._openrouter_key or "").strip():
            log.warning(
                "_write_ltm_graph %s: openrouter_key is empty — skipping "
                "extract_and_write_relations (rel_edges=0)",
                session_id,
            )
            return {"nodes": nodes, "edges": edges}

        rel_edges = 0
        try:
            rel_edges = await self._neo.extract_and_write_relations(
                stm_data=stm_data,
                openrouter_key=self._openrouter_key,
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
        """Embed entity names and upsert into pgvector entities table.

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

    async def _clear_stm(self, session_id: str) -> None:
        await self._mg.delete_session(session_id)
        await self._pg.clear_stm(session_id)
