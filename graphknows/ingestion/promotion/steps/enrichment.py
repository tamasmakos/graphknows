"""Enrichment + lifecycle promotion steps: PageRank, graph pruning, community
detection (GDS), topic summarization, community-metadata write-back, and STM clear.

PageRank and pruning run in both modes. Community detection, topics, and the
metadata write-back are llm_assisted-only and are added to the pipeline only for
that mode.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from graphknows.ingestion.models.contracts import CommunityAssignment
from graphknows.ingestion.promotion.base import PromotionContext

log = logging.getLogger("graphknows.ingestion.promotion")


class PagerankStep:
    """Compute and persist PageRank onto ENTITY nodes (both modes). Non-fatal."""

    name = "pagerank"

    async def run(self, ctx: PromotionContext) -> None:
        try:
            updated = await ctx.ltm.compute_and_persist_pagerank()
            log.info("pagerank: updated %d ENTITY nodes", updated)
        except Exception as exc:
            log.warning("pagerank failed (non-fatal): %s", exc)
            ctx.record.errors.append(f"pagerank_persistence: {exc}")


class GraphPruningStep:
    """Prune stale tombstoned + duplicate relation edges (both modes). Non-fatal."""

    name = "graph_pruning"

    async def run(self, ctx: PromotionContext) -> None:
        for fn_name in ("prune_tombstoned_edges", "prune_duplicate_edges"):
            try:
                deleted = await getattr(ctx.ltm, fn_name)()
                log.info("graph_pruning %s: deleted %d edges", fn_name, deleted)
            except Exception as exc:
                log.warning("graph_pruning %s failed (non-fatal): %s", fn_name, exc)
                ctx.record.errors.append(f"{fn_name}: {exc}")


class CommunityDetectionStep:
    """Detect entity communities and persist community_id (llm_assisted)."""

    name = "community_detection"

    async def run(self, ctx: PromotionContext) -> None:
        if not ctx.policy.run_gds:
            return
        try:
            result = await ctx.ltm.detect_communities()
        except Exception as exc:
            ctx.record.errors.append(f"gds: {exc}")
            log.warning("community detection failed: %s", exc)
            return
        log.info(
            "community detection: %d communities (algo=%s, modularity=%.3f)",
            getattr(result, "community_count", 0),
            getattr(result, "algorithm_used", "?"),
            getattr(result, "modularity", 0.0),
        )
        ctx.gds_result = result


class TopicSummaryStep:
    """Summarize communities into TOPIC nodes (llm_assisted)."""

    name = "topic_summary"

    async def run(self, ctx: PromotionContext) -> None:
        if not ctx.policy.run_topics:
            return
        try:
            summarizer = None
            if ctx.llm_api_key:
                from graphknows.ingestion.promotion.steps.topic_summary import TopicSummarizer

                summarizer = TopicSummarizer(ctx.llm_api_key, ctx.llm_model, ctx.llm_api_base)
            ctx.record.topics_created = await ctx.ltm.write_topic_summaries(
                session_id=ctx.request.session_id,
                summarizer=summarizer,
                vec_store=ctx.vec,
                gds_modularity=getattr(ctx.gds_result, "modularity", None),
                community_algorithm=getattr(ctx.gds_result, "algorithm_used", None),
            )
        except Exception as exc:
            ctx.record.errors.append(f"topics: {exc}")
            log.warning("Topic summarization failed: %s", exc)


class CommunityMetadataStep:
    """Write community assignments back onto semantic-memory rows (llm_assisted).

    Runs after GDS+topics so both the Leiden write-back and any leidenalg fallback
    in write_topic_summaries have already set community_id on ENTITY nodes.
    """

    name = "community_metadata"

    async def run(self, ctx: PromotionContext) -> None:
        if not ctx.policy.run_gds:
            return
        assignments = await self._read_community_assignments(ctx.ltm)
        ctx.record.audit.communities = assignments
        if not assignments:
            return
        try:
            await ctx.vec.write_community_assignments(
                ctx.request.session_id, [item.model_dump() for item in assignments]
            )
        except Exception as exc:
            ctx.record.errors.append(f"community_metadata: {exc}")

    async def _read_community_assignments(self, ltm: Any) -> list[CommunityAssignment]:
        try:
            rows = await ltm._qry(
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


class ClearStmStep:
    """Clear the promoted session from STM (graph + vectors)."""

    name = "clear_stm"

    async def run(self, ctx: PromotionContext) -> None:
        if not ctx.policy.clear_stm:
            return
        try:
            await ctx.stm.delete_session(ctx.request.session_id)
            await ctx.vec.clear_stm(ctx.request.session_id)
        except Exception as exc:
            ctx.record.errors.append(f"clear_stm: {exc}")


__all__ = [
    "ClearStmStep",
    "CommunityDetectionStep",
    "CommunityMetadataStep",
    "GraphPruningStep",
    "PagerankStep",
    "TopicSummaryStep",
]
