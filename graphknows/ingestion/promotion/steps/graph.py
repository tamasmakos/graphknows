"""Graph-building promotion steps: read STM, resolve entities, MERGE into LTM,
extract relations (llm_assisted), copy vectors, index entity embeddings, and run
channel promotions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from graphknows.channels.base import ChannelStores
from graphknows.ingestion.models.contracts import SessionSnapshot
from graphknows.ingestion.promotion.base import PromotionContext
from graphknows.ingestion.promotion.steps.entity_resolution import EntityResolution

log = logging.getLogger("graphknows.ingestion.promotion")


class ReadStmStep:
    """Read the SESSION subgraph from STM into ``ctx.stm_data``."""

    name = "read_stm"

    async def run(self, ctx: PromotionContext) -> None:
        try:
            stm_data = cast(dict[str, Any], await ctx.stm.read_session_subgraph(ctx.request.session_id))
        except Exception as exc:
            log.warning("read_stm %s: %s", ctx.request.session_id, exc)
            stm_data = {"entities": [], "files": [], "chunks": [], "dialogs": [], "turns": []}
        snapshot = SessionSnapshot.from_store(ctx.request.session_id, stm_data)
        ctx.stm_data = snapshot.to_store_dict()


class ResolveEntitiesStep:
    """Canonicalize entities (highest confidence) and remap alias names in edges."""

    name = "resolve_entities"

    def __init__(self) -> None:
        self._entity_resolution = EntityResolution()

    async def run(self, ctx: PromotionContext) -> None:
        if not ctx.policy.merge_entities:
            return
        stm_data = ctx.stm_data
        stm_data["entities"], ctx.record.audit = self._entity_resolution.resolve(
            stm_data.get("entities", [])
        )
        ctx.record.merged_entities = ctx.record.audit.merged_entities
        # Remap alias names in edges to the canonical form so ArcadeDB doesn't
        # create split entity nodes (one with CHUNK-[:MENTIONS] edges from
        # chunk_entity_edges, another with relation edges from the canonical).
        if ctx.record.audit.entity_aliases:
            alias_to_canon: dict[str, str] = {}
            for canon, aliases in ctx.record.audit.entity_aliases.items():
                for alias in aliases:
                    alias_to_canon[alias.casefold()] = canon
            for edge_key in ("chunk_entity_edges", "turn_entity_edges"):
                for edge in stm_data.get(edge_key, []):
                    name = edge.get("entity_name", "")
                    edge["entity_name"] = alias_to_canon.get(name.casefold(), name)


class WriteLtmGraphStep:
    """MERGE all STM nodes/edges into ArcadeDB LTM."""

    name = "write_ltm_graph"

    async def run(self, ctx: PromotionContext) -> None:
        nodes = edges = 0
        try:
            result = await ctx.ltm.write_session_subgraph(ctx.request.session_id, ctx.stm_data)
            nodes = result.get("nodes", 0)
            edges = result.get("edges", 0)
        except Exception as exc:
            log.warning("_write_ltm_graph %s: %s", ctx.request.session_id, exc)
            ctx.record.errors.append(f"write_ltm_graph: {exc}")
        ctx.record.nodes_written = nodes
        ctx.record.edges_written = edges


class ExtractRelationsStep:
    """LLM-assisted relation extraction into LTM (llm_assisted mode only)."""

    name = "extract_relations"

    async def run(self, ctx: PromotionContext) -> None:
        if not (ctx.llm_api_key or "").strip():
            log.warning(
                "extract_relations %s: llm_api_key is empty — skipping (rel_edges=0)",
                ctx.request.session_id,
            )
            return
        try:
            rel_edges = await ctx.ltm.extract_and_write_relations(
                stm_data=ctx.stm_data,
                llm_api_key=ctx.llm_api_key,
                session_id=ctx.request.session_id,
            )
            ctx.record.edges_written += rel_edges
        except Exception as exc:
            log.warning("extract_and_write_relations %s failed: %s", ctx.request.session_id, exc)
            ctx.record.errors.append(f"extract_and_write_relations: {exc}")


class CopyVectorStep:
    """Copy stm_chunks/stm_turns to their ltm_* counterparts."""

    name = "copy_vector"

    async def run(self, ctx: PromotionContext) -> None:
        try:
            chunks_promoted, turns_promoted = await ctx.vec.promote_to_ltm(ctx.request.session_id)
            ctx.record.chunks_promoted = chunks_promoted
            ctx.record.turns_promoted = turns_promoted
        except Exception as exc:
            ctx.record.errors.append(f"promote_to_ltm: {exc}")


class IndexEntitiesStep:
    """Embed entity names and upsert them into the vector store."""

    name = "index_entities"

    async def run(self, ctx: PromotionContext) -> None:
        from graphknows.storage.embedder import embed

        entities: list[dict[str, Any]] = ctx.stm_data.get("entities", [])
        if not entities:
            return

        chunk_entity_counts: dict[str, int] = {}
        for chunk in ctx.stm_data.get("chunks", []):
            meta = chunk.get("metadata", {}) or {}
            for name in meta.get("entity_names", []):
                if isinstance(name, str):
                    chunk_entity_counts[name] = chunk_entity_counts.get(name, 0) + 1

        names = [e["name"] for e in entities]
        try:
            emb_array = await asyncio.to_thread(embed, names)
        except Exception as exc:
            log.warning("index_entities: embed failed: %s", exc)
            ctx.record.errors.append(f"entity_embed: {exc}")
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
            await ctx.vec.upsert_entities(entity_records)
            log.info("index_entities: upserted %d entity embeddings", len(entity_records))
        except Exception as exc:
            log.warning("index_entities: upsert failed: %s", exc)
            ctx.record.errors.append(f"entity_upsert: {exc}")


class ChannelPromotionsStep:
    """Run each channel's promote() hook over the freshly-promoted LTM chunks."""

    name = "channel_promotions"

    async def run(self, ctx: PromotionContext) -> None:
        if not ctx.channels:
            return
        stores = ChannelStores(stm=ctx.stm, vec=ctx.vec, ltm=ctx.ltm)
        for channel in ctx.channels:
            try:
                await channel.promote(stores, ctx.request.session_id, ctx.record)
            except Exception as exc:
                log.warning("channel %s promote failed (non-fatal): %s", channel.name, exc)
                ctx.record.errors.append(f"channel_{channel.name}: {exc}")


__all__ = [
    "ChannelPromotionsStep",
    "CopyVectorStep",
    "ExtractRelationsStep",
    "IndexEntitiesStep",
    "ReadStmStep",
    "ResolveEntitiesStep",
    "WriteLtmGraphStep",
]
