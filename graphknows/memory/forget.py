"""LTM Forget — prune disconnected components, TTL decay, explicit forget."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

log = logging.getLogger("graphknows.memory.forget")


class LTMForget:
    def __init__(self, ltm_store: Any) -> None:
        self._ltm = ltm_store

    async def prune_disconnected(self, _min_component_size: int = 2) -> dict[str, Any]:
        """Drop Entity nodes with no relations (degree=0) or tiny components."""
        rows = await self._ltm._cmd(
            "MATCH (e:ENTITY) WHERE NOT (e)--() WITH e DETACH DELETE e RETURN count(e) as deleted"
        )
        deleted = rows[0]["deleted"] if rows else 0
        log.info("prune_disconnected: removed %d isolated entities", deleted)
        return {"deleted_entities": deleted}

    async def prune_by_ttl(
        self, ttl_days: int = 90, pagerank_floor: float = 0.001
    ) -> dict[str, Any]:
        """Drop entities not seen in ttl_days with pagerank below floor."""
        cutoff = (datetime.now(UTC) - timedelta(days=ttl_days)).isoformat()
        rows = await self._ltm._cmd(
            "MATCH (e:ENTITY) "
            "WHERE e.last_seen < $cutoff AND coalesce(e.pagerank, 0) < $floor "
            "DETACH DELETE e RETURN count(e) as deleted",
            cutoff=cutoff,
            floor=pagerank_floor,
        )
        deleted = rows[0]["deleted"] if rows else 0
        log.info("prune_by_ttl: removed %d stale low-rank entities", deleted)
        return {"deleted_entities": deleted}

    async def forget_community(self, community_id: int) -> dict[str, Any]:
        """Delete all entities exclusively in this community + their Topic node."""
        rows = await self._ltm._cmd(
            "MATCH (e:ENTITY {community_id: $cid}) "
            "WHERE NOT EXISTS { MATCH (e)-[:BELONGS_TO]->(t:TOPIC) WHERE t.community_id <> $cid } "
            "DETACH DELETE e RETURN count(e) as deleted",
            cid=community_id,
        )
        deleted = rows[0]["deleted"] if rows else 0
        await self._ltm._cmd(
            "MATCH (t:TOPIC {community_id: $cid}) DETACH DELETE t", cid=community_id
        )
        log.info("forget_community %d: deleted %d entities + topic node", community_id, deleted)
        return {"community_id": community_id, "deleted_entities": deleted}

    async def forget_before(self, before_date: str) -> dict[str, Any]:
        """Delete Segment nodes older than date and orphaned entities."""
        await self._ltm._cmd(
            "MATCH (seg:Segment) WHERE seg.timestamp < $date DETACH DELETE seg",
            date=before_date,
        )
        rows = await self._ltm._cmd(
            "MATCH (e:ENTITY) WHERE NOT (e)-[:APPEARED_IN]->() AND NOT (e)--() "
            "DETACH DELETE e RETURN count(e) as deleted"
        )
        return {
            "deleted_segments": True,
            "orphaned_entities_deleted": rows[0]["deleted"] if rows else 0,
        }

    async def prune(
        self, _session_id: str | None = None, utterance_ttl_days: int = 180
    ) -> dict[str, Any]:
        """Prune orphaned Document, Chunk, Dialog, Utterance nodes."""
        cutoff = (datetime.now(UTC) - timedelta(days=utterance_ttl_days)).isoformat()
        results: dict[str, Any] = {
            "documents_orphaned": 0,
            "chunks_orphaned": 0,
            "dialogs_empty": 0,
            "utterances_ttl": 0,
        }

        rows = await self._ltm._qry(
            "MATCH (d:Document) WHERE NOT (d)-[:HAS_CHUNK]->() RETURN count(d) AS c"
        )
        results["documents_orphaned"] = rows[0]["c"] if rows else 0
        if results["documents_orphaned"]:
            await self._ltm._cmd("MATCH (d:Document) WHERE NOT (d)-[:HAS_CHUNK]->() DELETE d")

        rows = await self._ltm._qry(
            "MATCH (c:CHUNK) WHERE NOT ()-[:HAS_CHUNK]->(c) OR NOT (c)-[:MENTIONS]->() "
            "RETURN count(c) AS c"
        )
        results["chunks_orphaned"] = rows[0]["c"] if rows else 0
        if results["chunks_orphaned"]:
            await self._ltm._cmd(
                "MATCH (c:CHUNK) WHERE NOT ()-[:HAS_CHUNK]->(c) OR NOT (c)-[:MENTIONS]->() DETACH DELETE c"
            )

        rows = await self._ltm._qry(
            "MATCH (d:DIALOG) WHERE NOT EXISTS { "
            "  MATCH (d)-[:PARTICIPANT]->(:Participant)-[:SAID]->(:Utterance) "
            "} RETURN count(d) AS c"
        )
        results["dialogs_empty"] = rows[0]["c"] if rows else 0
        if results["dialogs_empty"]:
            await self._ltm._cmd(
                "MATCH (d:DIALOG) WHERE NOT EXISTS { "
                "  MATCH (d)-[:PARTICIPANT]->(:Participant)-[:SAID]->(:Utterance) "
                "} DELETE d"
            )

        rows = await self._ltm._qry(
            "MATCH (u:Utterance) WHERE u.timestamp < $cutoff RETURN count(u) AS c",
            cutoff=cutoff,
        )
        results["utterances_ttl"] = rows[0]["c"] if rows else 0
        if results["utterances_ttl"]:
            await self._ltm._cmd(
                "MATCH (u:Utterance) WHERE u.timestamp < $cutoff DETACH DELETE u",
                cutoff=cutoff,
            )

        log.info("prune: %s", results)
        return results
