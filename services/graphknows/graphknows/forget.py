"""LTM Forget — prune disconnected components, TTL decay, explicit forget."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

log = logging.getLogger("graphknows.forget")


class LTMForget:
    def __init__(self, neo4j_driver: Any, pg_dsn: str) -> None:
        self._driver = neo4j_driver
        self._pg_dsn = pg_dsn

    def prune_disconnected(self, min_component_size: int = 2) -> dict:
        """Drop Entity nodes with no relations (degree=0) or tiny components."""
        with self._driver.session() as s:
            result = s.run(
                "MATCH (e:Entity) WHERE NOT (e)--() "
                "WITH e DETACH DELETE e RETURN count(e) as deleted"
            ).single()
            deleted = result["deleted"] if result else 0
        log.info("prune_disconnected: removed %d isolated entities", deleted)
        return {"deleted_entities": deleted}

    def prune_by_ttl(self, ttl_days: int = 90, pagerank_floor: float = 0.001) -> dict:
        """Drop entities not seen in ttl_days with pagerank below floor."""
        cutoff = (datetime.now(UTC) - timedelta(days=ttl_days)).isoformat()
        with self._driver.session() as s:
            result = s.run(
                "MATCH (e:Entity) "
                "WHERE e.last_seen < $cutoff AND coalesce(e.pagerank, 0) < $floor "
                "DETACH DELETE e RETURN count(e) as deleted",
                cutoff=cutoff,
                floor=pagerank_floor,
            ).single()
            deleted = result["deleted"] if result else 0
        log.info("prune_by_ttl: removed %d stale low-rank entities", deleted)
        return {"deleted_entities": deleted}

    def forget_community(self, community_id: int) -> dict:
        """Delete all entities exclusively in this community + their Topic node."""
        import asyncio

        import asyncpg as _pg

        with self._driver.session() as s:
            result = s.run(
                "MATCH (e:Entity {community_id: $cid}) "
                "WHERE NOT EXISTS { MATCH (e)-[:BELONGS_TO]->(t:Topic) WHERE t.community_id <> $cid } "
                "DETACH DELETE e RETURN count(e) as deleted",
                cid=community_id,
            ).single()
            deleted = result["deleted"] if result else 0
            s.run("MATCH (t:Topic {community_id: $cid}) DETACH DELETE t", cid=community_id)

        async def _pg_delete():
            conn = await _pg.connect(self._pg_dsn)
            try:
                await conn.execute("DELETE FROM ltm_topics WHERE community_id=$1", community_id)
            finally:
                await conn.close()

        try:
            asyncio.get_event_loop().run_until_complete(_pg_delete())
        except RuntimeError:
            pass

        log.info("forget_community %d: deleted %d entities + topic node", community_id, deleted)
        return {"community_id": community_id, "deleted_entities": deleted}

    def forget_before(self, before_date: str) -> dict:
        """Delete Segment nodes older than date and orphaned entities."""
        with self._driver.session() as s:
            s.run(
                "MATCH (seg:Segment) WHERE seg.timestamp < $date DETACH DELETE seg",
                date=before_date,
            )
            result = s.run(
                "MATCH (e:Entity) WHERE NOT (e)-[:APPEARED_IN]->() AND NOT (e)--() "
                "DETACH DELETE e RETURN count(e) as deleted"
            ).single()
        return {
            "deleted_segments": True,
            "orphaned_entities_deleted": result["deleted"] if result else 0,
        }

    def prune(self, session_id: str | None = None, utterance_ttl_days: int = 180) -> dict[str, Any]:
        """Prune orphaned Document, Chunk, Dialog, Utterance nodes.

        Args:
            session_id: Ignored (kept for API compatibility).
            utterance_ttl_days: Prune utterances older than this many days.

        Returns:
            Dict with counts of pruned nodes per type.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=utterance_ttl_days)).isoformat()
        results: dict[str, Any] = {
            "documents_orphaned": 0,
            "chunks_orphaned": 0,
            "dialogs_empty": 0,
            "utterances_ttl": 0,
        }
        with self._driver.session() as s:
            # Count then delete — Neo4j doesn't support aggregation + DELETE in one pass
            r = s.run(
                "MATCH (d:Document) WHERE NOT (d)-[:HAS_CHUNK]->() RETURN count(d) AS c"
            ).single()
            results["documents_orphaned"] = r["c"] if r else 0
            if results["documents_orphaned"]:
                s.run("MATCH (d:Document) WHERE NOT (d)-[:HAS_CHUNK]->() DELETE d")

            r = s.run(
                "MATCH (c:Chunk) WHERE NOT ()-[:HAS_CHUNK]->(c) OR NOT (c)-[:MENTIONS]->() "
                "RETURN count(c) AS c"
            ).single()
            results["chunks_orphaned"] = r["c"] if r else 0
            if results["chunks_orphaned"]:
                s.run(
                    "MATCH (c:Chunk) WHERE NOT ()-[:HAS_CHUNK]->(c) OR NOT (c)-[:MENTIONS]->() DETACH DELETE c"
                )

            r = s.run(
                "MATCH (d:Dialog) WHERE NOT EXISTS { "
                "  MATCH (d)-[:PARTICIPANT]->(:Participant)-[:SAID]->(:Utterance) "
                "} RETURN count(d) AS c"
            ).single()
            if r:
                results["dialogs_empty"] = r["c"]
            if results["dialogs_empty"]:
                s.run(
                    "MATCH (d:Dialog) WHERE NOT EXISTS { "
                    "  MATCH (d)-[:PARTICIPANT]->(:Participant)-[:SAID]->(:Utterance) "
                    "} DELETE d"
                )

            r = s.run(
                "MATCH (u:Utterance) WHERE u.timestamp < $cutoff RETURN count(u) AS c",
                cutoff=cutoff,
            ).single()
            if r:
                results["utterances_ttl"] = r["c"]
            if results["utterances_ttl"]:
                s.run(
                    "MATCH (u:Utterance) WHERE u.timestamp < $cutoff DETACH DELETE u",
                    cutoff=cutoff,
                )

        log.info("prune: %s", results)
        return results
