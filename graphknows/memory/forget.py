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
        """Delete SESSION subgraphs flushed before before_date and orphaned entities.

        Uses SESSION.flushed_at (the only reliable LTM timestamp) to identify
        stale sessions, then cascades through FILE → CHUNK and DIALOG → TURN.
        ENTITY nodes that become fully isolated (degree-0) afterward are removed.

        WARNING: destructive — pass an ISO-format date string.
        """
        # 1. Identify sessions older than the cutoff (read-only first).
        sess_rows = await self._ltm._qry(
            "MATCH (s:SESSION) WHERE s.flushed_at < $date RETURN s.id AS sid",
            date=before_date,
        )
        session_ids = [r["sid"] for r in sess_rows if r.get("sid")]
        deleted_sessions = len(session_ids)

        deleted_chunks = 0
        deleted_turns = 0
        for sid in session_ids:
            # Delete TURNs via DIALOG → HAS_TURN.
            t_rows = await self._ltm._cmd(
                "MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t:TURN) "
                "DETACH DELETE t RETURN count(t) AS deleted",
                sid=sid,
            )
            deleted_turns += t_rows[0]["deleted"] if t_rows else 0

            # Delete CHUNKs via FILE → HAS_CHUNK.
            c_rows = await self._ltm._cmd(
                "MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK) "
                "DETACH DELETE c RETURN count(c) AS deleted",
                sid=sid,
            )
            deleted_chunks += c_rows[0]["deleted"] if c_rows else 0

            # Delete FILE and DIALOG nodes (edges already detached above).
            await self._ltm._cmd(
                "MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(n) DETACH DELETE n",
                sid=sid,
            )
            # Delete the SESSION node itself.
            await self._ltm._cmd(
                "MATCH (s:SESSION {id: $sid}) DETACH DELETE s",
                sid=sid,
            )

        # 2. Clean up ENTITY nodes that are now fully isolated (degree-0).
        ent_rows = await self._ltm._cmd(
            "MATCH (e:ENTITY) WHERE NOT (e)--() DETACH DELETE e RETURN count(e) as deleted"
        )
        orphaned_entities = ent_rows[0]["deleted"] if ent_rows else 0

        log.info(
            "forget_before %s: sessions=%d chunks=%d turns=%d orphan_entities=%d",
            before_date,
            deleted_sessions,
            deleted_chunks,
            deleted_turns,
            orphaned_entities,
        )
        return {
            "deleted_sessions": deleted_sessions,
            "deleted_chunks": deleted_chunks,
            "deleted_turns": deleted_turns,
            "orphaned_entities_deleted": orphaned_entities,
        }

    async def prune(
        self, _session_id: str | None = None, utterance_ttl_days: int = 180
    ) -> dict[str, Any]:
        """Prune orphaned FILE/CHUNK nodes and stale TURN nodes by TTL.

        Schema: FILE (not Document), CHUNK via HAS_CHUNK, TURN via HAS_TURN,
        DIALOG, ENTITY, MENTIONS. TURN has no timestamp in LTM so TTL uses
        SESSION.flushed_at to find stale turns transitively.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=utterance_ttl_days)).isoformat()
        results: dict[str, Any] = {
            "files_orphaned": 0,
            "chunks_orphaned": 0,
            "dialogs_empty": 0,
            "turns_ttl": 0,
        }

        # Orphaned FILE nodes: not linked to any SESSION.
        rows = await self._ltm._qry(
            "MATCH (f:FILE) WHERE NOT ()-[:IN_SESSION]->(f) RETURN count(f) AS c"
        )
        results["files_orphaned"] = rows[0]["c"] if rows else 0
        if results["files_orphaned"]:
            await self._ltm._cmd("MATCH (f:FILE) WHERE NOT ()-[:IN_SESSION]->(f) DETACH DELETE f")

        # Orphaned CHUNK nodes: no incoming HAS_CHUNK edge (parent FILE deleted
        # or never linked). The MENTIONS disjunct is intentionally dropped —
        # entity-free chunks are valid in llm_free/gliner2 mode.
        rows = await self._ltm._qry(
            "MATCH (c:CHUNK) WHERE NOT ()-[:HAS_CHUNK]->(c) RETURN count(c) AS c"
        )
        results["chunks_orphaned"] = rows[0]["c"] if rows else 0
        if results["chunks_orphaned"]:
            await self._ltm._cmd("MATCH (c:CHUNK) WHERE NOT ()-[:HAS_CHUNK]->(c) DETACH DELETE c")

        # Empty DIALOG nodes: those with no HAS_TURN edges.
        rows = await self._ltm._qry(
            "MATCH (d:DIALOG) WHERE NOT (d)-[:HAS_TURN]->() RETURN count(d) AS c"
        )
        results["dialogs_empty"] = rows[0]["c"] if rows else 0
        if results["dialogs_empty"]:
            await self._ltm._cmd("MATCH (d:DIALOG) WHERE NOT (d)-[:HAS_TURN]->() DETACH DELETE d")

        # Stale TURNs: via sessions flushed before the TTL cutoff.
        rows = await self._ltm._qry(
            "MATCH (s:SESSION)-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t:TURN) "
            "WHERE s.flushed_at < $cutoff "
            "RETURN count(t) AS c",
            cutoff=cutoff,
        )
        results["turns_ttl"] = rows[0]["c"] if rows else 0
        if results["turns_ttl"]:
            await self._ltm._cmd(
                "MATCH (s:SESSION)-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t:TURN) "
                "WHERE s.flushed_at < $cutoff "
                "DETACH DELETE t",
                cutoff=cutoff,
            )

        log.info("prune: %s", results)
        return results
