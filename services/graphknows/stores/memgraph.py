"""Optional Memgraph backend for session entity graphs.

Used as an alternative to Neo4j for live session entity storage.
Memgraph uses the Bolt protocol with no authentication.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MemgraphStore:
    """Async Memgraph store for session-scoped entity graphs.

    Attributes:
        uri: Bolt URI for Memgraph, e.g. bolt://memgraph:7687.
    """

    def __init__(self, uri: str) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=None)

    def close(self) -> None:
        self._driver.close()

    async def ensure_indexes(self) -> None:
        """Create session entity indexes idempotently."""

        def _sync() -> None:
            with self._driver.session() as s:
                for stmt in [
                    "CREATE INDEX ON :Entity(session_id)",
                    "CREATE INDEX ON :Entity(id)",
                ]:
                    try:
                        s.run(stmt).consume()
                    except Exception:
                        pass

        await asyncio.to_thread(_sync)

    async def write_entities(self, session_id: str, entities: list[Any]) -> None:
        """Upsert entity nodes under *session_id*."""
        if not entities:
            return

        def _sync() -> None:
            with self._driver.session() as s:
                for e in entities:
                    s.run(
                        "MERGE (n:Entity {id: $id, session_id: $sid}) SET n += $props",
                        id=e.id,
                        sid=session_id,
                        props={
                            "name": e.name,
                            "label": e.label,
                            "confidence": e.confidence,
                            "doc_ids": e.doc_ids,
                        },
                    )

        await asyncio.to_thread(_sync)

    async def write_relations(self, session_id: str, relations: list[Any]) -> None:
        """Upsert RELATES edges between session entities."""
        if not relations:
            return

        def _sync() -> None:
            with self._driver.session() as s:
                for r in relations:
                    s.run(
                        """MATCH (a:Entity {id:$src, session_id:$sid}),
                                (b:Entity {id:$tgt, session_id:$sid})
                           MERGE (a)-[rel:RELATES {type:$rtype, session_id:$sid}]->(b)""",
                        src=r.source,
                        tgt=r.target,
                        rtype=r.type,
                        sid=session_id,
                    )

        await asyncio.to_thread(_sync)

    async def delete_session(self, session_id: str) -> None:
        """Delete all nodes and edges for *session_id*."""

        def _sync() -> None:
            with self._driver.session() as s:
                s.run("MATCH (n {session_id: $sid}) DETACH DELETE n", sid=session_id)

        await asyncio.to_thread(_sync)

    async def get_session_nodes(self, session_id: str) -> list[dict[str, Any]]:
        """Return all entity nodes for *session_id* as plain dicts."""

        def _sync() -> list[dict[str, Any]]:
            with self._driver.session() as s:
                rows = list(s.run(
                    "MATCH (n:Entity {session_id: $sid}) RETURN properties(n) AS props",
                    sid=session_id,
                ))
            return [dict(r["props"]) for r in rows]

        return await asyncio.to_thread(_sync)

    async def get_session_edges(self, session_id: str) -> list[dict[str, Any]]:
        """Return all RELATES edges for *session_id* as plain dicts."""

        def _sync() -> list[dict[str, Any]]:
            with self._driver.session() as s:
                rows = list(s.run(
                    """MATCH (a:Entity {session_id:$sid})-[r:RELATES]->(b:Entity {session_id:$sid})
                       RETURN a.name AS from, b.name AS to, r.type AS rel_type""",
                    sid=session_id,
                ))
            return [{"from": r["from"], "to": r["to"], "rel_type": r["rel_type"]} for r in rows]

        return await asyncio.to_thread(_sync)
