from __future__ import annotations

import asyncio
import logging
from typing import Any

from neo4j import GraphDatabase

from services.agent.graph.backend import NodeFeatures

logger = logging.getLogger(__name__)


class MemgraphBackend:
    def __init__(self, uri: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=None)

    async def ensure_indexes(self) -> None:
        def _sync() -> None:
            with self._driver.session() as s:
                try:
                    s.run("CREATE INDEX ON :Entity(session_id)").consume()
                except Exception:
                    pass
                try:
                    s.run("CREATE INDEX ON :Entity(id)").consume()
                except Exception:
                    pass

        await asyncio.to_thread(_sync)

    # Label priority for deduplication: higher index = higher priority
    _LABEL_PRIORITY: dict[str, int] = {
        "Entity": 0,
        "CONCEPT": 1,
        "LOCATION": 2,
        "EVENT": 3,
        "DOCUMENT": 4,
        "POLICY": 5,
        "ORGANIZATION": 6,
        "PERSON": 7,
    }

    async def write_entities(self, session_id: str, entities: list[Any]) -> None:
        if not entities:
            return

        def _sync() -> None:
            with self._driver.session() as s:
                for e in entities:
                    s.run(
                        """MERGE (n:Entity {name: $name, session_id: $sid})
                           ON CREATE SET
                               n.label = $label,
                               n.all_labels = [$label],
                               n.id = $id,
                               n.confidence = $confidence,
                               n.doc_ids = $doc_ids
                           ON MATCH SET
                               n.all_labels = CASE
                                   WHEN $label IN n.all_labels THEN n.all_labels
                                   ELSE n.all_labels + [$label]
                               END,
                               n.label = CASE
                                   WHEN n.label = 'Entity' THEN $label
                                   ELSE n.label
                               END,
                               n.confidence = CASE
                                   WHEN $confidence > n.confidence THEN $confidence
                                   ELSE n.confidence
                               END""",
                        name=e.name,
                        label=e.label,
                        id=e.id,
                        sid=session_id,
                        confidence=e.confidence,
                        doc_ids=e.doc_ids,
                    )

        await asyncio.to_thread(_sync)

    async def write_relations(self, session_id: str, relations: list[Any]) -> None:
        if not relations:
            return

        # Build a name lookup: entity id -> entity name, for relation endpoint resolution
        def _sync() -> None:
            with self._driver.session() as s:
                for r in relations:
                    s.run(
                        """MATCH (a:Entity {id:$src, session_id:$sid}), (b:Entity {id:$tgt, session_id:$sid})
                           MERGE (a)-[rel:RELATES {type:$rtype, session_id:$sid}]->(b)""",
                        src=r.source,
                        tgt=r.target,
                        rtype=r.type,
                        sid=session_id,
                    )

        await asyncio.to_thread(_sync)

    async def two_hop_neighbours(self, session_id: str, entity_ids: list[str]) -> list[str]:
        if not entity_ids:
            return []

        def _sync() -> list[str]:
            with self._driver.session() as s:
                result = s.run(
                    """
                    MATCH (e:Entity {session_id: $sid})-[*1..2]-(:Entity {session_id: $sid})
                    WHERE e.id IN $ids
                    RETURN DISTINCT e.id AS neighbour_id
                    """,
                    sid=session_id,
                    ids=entity_ids,
                )
                return [r["neighbour_id"] for r in result]

        return await asyncio.to_thread(_sync)

    async def delete_session(self, session_id: str) -> None:
        def _sync() -> None:
            with self._driver.session() as s:
                s.run("MATCH (n {session_id: $sid}) DETACH DELETE n", sid=session_id)

        await asyncio.to_thread(_sync)

    async def compute_node_features(self, session_id: str) -> NodeFeatures:
        return NodeFeatures(by_entity_id={})

    def close(self) -> None:
        self._driver.close()
