from __future__ import annotations

import asyncio
import logging
from typing import Any

from neo4j import GraphDatabase

from services.agent.graph.backend import EntityFeatures, NodeFeatures

logger = logging.getLogger(__name__)


class Neo4jGDSBackend:
    def __init__(self, uri: str, user: str, password: str) -> None:
        import warnings as _w

        _w.filterwarnings("ignore", category=DeprecationWarning)
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    async def ensure_indexes(self) -> None:
        def _sync() -> None:
            with self._driver.session() as s:
                # Use SessionEntity label to avoid conflicts with corpus Entity nodes
                s.run(
                    "CREATE INDEX sess_entity_session IF NOT EXISTS FOR (n:SessionEntity) ON (n.session_id)"
                ).consume()
                s.run(
                    "CREATE INDEX sess_entity_id IF NOT EXISTS FOR (n:SessionEntity) ON (n.entity_id)"
                ).consume()

        await asyncio.to_thread(_sync)

    async def write_entities(self, session_id: str, entities: list[Any]) -> None:
        if not entities:
            return

        def _sync() -> None:
            with self._driver.session() as s:
                for e in entities:
                    s.run(
                        "MERGE (n:SessionEntity {entity_id: $eid, session_id: $sid}) SET n += $props",
                        eid=e.id,
                        sid=session_id,
                        props={
                            "name": e.name,
                            "label": e.label,
                            "confidence": e.confidence,
                            "doc_ids": e.doc_ids,
                        },
                    ).consume()

        await asyncio.to_thread(_sync)

    async def write_ontology_relations(
        self,
        session_id: str,
        ontology_triples: list[dict],
        ontology_id: str,
    ) -> None:
        """Write ontology-typed relation triples to Neo4j as ONTOLOGY_REL edges.

        Args:
            session_id: Active session identifier.
            ontology_triples: List of dicts with keys ``subject``, ``predicate``,
                ``object``, and ``confidence``.
            ontology_id: Identifier of the ontology used for extraction.
        """
        if not ontology_triples:
            return

        def _sync() -> None:
            with self._driver.session() as s:
                for triple in ontology_triples:
                    s.run(
                        """
                        MERGE (a:SessionEntity {name: $subj, session_id: $sid})
                          ON CREATE SET a.label = 'CONCEPT', a.entity_id = $subj + '_' + $sid
                        MERGE (b:SessionEntity {name: $obj, session_id: $sid})
                          ON CREATE SET b.label = 'CONCEPT', b.entity_id = $obj + '_' + $sid
                        MERGE (a)-[r:ONTOLOGY_REL {
                            type: $pred,
                            ontology: $ont,
                            session_id: $sid
                        }]->(b)
                          ON CREATE SET r.confidence = $conf
                        """,
                        subj=triple["subject"],
                        obj=triple["object"],
                        pred=triple["predicate"],
                        ont=ontology_id,
                        sid=session_id,
                        conf=triple.get("confidence", 0.5),
                    ).consume()

        await asyncio.to_thread(_sync)

    async def write_relations(self, session_id: str, relations: list[Any]) -> None:
        if not relations:
            return

        def _sync() -> None:
            with self._driver.session() as s:
                for r in relations:
                    s.run(
                        """MATCH (a:SessionEntity {entity_id:$src, session_id:$sid}),
                                (b:SessionEntity {entity_id:$tgt, session_id:$sid})
                           MERGE (a)-[rel:SESSION_RELATES {type:$rtype, session_id:$sid}]->(b)""",
                        src=r.source,
                        tgt=r.target,
                        rtype=r.type,
                        sid=session_id,
                    ).consume()

        await asyncio.to_thread(_sync)

    async def two_hop_neighbours(self, session_id: str, entity_ids: list[str]) -> list[str]:
        if not entity_ids:
            return []

        def _sync() -> list[str]:
            with self._driver.session() as s:
                result = s.run(
                    """
                    MATCH (e:SessionEntity {session_id: $sid})-[*1..2]-(:SessionEntity {session_id: $sid})
                    WHERE e.entity_id IN $ids
                    RETURN DISTINCT e.entity_id AS neighbour_id
                    """,
                    sid=session_id,
                    ids=entity_ids,
                )
                return [r["neighbour_id"] for r in result]

        return await asyncio.to_thread(_sync)

    async def delete_session(self, session_id: str) -> None:
        def _sync() -> None:
            with self._driver.session() as s:
                s.run(
                    "MATCH (n:SessionEntity {session_id: $sid}) DETACH DELETE n", sid=session_id
                ).consume()

        await asyncio.to_thread(_sync)

    async def compute_node_features(self, session_id: str) -> NodeFeatures:
        def _sync() -> NodeFeatures:
            proj_name = f"sess_{session_id[:8]}"
            try:
                with self._driver.session() as session:
                    # GDS 2026: use Cypher projection (nodeFilter not supported)
                    session.run(
                        """
                        CALL gds.graph.project.cypher(
                            $proj,
                            'MATCH (n:SessionEntity) WHERE n.session_id = $sid RETURN id(n) AS id',
                            'MATCH (a:SessionEntity)-[r:SESSION_RELATES]->(b:SessionEntity) WHERE a.session_id = $sid RETURN id(a) AS source, id(b) AS target',
                            {parameters: {sid: $sid}}
                        )
                        """,
                        proj=proj_name,
                        sid=session_id,
                    ).consume()

                    pr = {
                        r["nodeId"]: r["score"]
                        for r in session.run(
                            "CALL gds.pageRank.stream($proj) YIELD nodeId, score",
                            proj=proj_name,
                        ).data()
                    }

                    bt = {
                        r["nodeId"]: r["score"]
                        for r in session.run(
                            "CALL gds.betweenness.stream($proj) YIELD nodeId, score",
                            proj=proj_name,
                        ).data()
                    }

                    lv = {
                        r["nodeId"]: r["communityId"]
                        for r in session.run(
                            "CALL gds.louvain.stream($proj) YIELD nodeId, communityId",
                            proj=proj_name,
                        ).data()
                    }

                    frp = {
                        r["nodeId"]: r["embedding"]
                        for r in session.run(
                            "CALL gds.fastRP.stream($proj, {embeddingDimension: 64}) "
                            "YIELD nodeId, embedding",
                            proj=proj_name,
                        ).data()
                    }

                    features: dict[str, EntityFeatures] = {}
                    for nid in pr:
                        node = session.run(
                            "MATCH (n:SessionEntity) WHERE id(n) = $nid RETURN n.entity_id AS eid",
                            nid=nid,
                        ).single()
                        if node and node["eid"]:
                            eid = node["eid"]
                            features[eid] = EntityFeatures(
                                entity_id=eid,
                                pagerank=pr.get(nid, 0.0),
                                betweenness=bt.get(nid, 0.0),
                                community=lv.get(nid, -1),
                                fastrp_embedding=frp.get(nid, []),
                            )
                    return NodeFeatures(by_entity_id=features)
            finally:
                try:
                    with self._driver.session() as session:
                        session.run("CALL gds.graph.drop($proj)", proj=proj_name).consume()
                except Exception:
                    pass

        return await asyncio.to_thread(_sync)

    def close(self) -> None:
        self._driver.close()
