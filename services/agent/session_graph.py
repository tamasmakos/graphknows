"""
SessionGraphManager — per-session ephemeral context graphs in Memgraph,
overlaying the persistent Neo4j knowledge graph.
"""
import logging
import time
from typing import Any

from neo4j import GraphDatabase

logger = logging.getLogger("agent.session_graph")


class SessionGraphManager:
    def __init__(
        self,
        memgraph_uri: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
    ):
        self._mg_driver = GraphDatabase.driver(memgraph_uri, auth=None)
        self._n4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )
        logger.info("SessionGraphManager initialised (mg=%s, n4j=%s)", memgraph_uri, neo4j_uri)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, session_id: str) -> str:
        """Create isolated namespace for this session. Returns session_id."""
        with self._mg_driver.session() as s:
            s.run(
                "MERGE (sess:Session {id: $sid}) "
                "ON CREATE SET sess.session_id = $sid, sess.created_at = $ts",
                sid=session_id,
                ts=int(time.time() * 1000),
            )
        logger.info("Session created: %s", session_id)
        return session_id

    def close_session(self, session_id: str) -> None:
        """Delete all session nodes from Memgraph."""
        with self._mg_driver.session() as s:
            s.run("MATCH (n {session_id: $sid}) DETACH DELETE n", sid=session_id)
            # Also delete the Session node itself
            s.run("MATCH (n:Session {id: $sid}) DETACH DELETE n", sid=session_id)
        logger.info("Session closed: %s", session_id)

    def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        with self._mg_driver.session() as s:
            result = s.run("MATCH (n:Session) RETURN n.id AS sid")
            return [r["sid"] for r in result]

    # ------------------------------------------------------------------
    # Seeding from Neo4j
    # ------------------------------------------------------------------

    def seed_from_neo4j(self, session_id: str, entity_names: list[str]) -> int:
        """Pull entities + 1-hop neighbours from Neo4j into Memgraph session."""
        count = 0
        try:
            with self._n4j_driver.session() as n4j:
                for name in entity_names:
                    rows = n4j.run(
                        """
                        MATCH (e:Entity {name: $name})
                        OPTIONAL MATCH (e)<-[:MENTIONS]-(c:Chunk)
                        RETURN e.name AS ename, e.label AS elabel,
                               c.id AS cid, c.text AS ctext
                        LIMIT 50
                        """,
                        name=name,
                    ).data()

                    if not rows:
                        logger.debug("Entity not found in Neo4j: %s", name)
                        continue

                    with self._mg_driver.session() as mg:
                        for row in rows:
                            # Merge entity node
                            mg.run(
                                "MERGE (e:Entity {name: $name, session_id: $sid}) "
                                "ON CREATE SET e.label = $label",
                                name=row["ename"],
                                sid=session_id,
                                label=row.get("elabel", ""),
                            )
                            count += 1

                            # Merge chunk + relationship if present
                            if row.get("cid"):
                                mg.run(
                                    "MERGE (c:Chunk {id: $cid, session_id: $sid}) "
                                    "ON CREATE SET c.text = $text",
                                    cid=row["cid"],
                                    sid=session_id,
                                    text=row.get("ctext", ""),
                                )
                                mg.run(
                                    "MATCH (c:Chunk {id: $cid, session_id: $sid}) "
                                    "MATCH (e:Entity {name: $ename, session_id: $sid}) "
                                    "MERGE (c)-[:MENTIONS]->(e)",
                                    cid=row["cid"],
                                    ename=row["ename"],
                                    sid=session_id,
                                )
                                count += 1

        except Exception as exc:
            logger.warning("Neo4j seed warning (continuing): %s", exc)

        logger.info("Seeded %d nodes for session %s", count, session_id)
        return count

    # ------------------------------------------------------------------
    # Reasoning steps
    # ------------------------------------------------------------------

    def add_reasoning_step(self, session_id: str, step: dict) -> None:
        """Add a ReasoningStep node to the session graph."""
        step_num = step.get("step_num", 0)
        with self._mg_driver.session() as mg:
            mg.run(
                "MERGE (rs:ReasoningStep {session_id: $sid, step_num: $sn}) "
                "ON CREATE SET rs.thought = $thought, rs.action = $action, "
                "rs.observation = $obs, rs.created_at = $ts",
                sid=session_id,
                sn=step_num,
                thought=step.get("thought", ""),
                action=step.get("action", ""),
                obs=step.get("observation", ""),
                ts=int(time.time() * 1000),
            )
            for chunk_id in step.get("chunk_ids", []):
                mg.run(
                    "MERGE (c:Chunk {id: $cid, session_id: $sid})",
                    cid=chunk_id,
                    sid=session_id,
                )
                mg.run(
                    "MATCH (rs:ReasoningStep {session_id: $sid, step_num: $sn}) "
                    "MATCH (c:Chunk {id: $cid, session_id: $sid}) "
                    "MERGE (rs)-[:RETRIEVED]->(c)",
                    sid=session_id,
                    sn=step_num,
                    cid=chunk_id,
                )
        logger.debug("Added reasoning step %d for session %s", step_num, session_id)

    # ------------------------------------------------------------------
    # Graph serialisation for frontend
    # ------------------------------------------------------------------

    def get_session_graph(self, session_id: str) -> dict[str, list]:
        """Return full session graph as {nodes, edges} for D3.js."""
        nodes: dict[str, dict] = {}
        edges: list[dict] = []

        with self._mg_driver.session() as mg:
            # All nodes in session
            result = mg.run(
                "MATCH (n) WHERE n.session_id = $sid RETURN n",
                sid=session_id,
            )
            for record in result:
                n = record["n"]
                node_id = n.element_id
                labels = list(n.labels)
                props = dict(n.items())
                nodes[node_id] = {
                    "id": node_id,
                    "label": labels[0] if labels else "Node",
                    "type": labels[0] if labels else "Node",
                    "properties": props,
                }

            # All edges where both endpoints belong to session
            result = mg.run(
                "MATCH (a)-[r]->(b) "
                "WHERE a.session_id = $sid AND b.session_id = $sid "
                "RETURN a, r, b",
                sid=session_id,
            )
            for record in result:
                a = record["a"]
                b = record["b"]
                r = record["r"]
                src_id = a.element_id
                tgt_id = b.element_id
                # Ensure endpoints in nodes dict
                if src_id not in nodes:
                    props = dict(a.items())
                    nodes[src_id] = {
                        "id": src_id,
                        "label": list(a.labels)[0] if a.labels else "Node",
                        "type": list(a.labels)[0] if a.labels else "Node",
                        "properties": props,
                    }
                if tgt_id not in nodes:
                    props = dict(b.items())
                    nodes[tgt_id] = {
                        "id": tgt_id,
                        "label": list(b.labels)[0] if b.labels else "Node",
                        "type": list(b.labels)[0] if b.labels else "Node",
                        "properties": props,
                    }
                edges.append({
                    "source": src_id,
                    "target": tgt_id,
                    "type": r.type,
                })

        return {"nodes": list(nodes.values()), "edges": edges}

    # ------------------------------------------------------------------
    # Write-back to Neo4j
    # ------------------------------------------------------------------

    def write_back_to_neo4j(
        self, session_id: str, confidence_threshold: float = 0.7
    ) -> int:
        """Write high-confidence ReasoningStep-derived edges back to Neo4j."""
        written = 0
        try:
            with self._mg_driver.session() as mg:
                result = mg.run(
                    "MATCH (rs:ReasoningStep {session_id: $sid})-[r:RETRIEVED]->(c:Chunk {session_id: $sid}) "
                    "WHERE coalesce(r.confidence, 1.0) >= $thresh "
                    "RETURN rs.step_num AS step_num, rs.action AS action, "
                    "rs.thought AS thought, c.id AS chunk_id, "
                    "coalesce(r.confidence, 1.0) AS confidence",
                    sid=session_id,
                    thresh=confidence_threshold,
                )
                rows = result.data()

            with self._n4j_driver.session() as n4j:
                for row in rows:
                    n4j.run(
                        """
                        MERGE (rs:ReasoningStep {session_id: $sid, step_num: $sn})
                        ON CREATE SET rs.action = $action, rs.thought = $thought
                        WITH rs
                        MATCH (c:Chunk {id: $cid})
                        MERGE (rs)-[:RETRIEVED {confidence: $conf, source: 'memgraph_session'}]->(c)
                        """,
                        sid=session_id,
                        sn=row["step_num"],
                        action=row.get("action", ""),
                        thought=row.get("thought", ""),
                        cid=row["chunk_id"],
                        conf=row["confidence"],
                    )
                    written += 1

        except Exception as exc:
            logger.warning("write_back_to_neo4j warning: %s", exc)

        logger.info("Wrote back %d edges to Neo4j for session %s", written, session_id)
        return written

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._mg_driver.close()
        self._n4j_driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
