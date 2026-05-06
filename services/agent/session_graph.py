"""
SessionGraphManager — per-session ephemeral context graphs in Memgraph,
overlaying the persistent Neo4j knowledge graph.

Falls back gracefully to an in-memory store when Memgraph is unavailable.
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
        pgvector_dsn: str | None = None,
    ):
        # Short connection timeout so probe fails fast when Memgraph is down
        self._mg_driver = GraphDatabase.driver(memgraph_uri, auth=None, connection_timeout=2)
        self._n4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        # In-memory fallback when Memgraph is unavailable
        self._memgraph_ok: bool = self._probe_memgraph()
        self._pgvector_dsn = pgvector_dsn
        self._chunk_cache: dict[str, dict[str, Any]] = {}
        # Fallback in-memory session store: session_id → {nodes: dict, edges: list}
        self._mem_sessions: dict[str, dict] = {}
        logger.info(
            "SessionGraphManager initialised (mg=%s ok=%s, n4j=%s)",
            memgraph_uri,
            self._memgraph_ok,
            neo4j_uri,
        )

    def _probe_memgraph(self) -> bool:
        try:
            with self._mg_driver.session() as s:
                s.run("RETURN 1").consume()
            return True
        except Exception as exc:
            logger.warning("Memgraph unavailable — using in-memory fallback: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, session_id: str) -> str:
        """Create isolated namespace for this session. Returns session_id."""
        if self._memgraph_ok:
            try:
                with self._mg_driver.session() as s:
                    s.run(
                        "MERGE (sess:Session {id: $sid}) "
                        "ON CREATE SET sess.session_id = $sid, sess.created_at = $ts",
                        sid=session_id,
                        ts=int(time.time() * 1000),
                    )
            except Exception as exc:
                logger.warning("Memgraph create_session failed, using fallback: %s", exc)
                self._memgraph_ok = False
        # Always ensure in-memory entry exists
        if session_id not in self._mem_sessions:
            self._mem_sessions[session_id] = {"nodes": {}, "edges": []}
        logger.info("Session created: %s (memgraph_ok=%s)", session_id, self._memgraph_ok)
        return session_id

    def close_session(self, session_id: str) -> None:
        """Delete all session nodes from Memgraph."""
        if self._memgraph_ok:
            try:
                with self._mg_driver.session() as s:
                    s.run("MATCH (n {session_id: $sid}) DETACH DELETE n", sid=session_id)
                    s.run("MATCH (n:Session {id: $sid}) DETACH DELETE n", sid=session_id)
            except Exception as exc:
                logger.warning("Memgraph close_session failed: %s", exc)
        self._mem_sessions.pop(session_id, None)
        logger.info("Session closed: %s", session_id)

    def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        if self._memgraph_ok:
            try:
                with self._mg_driver.session() as s:
                    result = s.run("MATCH (n:Session) RETURN n.id AS sid")
                    return [r["sid"] for r in result]
            except Exception as exc:
                logger.warning("Memgraph list_sessions failed: %s", exc)
        return list(self._mem_sessions.keys())

    # ------------------------------------------------------------------
    # Seeding from Neo4j
    # ------------------------------------------------------------------

    def seed_from_neo4j(self, session_id: str, entity_names: list[str]) -> int:
        """Pull entities + 1-hop neighbours from Neo4j into Memgraph session."""
        count = 0
        if session_id not in self._mem_sessions:
            self._mem_sessions[session_id] = {"nodes": {}, "edges": []}
        mem = self._mem_sessions[session_id]

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

                    if self._memgraph_ok:
                        try:
                            with self._mg_driver.session() as mg:
                                for row in rows:
                                    mg.run(
                                        "MERGE (e:Entity {name: $name, session_id: $sid}) "
                                        "ON CREATE SET e.label = $label",
                                        name=row["ename"],
                                        sid=session_id,
                                        label=row.get("elabel", ""),
                                    )
                                    count += 1
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
                            logger.warning("Memgraph seed failed, using fallback: %s", exc)
                            self._memgraph_ok = False

                    # Always store in memory fallback
                    for row in rows:
                        ename = row["ename"]
                        elabel = row.get("elabel", "Entity") or "Entity"
                        node_id = f"entity-{ename}-{session_id}"
                        mem["nodes"][node_id] = {
                            "id": node_id,
                            "label": elabel,
                            "type": elabel,
                            "name": ename,
                            "properties": {
                                "name": ename,
                                "label": elabel,
                                "session_id": session_id,
                            },
                        }
                        count += 1
                        if row.get("cid"):
                            cid = row["cid"]
                            chunk_node_id = f"chunk-{cid}-{session_id}"
                            chunk_props = {
                                "id": cid,
                                "text": row.get("ctext", ""),
                                "session_id": session_id,
                            }
                            mem["nodes"][chunk_node_id] = {
                                "id": chunk_node_id,
                                "label": "Chunk",
                                "type": "Chunk",
                                "name": cid,
                                "properties": chunk_props,
                            }
                            self._chunk_cache[cid] = {**chunk_props}
                            edge_key = f"{chunk_node_id}-MENTIONS-{node_id}"
                            existing_keys = {
                                f"{e['source']}-{e['type']}-{e['target']}" for e in mem["edges"]
                            }
                            if edge_key not in existing_keys:
                                mem["edges"].append(
                                    {
                                        "id": edge_key,
                                        "source": chunk_node_id,
                                        "target": node_id,
                                        "type": "MENTIONS",
                                        "properties": {},
                                    }
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
        """Add a ReasoningStep node to the session graph.

        Args:
            session_id: The active session identifier.
            step: Dict with keys step_num, thought, action, observation, chunk_ids.
        """
        step_num = step.get("step_num", 0)

        # Always store in in-memory fallback
        if session_id not in self._mem_sessions:
            self._mem_sessions[session_id] = {"nodes": {}, "edges": []}
        mem = self._mem_sessions[session_id]
        rs_node_id = f"reasoning-{session_id}-{step_num}"
        mem["nodes"][rs_node_id] = {
            "id": rs_node_id,
            "label": "ReasoningStep",
            "type": "ReasoningStep",
            "name": f"Step {step_num}",
            "properties": {
                "step_num": step_num,
                "thought": step.get("thought", ""),
                "action": step.get("action", ""),
                "observation": step.get("observation", ""),
                "session_id": session_id,
            },
        }
        for chunk_id in step.get("chunk_ids", []):
            chunk_node_id = f"chunk-{chunk_id}-{session_id}"
            if chunk_node_id not in mem["nodes"]:
                mem["nodes"][chunk_node_id] = {
                    "id": chunk_node_id,
                    "label": "Chunk",
                    "type": "Chunk",
                    "name": chunk_id,
                    "properties": {"id": chunk_id, "session_id": session_id},
                }
            edge_id = f"{rs_node_id}-RETRIEVED-{chunk_node_id}"
            if not any(e.get("id") == edge_id for e in mem["edges"]):
                mem["edges"].append(
                    {
                        "id": edge_id,
                        "source": rs_node_id,
                        "target": chunk_node_id,
                        "type": "RETRIEVED",
                        "properties": {},
                    }
                )

        if not self._memgraph_ok:
            logger.debug(
                "Stored reasoning step %d for session %s (in-memory only)", step_num, session_id
            )
            return
        try:
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
        except Exception:
            logger.exception("Failed to add reasoning step for session %s", session_id)
            return
        logger.debug("Added reasoning step %d for session %s", step_num, session_id)

    # ------------------------------------------------------------------
    # Graph serialisation for frontend
    # ------------------------------------------------------------------

    def get_session_graph(self, session_id: str) -> dict[str, list]:
        """Return full session graph as {nodes, edges} for D3.js."""
        # Try Memgraph first
        if self._memgraph_ok:
            try:
                nodes: dict[str, dict] = {}
                edges: list[dict] = []

                with self._mg_driver.session() as mg:
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
                        edge_id = f"{src_id}-{r.type}-{tgt_id}"
                        edges.append(
                            {
                                "id": edge_id,
                                "source": src_id,
                                "target": tgt_id,
                                "type": r.type,
                                "properties": dict(r.items()),
                            }
                        )

                return {"nodes": list(nodes.values()), "edges": edges}
            except Exception as exc:
                logger.warning("Memgraph get_session_graph failed, using fallback: %s", exc)
                self._memgraph_ok = False

        # In-memory fallback
        mem = self._mem_sessions.get(session_id, {"nodes": {}, "edges": []})
        nodes_list = list(mem["nodes"].values())
        # Add name field to properties for D3 label rendering
        for node in nodes_list:
            if "name" not in node:
                node["name"] = node.get("properties", {}).get("name", node["id"])
        return {"nodes": nodes_list, "edges": list(mem["edges"])}

    # ------------------------------------------------------------------
    # Graph enrichment from agent trace
    # ------------------------------------------------------------------

    def add_graph_enrichments(
        self,
        session_id: str,
        relates_to: list[tuple[str, str]],
        topic_nodes: list[dict],
        chunk_nodes: list[dict],
    ) -> None:
        """Upsert RELATES_TO edges, Topic nodes, and annotated Chunk nodes into the session graph.

        Args:
            session_id: The active session identifier.
            relates_to: List of (entity_a_name, entity_b_name) co-occurrence pairs.
            topic_nodes: List of dicts with keys {id, community_id, summary, label,
                         entity_name} — entity_name links entity → topic.
            chunk_nodes: List of dicts with keys {chunk_id, doc_id, excerpt, date,
                         institution, url}.
        """
        if session_id not in self._mem_sessions:
            self._mem_sessions[session_id] = {"nodes": {}, "edges": []}
        mem = self._mem_sessions[session_id]

        def _edge_exists(src: str, tgt: str, etype: str) -> bool:
            return any(
                e["source"] == src and e["target"] == tgt and e["type"] == etype
                for e in mem["edges"]
            )

        # ── RELATES_TO edges ────────────────────────────────────────────
        for entity_a, entity_b in relates_to:
            nid_a = f"entity-{entity_a}-{session_id}"
            nid_b = f"entity-{entity_b}-{session_id}"
            # Ensure both entity nodes exist (stub if not seeded yet)
            for nid, name in ((nid_a, entity_a), (nid_b, entity_b)):
                if nid not in mem["nodes"]:
                    mem["nodes"][nid] = {
                        "id": nid,
                        "label": "Entity",
                        "type": "Entity",
                        "name": name,
                        "properties": {"name": name, "label": "Entity", "session_id": session_id},
                    }
            if not _edge_exists(nid_a, nid_b, "RELATES_TO"):
                mem["edges"].append(
                    {
                        "id": f"{nid_a}-RELATES_TO-{nid_b}",
                        "source": nid_a,
                        "target": nid_b,
                        "type": "RELATES_TO",
                        "properties": {},
                    }
                )
            logger.debug("RELATES_TO edge: %s → %s (session %s)", entity_a, entity_b, session_id)

            # Mirror in Memgraph if available
            if self._memgraph_ok:
                try:
                    with self._mg_driver.session() as mg:
                        for name in (entity_a, entity_b):
                            mg.run(
                                "MERGE (e:Entity {name: $name, session_id: $sid})",
                                name=name,
                                sid=session_id,
                            )
                        mg.run(
                            "MATCH (a:Entity {name: $a, session_id: $sid}) "
                            "MATCH (b:Entity {name: $b, session_id: $sid}) "
                            "MERGE (a)-[:RELATES_TO]->(b)",
                            a=entity_a,
                            b=entity_b,
                            sid=session_id,
                        )
                except Exception as exc:
                    logger.warning("Memgraph RELATES_TO upsert failed: %s", exc)
                    self._memgraph_ok = False

        # ── Topic nodes + IN_COMMUNITY edges ────────────────────────────
        for topic in topic_nodes:
            tid = topic.get("id") or topic.get("community_id", "")
            if not tid:
                continue
            topic_node_id = f"topic-{tid}-{session_id}"
            mem["nodes"][topic_node_id] = {
                "id": topic_node_id,
                "label": "Topic",
                "type": "Topic",
                "name": topic.get("summary", str(tid))[:80],
                "properties": {
                    "community_id": str(tid),
                    "summary": topic.get("summary", ""),
                    "label": topic.get("label", "Topic"),
                    "session_id": session_id,
                },
            }
            logger.debug("Topic node: %s (session %s)", tid, session_id)
            # IN_COMMUNITY edge from entity to topic
            entity_name = topic.get("entity_name", "")
            if entity_name:
                entity_nid = f"entity-{entity_name}-{session_id}"
                if entity_nid not in mem["nodes"]:
                    mem["nodes"][entity_nid] = {
                        "id": entity_nid,
                        "label": "Entity",
                        "type": "Entity",
                        "name": entity_name,
                        "properties": {"name": entity_name, "session_id": session_id},
                    }
                if not _edge_exists(entity_nid, topic_node_id, "IN_COMMUNITY"):
                    mem["edges"].append(
                        {
                            "id": f"{entity_nid}-IN_COMMUNITY-{topic_node_id}",
                            "source": entity_nid,
                            "target": topic_node_id,
                            "type": "IN_COMMUNITY",
                            "properties": {},
                        }
                    )
            # Memgraph
            if self._memgraph_ok:
                try:
                    with self._mg_driver.session() as mg:
                        mg.run(
                            "MERGE (t:Topic {community_id: $cid, session_id: $sid}) "
                            "ON CREATE SET t.summary = $summary, t.label = $lbl",
                            cid=str(tid),
                            sid=session_id,
                            summary=topic.get("summary", ""),
                            lbl=topic.get("label", "Topic"),
                        )
                        if entity_name:
                            mg.run(
                                "MERGE (e:Entity {name: $name, session_id: $sid})",
                                name=entity_name,
                                sid=session_id,
                            )
                            mg.run(
                                "MATCH (e:Entity {name: $ename, session_id: $sid}) "
                                "MATCH (t:Topic {community_id: $cid, session_id: $sid}) "
                                "MERGE (e)-[:IN_COMMUNITY]->(t)",
                                ename=entity_name,
                                cid=str(tid),
                                sid=session_id,
                            )
                except Exception as exc:
                    logger.warning("Memgraph Topic upsert failed: %s", exc)
                    self._memgraph_ok = False

        # ── Chunk nodes with metadata ────────────────────────────────────
        for chunk in chunk_nodes:
            cid = chunk.get("chunk_id", "")
            if not cid:
                continue
            chunk_node_id = f"chunk-{cid}-{session_id}"
            existing_chunk = self._chunk_cache.get(cid, {})
            props: dict = {
                "id": cid,
                "doc_id": chunk.get("doc_id", "") or existing_chunk.get("doc_id", ""),
                "excerpt": chunk.get("excerpt", "") or existing_chunk.get("excerpt", ""),
                "text": chunk.get("text", "") or existing_chunk.get("text", ""),
                "date": chunk.get("date", "") or existing_chunk.get("date", ""),
                "institution": chunk.get("institution", "")
                or existing_chunk.get("institution", ""),
                "url": chunk.get("url", "") or existing_chunk.get("url", ""),
                "session_id": session_id,
            }
            self._chunk_cache[cid] = {**existing_chunk, **props}
            mem["nodes"][chunk_node_id] = {
                "id": chunk_node_id,
                "label": "Chunk",
                "type": "Chunk",
                "name": cid,
                "properties": props,
            }
            logger.debug(
                "Chunk node enriched: %s doc_id=%s (session %s)", cid, props["doc_id"], session_id
            )
            # Memgraph
            if self._memgraph_ok:
                try:
                    with self._mg_driver.session() as mg:
                        mg.run(
                            "MERGE (c:Chunk {id: $cid, session_id: $sid}) "
                            "SET c.doc_id=$doc_id, c.excerpt=$excerpt, c.text=$text, "
                            "c.date=$date, c.institution=$inst, c.url=$url",
                            cid=cid,
                            sid=session_id,
                            doc_id=props["doc_id"],
                            excerpt=props["excerpt"],
                            text=props["text"],
                            date=props["date"],
                            inst=props["institution"],
                            url=props["url"],
                        )
                except Exception as exc:
                    logger.warning("Memgraph Chunk enrichment failed: %s", exc)
                    self._memgraph_ok = False

    def add_cooccurs_edges(
        self,
        session_id: str,
        cooccurs: list[tuple[str, str, int]],
    ) -> None:
        """Upsert CO_OCCURS typed edges (from Neo4j) into the session graph.

        Args:
            session_id: The active session identifier.
            cooccurs: List of (subject_name, object_name, weight) tuples sourced
                from Neo4j CO_OCCURS relationships.
        """
        if session_id not in self._mem_sessions:
            self._mem_sessions[session_id] = {"nodes": {}, "edges": []}
        mem = self._mem_sessions[session_id]

        def _edge_exists(src: str, tgt: str, etype: str) -> bool:
            return any(
                e["source"] == src and e["target"] == tgt and e["type"] == etype
                for e in mem["edges"]
            )

        for subj, obj, weight in cooccurs:
            nid_a = f"entity-{subj}-{session_id}"
            nid_b = f"entity-{obj}-{session_id}"
            existing_edge = next(
                (
                    edge
                    for edge in mem["edges"]
                    if edge["source"] == nid_a
                    and edge["target"] == nid_b
                    and edge["type"] == "CO_OCCURS"
                ),
                None,
            )
            edge_props = dict((existing_edge or {}).get("properties") or {})
            edge_props["weight"] = weight
            for nid, name in ((nid_a, subj), (nid_b, obj)):
                if nid not in mem["nodes"]:
                    mem["nodes"][nid] = {
                        "id": nid,
                        "label": "Entity",
                        "type": "Entity",
                        "name": name,
                        "properties": {"name": name, "label": "Entity", "session_id": session_id},
                    }
            if not existing_edge:
                try:
                    with self._n4j_driver.session() as n4j:
                        row = n4j.run(
                            """
                            MATCH (a:Entity {name: $a})-[r:CO_OCCURS]->(b:Entity {name: $b})
                            RETURN r.via_chunks AS via_chunks
                            LIMIT 1
                            """,
                            a=subj,
                            b=obj,
                        ).single()
                        via_chunks = row["via_chunks"] if row and row.get("via_chunks") else []
                        if via_chunks:
                            edge_props["chunk_ids"] = via_chunks
                except Exception as exc:
                    logger.warning("Neo4j CO_OCCURS provenance lookup failed: %s", exc)
                mem["edges"].append(
                    {
                        "id": f"{subj}-CO_OCCURS-{obj}",
                        "source": nid_a,
                        "target": nid_b,
                        "type": "CO_OCCURS",
                        "properties": edge_props,
                    }
                )
                logger.debug(
                    "CO_OCCURS edge: %s → %s weight=%d (session %s)",
                    subj,
                    obj,
                    weight,
                    session_id,
                )
            else:
                existing_edge["properties"] = edge_props
            # Mirror in Memgraph if available
            if self._memgraph_ok:
                try:
                    with self._mg_driver.session() as mg:
                        for name in (subj, obj):
                            mg.run(
                                "MERGE (e:Entity {name: $name, session_id: $sid})",
                                name=name,
                                sid=session_id,
                            )
                        mg.run(
                            "MATCH (a:Entity {name: $a, session_id: $sid}) "
                            "MATCH (b:Entity {name: $b, session_id: $sid}) "
                            "MERGE (a)-[r:CO_OCCURS]->(b) "
                            "SET r.weight = $weight, r.chunk_ids = $chunk_ids",
                            a=subj,
                            b=obj,
                            sid=session_id,
                            weight=weight,
                            chunk_ids=edge_props.get("chunk_ids", []),
                        )
                except Exception as exc:
                    logger.warning("Memgraph CO_OCCURS upsert failed: %s", exc)
                    self._memgraph_ok = False

    def get_chunk_details(self, chunk_id: str) -> dict[str, Any]:
        """Return the best available provenance payload for a chunk id."""
        cid = str(chunk_id or "").strip()
        if not cid:
            return {"error": "missing_chunk_id"}

        cached = self._chunk_cache.get(cid)
        if cached and (cached.get("text") or cached.get("excerpt")):
            payload = {**cached}
            payload.setdefault("id", cid)
            return payload

        if self._pgvector_dsn:
            try:
                import psycopg2

                conn = psycopg2.connect(self._pgvector_dsn)
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT id, doc_id, text FROM chunks WHERE id = %s",
                        (cid,),
                    )
                    row = cur.fetchone()
                    cur.close()
                finally:
                    conn.close()
                if row:
                    payload = {
                        "id": row[0],
                        "doc_id": row[1] or "",
                        "text": (row[2] or "")[:1200],
                    }
                    self._chunk_cache[cid] = {**payload}
                    return payload
            except Exception as exc:
                logger.warning("Chunk detail lookup failed for %s: %s", cid, exc)

        return {"error": f"chunk_not_found:{cid}", "id": cid}

    def get_relation_sources(
        self,
        session_id: str,
        relation_type: str,
        source_name: str,
        target_name: str,
        chunk_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve chunk-level provenance for a graph edge shown in the session UI."""
        safe_chunk_ids = [str(cid) for cid in (chunk_ids or []) if str(cid)]
        if not safe_chunk_ids:
            return []

        seen: set[str] = set()
        resolved: list[dict[str, Any]] = []
        for cid in safe_chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            payload = self.get_chunk_details(cid)
            payload.setdefault("chunk_id", cid)
            payload.setdefault("relation_type", relation_type)
            payload.setdefault("source_entity", source_name)
            payload.setdefault("target_entity", target_name)
            resolved.append(payload)
        return resolved

    # ------------------------------------------------------------------
    # Write-back to Neo4j
    # ------------------------------------------------------------------

    def write_back_to_neo4j(self, session_id: str, confidence_threshold: float = 0.7) -> int:
        """Write high-confidence ReasoningStep-derived edges back to Neo4j."""
        if not self._memgraph_ok:
            logger.debug(
                "Memgraph unavailable — skipping write_back_to_neo4j for session %s", session_id
            )
            return 0
        written = 0
        try:
            with self._mg_driver.session() as mg:
                result = mg.run(
                    "MATCH (rs:ReasoningStep {session_id: $sid})"
                    "-[r:RETRIEVED]->(c:Chunk {session_id: $sid}) "
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
