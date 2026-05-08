"""Session end handler — merges session graph from Memgraph into persistent Neo4j KG."""
from __future__ import annotations

import logging
from typing import Any

from agent.session.manager import SessionGraphManager


log = logging.getLogger("agent.session_end")


def merge_session_to_neo4j(session_id: str, sgm: SessionGraphManager) -> dict[str, Any]:
    """Merge all session-scoped nodes and edges from Memgraph into Neo4j."""
    try:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        if getattr(sgm, "_memgraph_ok", False):
            with sgm._mg_driver.session() as mg:
                node_rows = mg.run(
                    "MATCH (n {session_id: $sid}) RETURN n",
                    sid=session_id,
                )
                for row in node_rows:
                    node = row["n"]
                    props = dict(node.items())
                    name = str(props.get("name") or props.get("id") or "")
                    if not name:
                        continue
                    node_type = str(props.get("ontology_class") or props.get("label") or list(node.labels)[0] if list(node.labels) else "Entity")
                    nodes.append(
                        {
                            "name": name,
                            "type": node_type,
                        }
                    )

                edge_rows = mg.run(
                    "MATCH (a {session_id: $sid})-[r]->(b {session_id: $sid}) RETURN a, r, b",
                    sid=session_id,
                )
                for row in edge_rows:
                    a = row["a"]
                    b = row["b"]
                    r = row["r"]
                    a_props = dict(a.items())
                    b_props = dict(b.items())
                    left_name = str(a_props.get("name") or a_props.get("id") or "")
                    right_name = str(b_props.get("name") or b_props.get("id") or "")
                    if not left_name or not right_name:
                        continue
                    edges.append(
                        {
                            "from": left_name,
                            "to": right_name,
                            "rel_type": str(r.type),
                            "weight": float(dict(r.items()).get("weight", 0.5) or 0.5),
                        }
                    )
        else:
            mem = getattr(sgm, "_mem_sessions", {}).get(session_id, {"nodes": {}, "edges": []})
            node_by_id = mem.get("nodes", {})
            for node in node_by_id.values():
                props = node.get("properties", {})
                name = str(props.get("name") or node.get("name") or props.get("id") or "")
                if not name:
                    continue
                nodes.append(
                    {
                        "name": name,
                        "type": str(props.get("ontology_class") or node.get("label") or "Entity"),
                    }
                )
            for edge in mem.get("edges", []):
                src = node_by_id.get(edge.get("source", ""), {})
                tgt = node_by_id.get(edge.get("target", ""), {})
                src_name = str(src.get("properties", {}).get("name") or src.get("name") or "")
                tgt_name = str(tgt.get("properties", {}).get("name") or tgt.get("name") or "")
                if not src_name or not tgt_name:
                    continue
                edges.append(
                    {
                        "from": src_name,
                        "to": tgt_name,
                        "rel_type": str(edge.get("type", "RELATION")),
                        "weight": float(edge.get("properties", {}).get("weight", 0.5) or 0.5),
                    }
                )

        merged_nodes = 0
        merged_edges = 0
        with sgm._n4j_driver.session() as n4j:
            for node in nodes:
                n4j.run(
                    """
                    MERGE (n:Entity {name: $name, type: $type})
                    ON CREATE SET n.first_seen = datetime(), n.source_sessions = [$sid]
                    ON MATCH SET
                        n.source_sessions = CASE
                            WHEN $sid IN coalesce(n.source_sessions, []) THEN n.source_sessions
                            ELSE coalesce(n.source_sessions, []) + [$sid]
                        END,
                        n.last_seen = datetime()
                    """,
                    name=node["name"],
                    type=node["type"],
                    sid=session_id,
                )
                merged_nodes += 1

            for edge in edges:
                n4j.run(
                    """
                    MATCH (a:Entity {name: $a_name})
                    MATCH (b:Entity {name: $b_name})
                    MERGE (a)-[r:RELATION {type: $rel_type}]->(b)
                    ON CREATE SET r.weight = $weight, r.sessions = [$sid]
                    ON MATCH SET
                        r.weight = (coalesce(r.weight, $weight) + $weight) / 2.0,
                        r.sessions = CASE
                            WHEN $sid IN coalesce(r.sessions, []) THEN r.sessions
                            ELSE coalesce(r.sessions, []) + [$sid]
                        END
                    """,
                    a_name=edge["from"],
                    b_name=edge["to"],
                    rel_type=edge["rel_type"],
                    weight=edge["weight"],
                    sid=session_id,
                )
                merged_edges += 1

        return {
            "merged_nodes": merged_nodes,
            "merged_edges": merged_edges,
            "session_id": session_id,
        }
    except Exception as exc:
        log.exception("Failed to merge session %s to Neo4j", session_id)
        return {
            "error": str(exc),
            "merged_nodes": 0,
            "merged_edges": 0,
            "session_id": session_id,
        }
