"""Session end handler — merges session graph from Memgraph into persistent Neo4j KG.

Optionally exports a D3.js HTML visualisation of the session graph when
``EXPORT_GRAPH_HTML=true`` is set in the environment / AgentSettings.

Also exports an intelligence brief HTML via ``export_brief_html`` which is
available as a public helper for the communicate node and any external caller.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.session.manager import SessionGraphManager
from services.agent.graph.html_report import export_brief_html  # noqa: F401 — re-exported


log = logging.getLogger("agent.session_end")


def merge_session_to_neo4j(
    session_id: str,
    sgm: SessionGraphManager,
    *,
    export_html: bool = False,
    html_output_dir: str | Path = "/root/graphknows/output",
) -> dict[str, Any]:
    """Merge all session-scoped nodes and edges from Memgraph into Neo4j.

    Args:
        session_id: The session identifier to merge and optionally visualise.
        sgm: Initialised SessionGraphManager with live Memgraph + Neo4j drivers.
        export_html: When True, render a D3.js HTML graph after merging.
        html_output_dir: Directory to write the HTML file into.

    Returns:
        Dict with keys ``merged_nodes``, ``merged_edges``, ``session_id``,
        and optionally ``graph_html_path`` when export_html is True.
    """
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
                    node_type = str(
                        props.get("ontology_class")
                        or props.get("label")
                        or (list(node.labels)[0] if list(node.labels) else "Entity")
                    )
                    nodes.append({"name": name, "type": node_type})

                edge_rows = mg.run(
                    "MATCH (a {session_id: $sid})-[r]->(b {session_id: $sid}) "
                    "RETURN a, r, b",
                    sid=session_id,
                )
                for row in edge_rows:
                    a, b, r = row["a"], row["b"], row["r"]
                    a_props, b_props = dict(a.items()), dict(b.items())
                    left  = str(a_props.get("name") or a_props.get("id") or "")
                    right = str(b_props.get("name") or b_props.get("id") or "")
                    if not left or not right:
                        continue
                    edges.append({
                        "from":     left,
                        "to":       right,
                        "rel_type": str(r.type),
                        "weight":   float(dict(r.items()).get("weight", 0.5) or 0.5),
                    })
        else:
            mem = getattr(sgm, "_mem_sessions", {}).get(
                session_id, {"nodes": {}, "edges": []}
            )
            node_by_id = mem.get("nodes", {})
            for node in node_by_id.values():
                props = node.get("properties", {})
                name = str(
                    props.get("name") or node.get("name") or props.get("id") or ""
                )
                if not name:
                    continue
                nodes.append({
                    "name": name,
                    "type": str(
                        props.get("ontology_class") or node.get("label") or "Entity"
                    ),
                })
            for edge in mem.get("edges", []):
                src = node_by_id.get(edge.get("source", ""), {})
                tgt = node_by_id.get(edge.get("target", ""), {})
                src_name = str(src.get("properties", {}).get("name") or src.get("name") or "")
                tgt_name = str(tgt.get("properties", {}).get("name") or tgt.get("name") or "")
                if not src_name or not tgt_name:
                    continue
                edges.append({
                    "from":     src_name,
                    "to":       tgt_name,
                    "rel_type": str(edge.get("type", "RELATION")),
                    "weight":   float(edge.get("properties", {}).get("weight", 0.5) or 0.5),
                })

        # ── write back to Neo4j ──────────────────────────────────────────────
        merged_nodes = 0
        merged_edges = 0
        with sgm._n4j_driver.session() as n4j:
            for node in nodes:
                n4j.run(
                    """
                    MERGE (n:Entity {name: $name})
                    ON CREATE SET n.type = $type, n.first_seen = datetime(), n.source_sessions = [$sid]
                    ON MATCH SET
                        n.type = $type,
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

        result: dict[str, Any] = {
            "merged_nodes": merged_nodes,
            "merged_edges": merged_edges,
            "session_id":   session_id,
        }

        # ── optional HTML export ─────────────────────────────────────────────
        if export_html and getattr(sgm, "_memgraph_ok", False):
            try:
                from services.agent.graph.viz import render_session_graph
                html_path = render_session_graph(
                    session_id=session_id,
                    mg_driver=sgm._mg_driver,
                    output_dir=html_output_dir,
                )
                result["graph_html_path"] = str(html_path)
                log.info("Session graph exported: %s", html_path)
            except Exception as viz_exc:
                log.warning("Graph HTML export failed for %s: %s", session_id, viz_exc)

        return result

    except Exception as exc:
        log.exception("Failed to merge session %s to Neo4j", session_id)
        return {
            "error":         str(exc),
            "merged_nodes":  0,
            "merged_edges":  0,
            "session_id":    session_id,
        }
