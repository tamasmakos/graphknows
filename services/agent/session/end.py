"""Session end handler — merges session graph from Memgraph into persistent Neo4j KG.

Optionally exports a D3.js HTML visualisation of the session graph when
``EXPORT_GRAPH_HTML=true`` is set in the environment / AgentSettings.

Also exports an intelligence brief HTML via ``export_brief_html`` which is
available as a public helper for the communicate node and any external caller.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from agent.config import AgentSettings
from agent.session.manager import SessionGraphManager
from agent.session.writeback import WriteBackService
from services.agent.graph.html_report import export_brief_html  # noqa: F401 — re-exported


log = logging.getLogger("agent.session.end")


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

        memgraph_ok = getattr(sgm, "_memgraph_ok", False)

        # ── Determine data source ────────────────────────────────────────────
        # Prefer _mem_sessions (fresh extraction output from the current process)
        # over Memgraph which may hold stale data from previous runs or the
        # web server.  Only read Memgraph when _mem_sessions has no nodes for
        # this session (i.e. the web-server path where DualWriter writes to
        # Memgraph but no in-process extraction was run).
        mem_current = getattr(sgm, "_mem_sessions", {}).get(session_id)
        use_mem_sessions = bool(mem_current and mem_current.get("nodes"))

        log.info(
            "merge_session_to_neo4j START  session=%s  backend=%s",
            session_id,
            "in-memory" if use_mem_sessions else ("memgraph" if memgraph_ok else "in-memory"),
        )

        def _load_from_mem_sessions(mem: dict) -> None:
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

        if use_mem_sessions:
            # Fresh extraction data — use directly
            _load_from_mem_sessions(mem_current)
            log.debug(
                "  in-memory read: %d nodes, %d edges for session=%s",
                len(nodes),
                len(edges),
                session_id,
            )
        elif memgraph_ok:
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

                log.debug(
                    "  Memgraph read: %d nodes for session=%s",
                    len(nodes),
                    session_id,
                )
                edge_rows = mg.run(
                    "MATCH (a {session_id: $sid})-[r]->(b {session_id: $sid}) RETURN a, r, b",
                    sid=session_id,
                )
                for row in edge_rows:
                    a, b, r = row["a"], row["b"], row["r"]
                    a_props, b_props = dict(a.items()), dict(b.items())
                    left = str(a_props.get("name") or a_props.get("id") or "")
                    right = str(b_props.get("name") or b_props.get("id") or "")
                    if not left or not right:
                        continue
                    edges.append(
                        {
                            "from": left,
                            "to": right,
                            "rel_type": str(r.type),
                            "weight": float(dict(r.items()).get("weight", 0.5) or 0.5),
                        }
                    )
            log.debug(
                "  Memgraph read complete: %d nodes, %d edges for session=%s",
                len(nodes),
                len(edges),
                session_id,
            )
            if not nodes:
                raise RuntimeError(
                    f"Memgraph returned no data for session={session_id!r} — cannot merge"
                )
        else:
            raise RuntimeError(
                f"No extraction data in _mem_sessions and Memgraph is unavailable"
                f" for session={session_id!r}"
            )

        # ── write back to Neo4j ──────────────────────────────────────────────
        log.info(
            "  writing to Neo4j: %d nodes, %d edges for session=%s",
            len(nodes),
            len(edges),
            session_id,
        )
        merged_nodes = 0
        merged_edges = 0
        with sgm._n4j_driver.session() as n4j:
            # Remove stale edges from previous runs of this session before
            # re-merging so that deleted or corrected edges do not persist.
            log.debug("  purging stale Neo4j edges for session=%s", session_id)
            n4j.run(
                """
                MATCH ()-[r:RELATION]->()
                WHERE $sid IN coalesce(r.sessions, [])
                WITH r, [s IN r.sessions WHERE s <> $sid] AS remaining
                CALL {
                    WITH r, remaining
                    FOREACH (_ IN CASE WHEN size(remaining) = 0 THEN [1] ELSE [] END |
                        DELETE r
                    )
                    FOREACH (_ IN CASE WHEN size(remaining) > 0 THEN [1] ELSE [] END |
                        SET r.sessions = remaining
                    )
                }
                """,
                sid=session_id,
            )
            log.debug("  purge complete for session=%s", session_id)

            for node in nodes:
                log.debug("  MERGE node  name=%r  type=%s", node["name"], node["type"])
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
                log.debug(
                    "  MERGE edge  %r -[%s]-> %r  w=%.2f",
                    edge["from"],
                    edge["rel_type"],
                    edge["to"],
                    edge["weight"],
                )
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

        log.info(
            "merge_session_to_neo4j DONE  session=%s  merged_nodes=%d  merged_edges=%d",
            session_id,
            merged_nodes,
            merged_edges,
        )

        # ── post-merge graph analytics: dedup → pruning → GDS → write-back ───────────
        _gds_result: dict[str, Any] = {}
        if nodes:
            try:
                import re as _re
                import networkx as nx

                # Build networkx DiGraph from the merged nodes / edges
                G: nx.DiGraph = nx.DiGraph()
                for n in nodes:
                    G.add_node(n["name"], node_type=n["type"])
                for e in edges:
                    G.add_edge(
                        e["from"],
                        e["to"],
                        rel_type=e["rel_type"],
                        confidence=e["weight"],
                        graph_type="entity_relation",
                    )
                log.info(
                    "post-merge graph built  nodes=%d  edges=%d  session=%s",
                    G.number_of_nodes(),
                    G.number_of_edges(),
                    session_id,
                )

                # 1. Entity deduplication — merge nodes with the same
                #    normalised name (lowercase + strip non-alphanumeric)
                def _norm(s: str) -> str:
                    return _re.sub(r"[^a-z0-9]", "", s.lower())

                _norm_to_canon: dict[str, str] = {}
                _aliases: list[tuple[str, str]] = []
                for node_name in list(G.nodes()):
                    key = _norm(node_name)
                    if not key:
                        continue
                    if key not in _norm_to_canon:
                        _norm_to_canon[key] = node_name
                    else:
                        canon = _norm_to_canon[key]
                        _aliases.append((node_name, canon))
                        # redirect in-bound edges
                        for pred in list(G.predecessors(node_name)):
                            if pred != canon and not G.has_edge(pred, canon):
                                G.add_edge(pred, canon, **G.edges[pred, node_name])
                        # redirect out-bound edges
                        for succ in list(G.successors(node_name)):
                            if succ != canon and not G.has_edge(canon, succ):
                                G.add_edge(canon, succ, **G.edges[node_name, succ])
                        G.remove_node(node_name)
                        log.debug("  dedup  merged %r → %r", node_name, canon)

                log.info(
                    "entity dedup  removed=%d duplicates  graph: %d nodes, %d edges",
                    len(_aliases),
                    G.number_of_nodes(),
                    G.number_of_edges(),
                )

                # 2. Graph pruning
                from graphgen.pipeline.graph_cleaning.pruning import prune_graph as _prune

                prune_stats = _prune(
                    G,
                    {
                        "min_edge_confidence": 0.3,
                        "prune_isolated_nodes": True,
                    },
                )
                log.info(
                    "graph pruning  nodes_removed=%d  edges_removed=%d  final: %d nodes, %d edges",
                    prune_stats["nodes_removed"],
                    prune_stats["edges_removed"],
                    prune_stats["final_nodes"],
                    prune_stats["final_edges"],
                )

                # 3. GDS feature engineering
                if G.number_of_nodes() >= 2:
                    pagerank: dict[str, float] = nx.pagerank(G, alpha=0.85, max_iter=200)
                    betweenness: dict[str, float] = nx.betweenness_centrality(G, normalized=True)
                    communities = list(
                        nx.community.greedy_modularity_communities(G.to_undirected())
                    )
                    node_community: dict[str, int] = {
                        node: i for i, comm in enumerate(communities) for node in comm
                    }
                    top5 = sorted(pagerank.items(), key=lambda x: -x[1])[:5]
                    log.info(
                        "GDS features computed  nodes=%d  communities=%d  top_pagerank=%s",
                        G.number_of_nodes(),
                        len(communities),
                        [(n, round(v, 4)) for n, v in top5],
                    )

                    # Write GDS features back to Neo4j
                    with sgm._n4j_driver.session() as n4j:
                        for node_name in G.nodes():
                            pr = pagerank.get(node_name, 0.0)
                            bc = betweenness.get(node_name, 0.0)
                            comm_id = node_community.get(node_name, -1)
                            log.debug(
                                "  GDS  name=%r  pagerank=%.4f  betweenness=%.4f  community=%d",
                                node_name,
                                pr,
                                bc,
                                comm_id,
                            )
                            n4j.run(
                                "MATCH (n:Entity {name: $name}) "
                                "SET n.pagerank = $pr, n.betweenness = $bc, n.community = $comm",
                                name=node_name,
                                pr=round(pr, 6),
                                bc=round(bc, 6),
                                comm=comm_id,
                            )
                    log.info(
                        "GDS features written to Neo4j  nodes=%d  session=%s",
                        G.number_of_nodes(),
                        session_id,
                    )
                    _gds_result = {
                        "gds_communities": len(communities),
                        "gds_top_pagerank": [(n, round(v, 4)) for n, v in top5],
                        "gds_nodes_enriched": G.number_of_nodes(),
                    }
                else:
                    log.info(
                        "GDS skipped — graph too small (%d nodes)  session=%s",
                        G.number_of_nodes(),
                        session_id,
                    )

            except Exception:
                raise

        # ── LTM write-back (fire-and-forget, non-blocking) ────────────────
        async def _async_writeback():
            try:
                _settings = AgentSettings()
                wb = WriteBackService(
                    neo4j_uri=_settings.neo4j_uri,
                    neo4j_user=_settings.neo4j_user,
                    neo4j_password=_settings.neo4j_password,
                )
                await wb.write_back(
                    session_id=session_id,
                    entities=[{"name": n["name"], "type": n.get("type", "Entity"), "confidence": 1.0}
                              for n in nodes],
                    relations=[{"source": e["from"], "target": e["to"],
                                "type": e.get("rel_type", "RELATES"), "confidence": 1.0}
                               for e in edges],
                    query="",
                )
            except Exception:
                log.exception("LTM writeback failed for session %s — non-fatal", session_id)

        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_async_writeback())
            except RuntimeError:
                # No running loop in this thread (e.g. called from asyncio.to_thread)
                asyncio.run(_async_writeback())
        except Exception:
            log.exception("Failed to schedule async writeback for session %s", session_id)

        result: dict[str, Any] = {
            "merged_nodes": merged_nodes,
            "merged_edges": merged_edges,
            "session_id": session_id,
            **_gds_result,
        }

        # ── optional HTML export ─────────────────────────────────────────────
        if export_html and getattr(sgm, "_memgraph_ok", False):
            from services.agent.graph.viz import render_session_graph

            html_path = render_session_graph(
                session_id=session_id,
                mg_driver=sgm._mg_driver,
                output_dir=html_output_dir,
            )
            result["graph_html_path"] = str(html_path)
            log.info("Session graph exported: %s", html_path)

        return result

    finally:
        # Release in-process session accumulators to prevent memory leaks
        from agent.graph.agent import clear_session_state

        clear_session_state(session_id)
