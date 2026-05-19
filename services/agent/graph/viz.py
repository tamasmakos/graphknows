"""Session graph HTML visualisation generator.

Reads entities, relations, and GDS scores from Memgraph and renders a
self-contained D3.js HTML file. Activated via EXPORT_GRAPH_HTML=true.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("agent.graph.viz")

# ── Palette / type map ────────────────────────────────────────────────────────
_COMMUNITY_PALETTE = [
    {"base": "#4f8ef7", "glow": "rgba(79,142,247,0.40)"},
    {"base": "#f7914f", "glow": "rgba(247,145,79,0.40)"},
    {"base": "#4fc97f", "glow": "rgba(79,201,127,0.40)"},
    {"base": "#c94f8e", "glow": "rgba(201,79,142,0.40)"},
    {"base": "#e8c84f", "glow": "rgba(232,200,79,0.40)"},
    {"base": "#4fc8c9", "glow": "rgba(79,200,201,0.40)"},
]
_TYPE_ICON = {
    "Organization": "",
    "Person": "◎",
    "Legislation": "▣",
    "Policy": "◈",
    "Location": "◉",
    "Concept": "◇",
    "Entity": "○",
}


def _scale(v: float, mn: float, mx: float, lo: float = 18, hi: float = 44) -> float:
    if mx == mn:
        return (lo + hi) / 2
    return lo + (v - mn) / (mx - mn) * (hi - lo)


def render_session_graph(
    session_id: str,
    mg_driver: Any,
    output_dir: str | Path,
) -> Path:
    """Query Memgraph for session data, run GDS, write HTML, return path.

    Args:
        session_id: The session whose graph to visualise.
        mg_driver: Live neo4j.GraphDatabase driver pointed at Memgraph.
        output_dir: Directory to write the HTML file into (created if absent).

    Returns:
        Path to the written HTML file.

    Raises:
        RuntimeError: If the Memgraph query or render fails.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with mg_driver.session() as s:
        # ── raw nodes ────────────────────────────────────────────────────────
        nodes_raw = s.run(
            "MATCH (n {session_id: $sid}) "
            "RETURN n.id AS id, n.name AS name, n.label AS label, "
            "       n.ontology_class AS otype, n.confidence AS conf "
            "LIMIT 500",
            sid=session_id,
        ).data()

        if not nodes_raw:
            raise RuntimeError(f"No nodes found for session {session_id!r}")

        # Build name → int-id map from whatever identifier is present
        seen: dict[str, int] = {}
        entities: list[dict] = []
        for row in nodes_raw:
            name = str(row.get("name") or row.get("id") or "")
            if not name or name in seen:
                continue
            seen[name] = len(entities)
            entities.append(
                {
                    "name": name,
                    "type": str(row.get("otype") or row.get("label") or "Entity"),
                    "conf": float(row.get("conf") or 0.0),
                }
            )

        if not entities:
            raise RuntimeError(f"Session {session_id!r} has no named nodes")

        # ── raw edges ────────────────────────────────────────────────────────
        edges_raw = s.run(
            "MATCH (a {session_id:$sid})-[r]->(b {session_id:$sid}) "
            "RETURN coalesce(a.name, a.id) AS src, "
            "       type(r) AS rel, "
            "       coalesce(b.name, b.id) AS dst "
            "LIMIT 2000",
            sid=session_id,
        ).data()

        links: list[dict] = [
            {
                "source": seen[r["src"]],
                "target": seen[r["dst"]],
                "label": r["rel"].replace("_", " "),
            }
            for r in edges_raw
            if r.get("src") in seen and r.get("dst") in seen
        ]

        # ── GDS: PageRank ────────────────────────────────────────────────────
        pr: dict[str, float] = {}
        try:
            for row in s.run(
                """
                CALL pagerank.get() YIELD node, rank
                WITH node, rank WHERE node.session_id = $sid
                RETURN coalesce(node.name, node.id) AS name, rank
            """,
                sid=session_id,
            ).data():
                if row.get("name"):
                    pr[row["name"]] = float(row["rank"])
        except Exception as exc:
            log.warning("PageRank skipped: %s", exc)

        # ── GDS: Betweenness ────────────────────────────────────────────────
        bc: dict[str, float] = {}
        try:
            for row in s.run(
                """
                CALL betweenness_centrality.get() YIELD node, betweenness_centrality
                WITH node, betweenness_centrality WHERE node.session_id = $sid
                RETURN coalesce(node.name, node.id) AS name,
                       betweenness_centrality AS bc
            """,
                sid=session_id,
            ).data():
                if row.get("name"):
                    bc[row["name"]] = float(row["bc"])
        except Exception as exc:
            log.warning("Betweenness skipped: %s", exc)

        # ── GDS: Community ──────────────────────────────────────────────────
        cd: dict[str, int] = {}
        try:
            for row in s.run(
                """
                CALL community_detection.get() YIELD node, community_id
                WITH node, community_id WHERE node.session_id = $sid
                RETURN coalesce(node.name, node.id) AS name, community_id
            """,
                sid=session_id,
            ).data():
                if row.get("name"):
                    cd[row["name"]] = int(row["community_id"])
        except Exception as exc:
            log.warning("Community detection skipped: %s", exc)

    # ── build vis data ───────────────────────────────────────────────────────
    pr_vals = list(pr.values()) or [0.0]
    pr_mn, pr_mx = min(pr_vals), max(pr_vals)

    comm_ids = sorted(set(cd.values()))
    comm_pal = {c: _COMMUNITY_PALETTE[i % len(_COMMUNITY_PALETTE)] for i, c in enumerate(comm_ids)}
    default_pal = _COMMUNITY_PALETTE[0]

    nodes_js: list[dict] = []
    for e in entities:
        name = e["name"]
        comm = cd.get(name)
        pal = comm_pal.get(comm, default_pal) if comm is not None else default_pal
        nodes_js.append(
            {
                "id": seen[name],
                "name": name,
                "label": name.replace("_", " "),
                "type": e["type"],
                "icon": _TYPE_ICON.get(e["type"], "○"),
                "community": comm,
                "color": pal["base"],
                "glow": pal["glow"],
                "r": round(_scale(pr.get(name, pr_mn), pr_mn, pr_mx), 1),
                "pr": round(pr.get(name, 0.0), 5),
                "bc": round(bc.get(name, 0.0), 5),
                "conf": round(e["conf"], 3),
            }
        )

    # ── sidebar HTML fragments ───────────────────────────────────────────────
    legend_html = ""
    for cid in comm_ids:
        members = [k.replace("_", " ") for k, v in cd.items() if v == cid]
        col = comm_pal.get(cid, default_pal)["base"]
        legend_html += (
            f'<div class="legend-row">'
            f'<span class="lbullet" style="background:{col}"></span>'
            f'<div><b style="color:{col}">Community {cid}</b>'
            f'<br><span class="lmembers">{", ".join(members)}</span></div></div>'
        )

    table_html = ""
    for n in sorted(nodes_js, key=lambda x: -x["pr"]):
        col = (
            comm_pal.get(n["community"], default_pal)["base"]
            if n["community"] is not None
            else "#888"
        )
        table_html += (
            f"<tr>"
            f"<td><span class='ttype'>{n['icon']}</span> {n['label']}</td>"
            f"<td style='color:{col};font-weight:600'>{n['community'] if n['community'] is not None else '—'}</td>"
            f"<td class='num'>{n['pr']:.5f}</td>"
            f"<td class='num'>{n['bc']:.5f}</td>"
            f"</tr>"
        )

    n_nodes = len(nodes_js)
    n_edges = len(links)
    n_comms = len(comm_ids)
    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Graphknows · {session_id}</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
  :root {{
    --bg:#08090d; --surface:#0d1017; --border:#1a1f30;
    --text:#c8cfe8; --muted:#4a5270; --accent:#4f8ef7;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:'Inter','Segoe UI',sans-serif;
        display:flex;flex-direction:column;height:100vh;overflow:hidden;}}
  header{{padding:10px 20px;background:var(--surface);border-bottom:1px solid var(--border);
          display:flex;align-items:center;justify-content:space-between;flex-shrink:0;}}
  .logo{{display:flex;align-items:center;gap:10px;}}
  .logo-mark{{font-size:1.3rem;}}
  .logo-text{{font-size:.9rem;font-weight:700;color:#fff;letter-spacing:-.01em;}}
  .logo-sub{{font-size:.68rem;color:var(--muted);margin-top:1px;}}
  .header-meta{{font-size:.68rem;color:var(--muted);text-align:right;line-height:1.8;}}
  #layout{{display:flex;flex:1;overflow:hidden;}}
  #canvas{{flex:1;position:relative;background:var(--bg);}}
  svg{{width:100%;height:100%;}}
  #panel{{width:280px;flex-shrink:0;background:var(--surface);
          border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;}}
  .panel-inner{{overflow-y:auto;flex:1;padding:14px;display:flex;flex-direction:column;gap:12px;}}
  .card{{background:#0f1320;border:1px solid var(--border);border-radius:8px;padding:12px;}}
  .card-title{{font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
               color:var(--muted);margin-bottom:9px;}}
  .legend-row{{display:flex;align-items:flex-start;gap:9px;margin-bottom:7px;font-size:.75rem;}}
  .lbullet{{width:9px;height:9px;border-radius:50%;flex-shrink:0;margin-top:3px;}}
  .lmembers{{color:var(--muted);font-size:.68rem;line-height:1.5;}}
  .type-grid{{display:flex;flex-wrap:wrap;gap:5px;}}
  .type-chip{{background:#12172a;border:1px solid var(--border);border-radius:20px;
              padding:2px 9px;font-size:.68rem;color:var(--muted);}}
  table{{width:100%;border-collapse:collapse;font-size:.68rem;}}
  th{{color:var(--muted);text-align:left;padding:3px 5px;border-bottom:1px solid var(--border);
      font-weight:500;letter-spacing:.03em;}}
  td{{padding:4px 5px;border-bottom:1px solid #0a0c14;vertical-align:middle;}}
  .ttype{{opacity:.5;margin-right:2px;}}
  .num{{font-variant-numeric:tabular-nums;color:#6a7aa0;}}
  /* graph */
  .link{{fill:none;stroke-width:1.2px;}}
  .link-label{{font-size:8px;fill:#2a3255;pointer-events:none;
               font-family:'Inter','Segoe UI',sans-serif;}}
  .node-glow{{pointer-events:none;}}
  .node-ring{{cursor:pointer;}}
  .node-icon{{pointer-events:none;dominant-baseline:central;text-anchor:middle;}}
  .node-name{{pointer-events:none;dominant-baseline:auto;text-anchor:middle;
              font-family:'Inter','Segoe UI',sans-serif;font-weight:500;fill:#c8cfe8;}}
  /* tooltip */
  #tip{{position:fixed;pointer-events:none;display:none;background:#0f1320;
        border:1px solid #2a3050;border-radius:8px;padding:10px 13px;font-size:.75rem;
        line-height:1.6;max-width:210px;z-index:999;box-shadow:0 8px 28px rgba(0,0,0,.7);}}
  #tip b{{color:#fff;font-size:.8rem;}}
  #tip .tip-sub{{color:var(--muted);font-size:.67rem;margin-bottom:5px;}}
  #tip .tip-row{{display:flex;justify-content:space-between;gap:14px;}}
  #tip .tip-val{{color:var(--accent);font-variant-numeric:tabular-nums;}}
</style>
</head>
<body>
<header>
  <div class="logo">
    <span class="logo-mark"></span>
    <div>
      <div class="logo-text">Graphknows</div>
      <div class="logo-sub">Session · {session_id[:16]}…</div>
    </div>
  </div>
  <div class="header-meta">
    {n_nodes} entities &nbsp;·&nbsp; {n_edges} relations &nbsp;·&nbsp;
    {n_comms} communities<br>
    Node size = PageRank &nbsp;·&nbsp; Colour = Community &nbsp;·&nbsp; {ts}
  </div>
</header>
<div id="layout">
  <div id="canvas"><svg id="svg"></svg></div>
  <div id="panel">
    <div class="panel-inner">
      <div class="card">
        <div class="card-title">Communities</div>
        {legend_html if legend_html else '<span style="color:var(--muted);font-size:.72rem">No community data</span>'}
      </div>
      <div class="card">
        <div class="card-title">Node Types</div>
        <div class="type-grid">
          {"".join(f'<div class="type-chip">{icon} {t}</div>' for t, icon in _TYPE_ICON.items())}
        </div>
      </div>
      <div class="card">
        <div class="card-title">GDS · PageRank descending</div>
        <table>
          <tr><th>Entity</th><th>Comm</th><th>PR</th><th>BC</th></tr>
          {table_html}
        </table>
      </div>
    </div>
  </div>
</div>
<div id="tip"></div>
<script>
const NODES={json.dumps(nodes_js)};
const LINKS={json.dumps(links)};

const svg=d3.select("#svg");
const el=document.getElementById("canvas");

function draw(){{
  svg.selectAll("*").remove();
  const W=el.clientWidth,H=el.clientHeight;
  svg.attr("viewBox",`0 0 ${{W}} ${{H}}`);

  const defs=svg.append("defs");

  // one radial gradient + blur filter per community colour
  const glows=[...new Set(NODES.map(n=>n.glow))];
  const glowIdx=Object.fromEntries(glows.map((g,i)=>[g,i]));
  glows.forEach((glow,i)=>{{
    const f=defs.append("filter").attr("id",`gf${{i}}`)
      .attr("x","-80%").attr("y","-80%").attr("width","260%").attr("height","260%");
    f.append("feGaussianBlur").attr("stdDeviation","6").attr("result","blur");
    const m=f.append("feMerge");
    m.append("feMergeNode").attr("in","blur");
    m.append("feMergeNode").attr("in","SourceGraphic");
  }});

  defs.append("marker").attr("id","arr").attr("viewBox","0 -4 8 8")
    .attr("refX",8).attr("refY",0).attr("markerWidth",5).attr("markerHeight",5)
    .attr("orient","auto")
    .append("path").attr("d","M0,-4L8,0L0,4").attr("fill","#252c48");

  const g=svg.append("g");
  svg.call(d3.zoom().scaleExtent([.2,5]).on("zoom",e=>g.attr("transform",e.transform)));

  const sim=d3.forceSimulation(NODES)
    .force("link",d3.forceLink(LINKS).id(d=>d.id).distance(150).strength(.5))
    .force("charge",d3.forceManyBody().strength(-380))
    .force("center",d3.forceCenter(W/2,H/2))
    .force("collide",d3.forceCollide(d=>d.r+20));

  // edges
  const link=g.append("g").selectAll("path")
    .data(LINKS).join("path")
    .attr("class","link")
    .attr("stroke",d=>{{
      const s=NODES.find(n=>n.id===d.source.id||n.id===d.source);
      return s ? s.color+"55" : "#2a3255";
    }})
    .attr("marker-end","url(#arr)");

  const llabel=g.append("g").selectAll("text")
    .data(LINKS).join("text")
    .attr("class","link-label").attr("text-anchor","middle")
    .text(d=>d.label);

  // nodes
  const node=g.append("g").selectAll("g")
    .data(NODES).join("g")
    .call(d3.drag()
      .on("start",(e,d)=>{{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;}})
      .on("drag", (e,d)=>{{d.fx=e.x;d.fy=e.y;}})
      .on("end",  (e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}));

  // outer glow ring
  node.append("circle").attr("class","node-glow")
    .attr("r",d=>d.r+6)
    .attr("fill","none")
    .attr("stroke",d=>d.glow)
    .attr("stroke-width",6)
    .attr("filter",d=>`url(#gf${{glowIdx[d.glow]}})`);

  // main ring
  node.append("circle").attr("class","node-ring")
    .attr("r",d=>d.r)
    .attr("fill",d=>d.color+"18")
    .attr("stroke",d=>d.color)
    .attr("stroke-width",1.8);

  // type icon
  node.append("text").attr("class","node-icon")
    .attr("font-size",d=>d.r*.85)
    .attr("fill",d=>d.color+"cc")
    .text(d=>d.icon);

  // label below
  node.append("text").attr("class","node-name")
    .attr("font-size",10)
    .attr("dy",d=>d.r+13)
    .text(d=>d.label);

  // tooltip
  const tip=document.getElementById("tip");
  node
    .on("mouseenter",(e,d)=>{{
      tip.innerHTML=`<b>${{d.label}}</b>
        <div class="tip-sub">${{d.icon}} ${{d.type}}${{d.community!=null?" · Community "+d.community:""}}</div>
        <div class="tip-row"><span>PageRank</span><span class="tip-val">${{d.pr.toFixed(5)}}</span></div>
        <div class="tip-row"><span>Betweenness</span><span class="tip-val">${{d.bc.toFixed(5)}}</span></div>`;
      tip.style.display="block";
    }})
    .on("mousemove",e=>{{tip.style.left=(e.clientX+16)+"px";tip.style.top=(e.clientY+14)+"px";}})
    .on("mouseleave",()=>tip.style.display="none");

  sim.on("tick",()=>{{
    link.attr("d",d=>{{
      const dx=d.target.x-d.source.x,dy=d.target.y-d.source.y;
      const len=Math.sqrt(dx*dx+dy*dy)||1;
      const ang=Math.atan2(dy,dx);
      const tr=d.target.r+9;
      const tx=d.target.x-Math.cos(ang)*tr, ty=d.target.y-Math.sin(ang)*tr;
      const dr=len*1.4;
      return `M${{d.source.x}},${{d.source.y}} A${{dr}},${{dr}} 0 0,1 ${{tx}},${{ty}}`;
    }});
    llabel.attr("transform",d=>`translate(${{(d.source.x+d.target.x)/2}},${{(d.source.y+d.target.y)/2}})`);
    node.attr("transform",d=>`translate(${{d.x}},${{d.y}})`);
  }});
}}

draw();
window.addEventListener("resize",draw);
</script>
</body>
</html>"""

    out_path = out_dir / f"session_{session_id}.html"
    out_path.write_text(html, encoding="utf-8")
    log.info("Graph HTML written: %s (%d nodes, %d edges)", out_path, n_nodes, n_edges)
    return out_path
