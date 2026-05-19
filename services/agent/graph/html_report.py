"""Dynamic HTML intelligence brief renderer for GraphKnows agentic sessions.

Produces the same visual design as the reference ECB brief but fully populated
from a live ``InvestigativeState`` — query, entities, relations, sources, and
the complete workflow timeline for observability.
"""

from __future__ import annotations

import datetime
import html
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from services.agent.graph.state import InvestigativeState


# ── colour constants ───────────────────────────────────────────────────────────
_TYPE_COLOR: dict[str, str] = {
    "PERSON": "#4a9eff",
    "ORGANIZATION": "#22c55e",
    "POLICY": "#f59e0b",
    "LOCATION": "#e879f9",
}
_DEFAULT_NODE_COLOR = "#94a3b8"

_EVENT_COLOR: dict[str, str] = {
    "plan_ready": "#22c55e",
    "search_start": "#4a9eff",
    "search_done": "#4a9eff",
    "docs_fetched": "#f59e0b",
    "chunks_written": "#f59e0b",
    "entities_written": "#f59e0b",
    "novelty_check": "#f59e0b",
    "retrieval_done": "#4a9eff",
    "brief_chunk": "#e879f9",
    "brief_done": "#e879f9",
    "error": "#ef4444",
}
_DEFAULT_EVENT_COLOR = "#64748b"

_SOURCE_EMOJI: dict[str, str] = {
    "web": "",
    "tavily": "",
    "eu": "️",
    "eprs": "️",
    "ep_": "️",
    "cellar": "",
}


# ── helpers ────────────────────────────────────────────────────────────────────


def _logo_text(topic: str) -> str:
    """Derive a 2–4 char logo from the topic string."""
    stop = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or"}
    words = [w for w in re.split(r"\W+", topic) if w and w.lower() not in stop]
    initials = "".join(w[0].upper() for w in words[:4])
    return initials[:4] if initials else "GK"


def _source_emoji(source: str) -> str:
    sl = source.lower()
    for key, emoji in _SOURCE_EMOJI.items():
        if key in sl:
            return emoji
    return ""


def _md_to_html(text: str) -> str:
    """Lightweight markdown → HTML conversion (headers, bold, lists, links, paras)."""
    text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"^\d+\. (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*] (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank">\1</a>', text)
    parts = text.split("\n\n")
    result: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("<h") or part.startswith("<li"):
            result.append(part)
        else:
            result.append(f"<p>{part.replace(chr(10), ' ')}</p>")
    return "\n".join(result)


def _force_layout(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    width: int = 900,
    height: int = 500,
    iterations: int = 200,
    seed: int = 99,
) -> dict[str, list[float]]:
    """Fruchterman-Reingold force-directed layout. Returns {node_id: [x, y]}."""
    if not nodes:
        return {}
    random.seed(seed)
    pos = {
        n["id"]: [random.uniform(60, width - 60), random.uniform(60, height - 60)] for n in nodes
    }
    k = math.sqrt(width * height / max(len(nodes), 1)) * 0.7
    valid_ids = {n["id"] for n in nodes}
    valid_edges = [e for e in edges if e["source"] in valid_ids and e["target"] in valid_ids]

    for _ in range(iterations):
        disp: dict[str, list[float]] = {nid: [0.0, 0.0] for nid in pos}
        nlist = list(pos.items())
        for i, (a, pa) in enumerate(nlist):
            for b, pb in nlist[i + 1 :]:
                dx, dy = pa[0] - pb[0], pa[1] - pb[1]
                dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
                f = k * k / dist
                disp[a][0] += dx / dist * f
                disp[a][1] += dy / dist * f
                disp[b][0] -= dx / dist * f
                disp[b][1] -= dy / dist * f
        for e in valid_edges:
            s, t = e["source"], e["target"]
            ps, pt = pos[s], pos[t]
            dx, dy = pt[0] - ps[0], pt[1] - ps[1]
            dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
            f = dist * dist / k
            disp[s][0] += dx / dist * f
            disp[s][1] += dy / dist * f
            disp[t][0] -= dx / dist * f
            disp[t][1] -= dy / dist * f
        for nid, dv in disp.items():
            dist = max(math.sqrt(dv[0] ** 2 + dv[1] ** 2), 0.01)
            sc = min(dist, 12.0) / dist
            pos[nid][0] = max(45, min(width - 45, pos[nid][0] + dv[0] * sc))
            pos[nid][1] = max(45, min(height - 45, pos[nid][1] + dv[1] * sc))

    return pos


def _build_svg(
    entities: list[Any],
    relations: list[Any],
    *,
    width: int = 900,
    height: int = 500,
) -> str:
    """Build a static SVG force-directed graph from entities and relations."""
    if not entities:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'style="background:#0a1628;border-radius:12px;width:100%;max-width:{width}px">'
            f'<text x="{width // 2}" y="{height // 2}" fill="#64748b" font-size="16" '
            f'text-anchor="middle" dominant-baseline="middle">No graph data available</text>'
            f"</svg>"
        )

    # normalise entity dicts vs dataclasses
    def _e_name(e: Any) -> str:
        return e.name if hasattr(e, "name") else e.get("name", "")

    def _e_label(e: Any) -> str:
        return e.label if hasattr(e, "label") else e.get("label", "Entity")

    def _r_src(r: Any) -> str:
        return r.source if hasattr(r, "source") else r.get("source", "")

    def _r_tgt(r: Any) -> str:
        return r.target if hasattr(r, "target") else r.get("target", "")

    def _r_type(r: Any) -> str:
        return r.type if hasattr(r, "type") else r.get("type", "RELATION")

    nodes = []
    node_ids: dict[str, int] = {}
    for e in entities:
        name = _e_name(e)
        nid = re.sub(r"[^A-Za-z0-9_]", "_", name)
        if nid not in node_ids:
            node_ids[nid] = len(nodes)
            nodes.append({"id": nid, "label": name, "type": _e_label(e)})

    name_to_id = {n["label"]: n["id"] for n in nodes}
    edges = []
    for r in relations:
        src_name, tgt_name = _r_src(r), _r_tgt(r)
        src_id = name_to_id.get(src_name, re.sub(r"[^A-Za-z0-9_]", "_", src_name))
        tgt_id = name_to_id.get(tgt_name, re.sub(r"[^A-Za-z0-9_]", "_", tgt_name))
        edges.append({"source": src_id, "target": tgt_id, "label": _r_type(r)})

    pos = _force_layout(nodes, edges, width=width, height=height)
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="background:#0a1628;border-radius:12px;width:100%;max-width:{width}px">',
        '<defs><marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">'
        '<polygon points="0 0, 8 3, 0 6" fill="#4a5568"/></marker></defs>',
    ]

    valid_ids = {n["id"] for n in nodes}
    for e in edges:
        if e["source"] not in valid_ids or e["target"] not in valid_ids:
            continue
        if e["source"] not in pos or e["target"] not in pos:
            continue
        x1, y1 = pos[e["source"]]
        x2, y2 = pos[e["target"]]
        dx, dy = x2 - x1, y2 - y1
        dist = max(math.sqrt(dx * dx + dy * dy), 1)
        r = 22
        ex = x2 - dx / dist * r
        ey = y2 - dy / dist * r
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        lbl = html.escape(e["label"])
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="#2d3748" stroke-width="1.5" marker-end="url(#arr)"/>'
        )
        lines.append(
            f'<text x="{mx:.1f}" y="{my:.1f}" fill="#64748b" font-size="8" '
            f'text-anchor="middle" dominant-baseline="middle">{lbl}</text>'
        )

    for n in nodes:
        if n["id"] not in pos:
            continue
        x, y = pos[n["id"]]
        color = _TYPE_COLOR.get(n["type"], _DEFAULT_NODE_COLOR)
        disp_lbl = html.escape(n["label"])
        if len(disp_lbl) > 14:
            disp_lbl = disp_lbl[:12] + "…"
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="20" fill="{color}" '
            f'fill-opacity="0.85" stroke="{color}" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="white" font-size="7.5" '
            f'font-family="sans-serif" text-anchor="middle" dominant-baseline="middle" '
            f'font-weight="bold">{disp_lbl}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _build_legend() -> str:
    parts = [
        f'<span class="legend-item"><span class="dot" style="background:{c}"></span>{t}</span>'
        for t, c in _TYPE_COLOR.items()
    ]
    return '<div class="legend">' + "".join(parts) + "</div>"


def _timeline_rows(ws_events: list[Any]) -> str:
    """Render ws_events as observability timeline rows."""
    if not ws_events:
        return '<p style="color:#64748b;font-size:13px;">No workflow events recorded.</p>'

    rows: list[str] = []
    for ev in ws_events:
        if hasattr(ev, "__getitem__"):
            ts = ev.get("ts", 0)
            etype = ev.get("event_type", "unknown")
            payload = ev.get("payload", {})
        else:
            ts = getattr(ev, "ts", 0)
            etype = getattr(ev, "event_type", "unknown")
            payload = getattr(ev, "payload", {})

        try:
            dt = datetime.datetime.fromtimestamp(float(ts))
            ts_str = dt.strftime("%H:%M:%S")
        except Exception:
            ts_str = "–"

        color = _EVENT_COLOR.get(etype, _DEFAULT_EVENT_COLOR)

        # Short payload summary
        if isinstance(payload, dict):
            summary_parts: list[str] = []
            for key in ("query", "source", "count", "doc_count", "n_entities", "n_chunks"):
                if key in payload:
                    summary_parts.append(f"{key}={html.escape(str(payload[key]))}")
            summary = " · ".join(summary_parts[:3]) if summary_parts else ""
        else:
            summary = html.escape(str(payload))[:80]

        rows.append(
            f'<div class="timeline-row">'
            f'<span class="timeline-ts">{ts_str}</span>'
            f'<span class="timeline-badge" style="background:{color}">{html.escape(etype)}</span>'
            f'<span class="timeline-summary">{summary}</span>'
            f"</div>"
        )
    return "\n".join(rows)


# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  :root {
    --bg: #0a1628; --surface: #0f2040; --surface2: #162850;
    --border: #1e3a5f; --text: #e2e8f0; --muted: #94a3b8;
    --ecb-blue: #003087; --ecb-gold: #ffb800;
    --accent: #4a9eff; --green: #22c55e; --amber: #f59e0b; --purple: #e879f9;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; font-size: 15px; line-height: 1.7; }

  .header { background: linear-gradient(135deg, #001a4d 0%, #003087 60%, #004db3 100%); border-bottom: 3px solid var(--ecb-gold); padding: 36px 48px; }
  .header-top { display: flex; align-items: center; gap: 20px; margin-bottom: 16px; }
  .logo { width: 56px; height: 56px; background: var(--ecb-gold); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 16px; color: #001a4d; flex-shrink: 0; letter-spacing: -0.5px; }
  .header-titles h1 { font-size: 22px; font-weight: 700; color: #fff; letter-spacing: 0.3px; }
  .header-titles .subtitle { font-size: 13px; color: #a0c4ff; margin-top: 3px; }
  .header-meta { display: flex; gap: 32px; margin-top: 12px; flex-wrap: wrap; }
  .meta-item { font-size: 12px; color: #a0c4ff; }
  .meta-item strong { color: var(--ecb-gold); }
  .classification { display: inline-block; background: #c0392b; color: #fff; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 3px; letter-spacing: 1px; margin-top: 12px; }
  .phase-badge { display: inline-block; background: var(--accent); color: #000; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 3px; letter-spacing: 0.5px; margin-top: 12px; margin-left: 8px; }

  .container { max-width: 1200px; margin: 0 auto; padding: 32px; }

  .report { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 36px; margin-bottom: 28px; }
  .report h2 { font-size: 20px; font-weight: 600; color: var(--ecb-gold); border-bottom: 1px solid var(--border); padding-bottom: 8px; margin: 28px 0 14px; }
  .report h2:first-child { margin-top: 0; }
  .report h3 { font-size: 16px; font-weight: 600; color: var(--accent); margin: 20px 0 10px; }
  .report p { margin-bottom: 12px; color: var(--text); }
  .report li { margin-left: 20px; margin-bottom: 8px; color: var(--text); list-style: disc; }
  .report a { color: var(--accent); text-decoration: none; }
  .report a:hover { text-decoration: underline; }
  .report strong { color: #fff; }

  .graph-section { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 28px; margin-bottom: 28px; }
  .graph-section h2 { font-size: 18px; font-weight: 600; color: var(--ecb-gold); margin-bottom: 16px; }
  .graph-wrap { overflow: hidden; border-radius: 8px; border: 1px solid var(--border); }
  .legend { display: flex; gap: 20px; margin-top: 14px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); }
  .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

  .sources-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 28px; margin-bottom: 28px; }
  .sources-panel h2 { font-size: 18px; font-weight: 600; color: var(--ecb-gold); margin-bottom: 16px; }
  .source-row { display: flex; align-items: center; justify-content: space-between; padding: 10px 16px; background: var(--surface2); border-radius: 8px; margin-bottom: 8px; border: 1px solid var(--border); }
  .source-name { font-weight: 500; color: var(--text); font-size: 14px; }
  .source-badge { background: var(--accent); color: #000; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; }
  .source-type { font-size: 12px; color: var(--muted); }

  .stats-bar { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; flex: 1; min-width: 160px; }
  .stat-card .val { font-size: 28px; font-weight: 700; color: var(--ecb-gold); }
  .stat-card .lbl { font-size: 12px; color: var(--muted); margin-top: 2px; }

  .timeline-section { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 28px; margin-bottom: 28px; }
  .timeline-section h2 { font-size: 18px; font-weight: 600; color: var(--ecb-gold); margin-bottom: 16px; }
  .timeline-row { display: flex; align-items: center; gap: 12px; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; background: var(--surface2); border: 1px solid var(--border); }
  .timeline-ts { font-size: 12px; color: var(--muted); font-family: monospace; min-width: 64px; }
  .timeline-badge { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; color: #000; white-space: nowrap; }
  .timeline-summary { font-size: 12px; color: var(--muted); }

  .footer { text-align: center; padding: 24px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--border); margin-top: 24px; }

  @media print {
    body { background: #fff; color: #000; }
    .header { background: #003087; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .report, .graph-section, .sources-panel, .timeline-section { background: #f8f9fa; border-color: #ddd; }
  }
"""


# ── main renderer ──────────────────────────────────────────────────────────────


def render_brief_html(
    state: InvestigativeState,
    *,
    analyst_name: str = "GraphKnows Intelligence Platform",
) -> str:
    """Render a full intelligence brief HTML from a live InvestigativeState.

    Args:
        state: The complete (or partial) investigative workflow state.
        analyst_name: Display name for the analyst / system credit line.

    Returns:
        A self-contained HTML string ready to write to disk or stream.
    """
    session_id: str = state["session_id"]
    topic: str = state["topic"]
    briefing: str | None = state.get("briefing")
    entities = state.get("entities") or []
    relations = state.get("relations") or []
    raw_docs = state.get("raw_docs") or []
    iteration_count: int = state.get("iteration_count", 0)
    dup_ratio: float = state.get("dup_ratio", 0.0)
    ws_events = state.get("ws_events") or []
    current_phase: str = state.get("current_macro_phase", "")
    plan = state.get("plan") or []

    today = datetime.date.today().strftime("%d %B %Y")
    logo = _logo_text(topic)

    # ── stats ──────────────────────────────────────────────────────────────────
    def _doc_source(d: Any) -> str:
        return d.source if hasattr(d, "source") else d.get("source", "unknown")

    unique_sources = len({_doc_source(d) for d in raw_docs})
    n_entities = len(entities)
    n_relations = len(relations)

    # ── report body ────────────────────────────────────────────────────────────
    if briefing:
        report_body = _md_to_html(briefing)
    else:
        report_body = (
            '<p style="color:var(--muted);font-style:italic;">Brief generation in progress…</p>'
        )

    # ── research plan section ──────────────────────────────────────────────────
    if plan:
        plan_items = "".join(
            f"<li>{html.escape(q.text if hasattr(q, 'text') else q.get('text', str(q)))}</li>"
            for q in plan
        )
        plan_html = f"<h2>Research Plan</h2><ul>{plan_items}</ul>"
    else:
        plan_html = ""

    # ── sources panel ──────────────────────────────────────────────────────────
    source_counts: Counter[str] = Counter(_doc_source(d) for d in raw_docs)
    sources_rows = ""
    if source_counts:
        for src, cnt in source_counts.most_common():
            emoji = _source_emoji(src)
            sources_rows += (
                f'<div class="source-row">'
                f"  <div>"
                f'    <div class="source-name">{emoji} {html.escape(src)}</div>'
                f"  </div>"
                f'  <span class="source-badge">{cnt} docs</span>'
                f"</div>\n"
            )
    else:
        sources_rows = '<p style="color:var(--muted);font-size:13px;">No sources recorded.</p>'

    # ── SVG graph ──────────────────────────────────────────────────────────────
    svg = _build_svg(entities, relations)
    legend_html = _build_legend()

    # ── timeline ───────────────────────────────────────────────────────────────
    timeline_html = _timeline_rows(ws_events)

    # ── phase badge ───────────────────────────────────────────────────────────
    phase_badge = (
        f'<span class="phase-badge">phase: {html.escape(current_phase)}</span>'
        if current_phase
        else ""
    )

    # ── title for header ──────────────────────────────────────────────────────
    brief_title = html.escape(topic[:80] + ("…" if len(topic) > 80 else ""))

    # ── assemble ───────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Intelligence Brief — {html.escape(topic[:60])}</title>
<style>{_CSS}</style>
</head>
<body>

<div class="header">
  <div class="header-top">
    <div class="logo">{html.escape(logo)}</div>
    <div class="header-titles">
      <h1>Intelligence Brief: {brief_title}</h1>
      <div class="subtitle">Prepared for the EU Economic and Monetary Affairs Committee (ECON) · European Parliament</div>
    </div>
  </div>
  <div class="header-meta">
    <div class="meta-item"><strong>Date:</strong> {today}</div>
    <div class="meta-item"><strong>Session ID:</strong> {html.escape(session_id)}</div>
    <div class="meta-item"><strong>Analyst:</strong> {html.escape(analyst_name)}</div>
    <div class="meta-item"><strong>Iterations:</strong> {iteration_count}</div>
  </div>
  <div>
    <span class="classification">ECON COMMITTEE — RESTRICTED CIRCULATION</span>
    {phase_badge}
  </div>
</div>

<div class="container">

  <!-- Stats bar -->
  <div class="stats-bar">
    <div class="stat-card"><div class="val">{unique_sources}</div><div class="lbl">Source Archives Queried</div></div>
    <div class="stat-card"><div class="val">{n_entities}</div><div class="lbl">Knowledge Graph Entities</div></div>
    <div class="stat-card"><div class="val">{n_relations}</div><div class="lbl">Entity Relations Mapped</div></div>
    <div class="stat-card"><div class="val">{iteration_count}</div><div class="lbl">Search Iterations</div></div>
  </div>

  <!-- Main Report -->
  <div class="report">
    {plan_html}
    {report_body}
  </div>

  <!-- Knowledge Graph -->
  <div class="graph-section">
    <h2> Knowledge Graph — {html.escape(topic[:50])} Entity Network</h2>
    <p style="color:var(--muted);font-size:13px;margin-bottom:14px;">Static force-directed layout (Fruchterman-Reingold, 200 iterations). {n_entities} nodes · {n_relations} edges.</p>
    <div class="graph-wrap">
      {svg}
    </div>
    {legend_html}
  </div>

  <!-- Workflow Timeline (observability) -->
  <div class="timeline-section">
    <h2>️ Workflow Timeline</h2>
    {timeline_html}
  </div>

  <!-- Sources Panel -->
  <div class="sources-panel">
    <h2> Archive Sources Queried</h2>
    {sources_rows}
  </div>

</div>

<div class="footer">
  <p>Intelligence Brief · Generated by GraphKnows Intelligence Platform · Session {html.escape(session_id)} · {today}</p>
  <p style="margin-top:6px;">Prepared for the EU Economic and Monetary Affairs Committee (ECON) · European Parliament · Strasbourg/Brussels</p>
  <p style="margin-top:6px;color:#4a5568;">FOR COMMITTEE USE ONLY — NOT FOR PUBLIC DISTRIBUTION</p>
</div>

</body>
</html>"""


# ── standalone export helper ───────────────────────────────────────────────────


def export_brief_html(
    state: InvestigativeState,
    output_dir: str | Path = "/root/graphknows/output",
) -> Path:
    """Export the intelligence brief HTML from a complete InvestigativeState.

    Args:
        state: The workflow state to render.
        output_dir: Directory to write the HTML file into.

    Returns:
        Path to the written HTML file.
    """
    html_str = render_brief_html(state)
    out_path = Path(output_dir) / f"brief_{state['session_id']}.html"
    out_path.write_text(html_str, encoding="utf-8")
    return out_path


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    from services.agent.graph.state import Entity, RawDoc, Relation, SubQuestion, WSEvent

    dummy_state: InvestigativeState = {
        "session_id": "test-smoke-001",
        "topic": "Russian Disinformation Campaigns Targeting the 2024 European Parliament Elections",
        "plan": [
            SubQuestion(id="q1", text="Who are the principal actor networks behind the campaigns?"),
            SubQuestion(id="q2", text="What platforms and technical vectors were used?"),
            SubQuestion(id="q3", text="What was the measurable impact on electoral discourse?"),
        ],
        "queries": ["Russian disinfo EP elections 2024"],
        "raw_docs": [
            RawDoc(
                id="d1",
                title="EU DisinfoLab Report",
                text="...",
                url="https://example.com",
                source="eprs_briefings",
                date="2024-03-01",
            ),
            RawDoc(
                id="d2",
                title="Tavily search result",
                text="...",
                url="https://example.com/2",
                source="tavily",
                date=None,
            ),
        ],
        "chunks": [],
        "entities": [
            Entity(id="e1", name="Kremlin", label="ORGANIZATION", confidence=0.95),
            Entity(id="e2", name="RT (Russia Today)", label="ORGANIZATION", confidence=0.9),
            Entity(id="e3", name="Vladimir Putin", label="PERSON", confidence=0.92),
            Entity(id="e4", name="European Parliament", label="ORGANIZATION", confidence=0.98),
            Entity(id="e5", name="DSA", label="POLICY", confidence=0.85),
            Entity(id="e6", name="Russia", label="LOCATION", confidence=0.97),
        ],
        "relations": [
            Relation(source="Kremlin", target="RT (Russia Today)", type="CONTROLS", confidence=0.9),
            Relation(source="Vladimir Putin", target="Kremlin", type="LEADS", confidence=0.95),
            Relation(
                source="RT (Russia Today)",
                target="European Parliament",
                type="TARGETS",
                confidence=0.8,
            ),
            Relation(source="DSA", target="RT (Russia Today)", type="REGULATES", confidence=0.85),
        ],
        "iteration_count": 3,
        "dup_ratio": 0.12,
        "retrieval_hits": [],
        "briefing": "## Executive Summary\n\nAvailable evidence indicates that **Russian state-aligned** actors conducted coordinated influence operations targeting the 2024 EP elections.\n\n## 1. Background\n\nThe campaigns exploited social media fragmentation and low digital literacy in key target member states.",
        "ws_events": [
            WSEvent(
                session_id="test-smoke-001",
                event_type="plan_ready",
                payload={"n_subquestions": 3},
                ts=time.time() - 120,
            ),
            WSEvent(
                session_id="test-smoke-001",
                event_type="search_start",
                payload={"query": "Russian disinfo EP elections 2024", "source": "eprs_briefings"},
                ts=time.time() - 110,
            ),
            WSEvent(
                session_id="test-smoke-001",
                event_type="docs_fetched",
                payload={"count": 2},
                ts=time.time() - 100,
            ),
            WSEvent(
                session_id="test-smoke-001",
                event_type="entities_written",
                payload={"n_entities": 6},
                ts=time.time() - 90,
            ),
            WSEvent(
                session_id="test-smoke-001",
                event_type="brief_done",
                payload={},
                ts=time.time() - 10,
            ),
        ],
        "node_features": {},
        "current_macro_phase": "communicate",
    }

    html_str = render_brief_html(dummy_state)
    out = Path("/root/graphknows/output/brief_test-smoke-001.html")
    out.write_text(html_str, encoding="utf-8")
    print(f"OK — {len(html_str):,} bytes → {out}")
