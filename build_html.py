#!/usr/bin/env python3
"""Build a rich investigation-brief HTML from investigation_trace.json."""

import html
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

TRACE_PATH = Path('/root/graphknows/output/investigation_trace.json')
OUTPUT_PATH = Path('/root/graphknows/output/ecb_cbdc_brief_v3.html')

# ── Load trace ─────────────────────────────────────────────────────────────────
if TRACE_PATH.exists():
    with open(TRACE_PATH) as fh:
        trace = json.load(fh)
else:
    answer_path = Path('/root/graphknows/output/agent_answer.txt')
    answer_raw = answer_path.read_text() if answer_path.exists() else ''
    trace = {
        'query': 'ECB digital euro CBDC investigation',
        'session_id': 'ecb-cbdc-001',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'finished_at': datetime.now(timezone.utc).isoformat(),
        'total_seconds': 0,
        'tool_calls': [],
        'workflow_nodes_visited': [],
        'graph': {'nodes': [], 'edges': []},
        'lexical': {'nodes': [], 'edges': []},
        'answer': answer_raw,
        'evidence_added': [],
        'phase_events': [],
    }

query           = trace.get('query', '')
session_id      = trace.get('session_id', '')
started_at      = trace.get('started_at', '')
total_seconds   = trace.get('total_seconds', 0)
tool_calls      = trace.get('tool_calls', [])
workflow_nodes  = trace.get('workflow_nodes_visited', [])
graph_data      = trace.get('graph', {'nodes': [], 'edges': []})
lexical_data    = trace.get('lexical', {'nodes': [], 'edges': []})
answer          = trace.get('answer', '')
evidence_events = trace.get('evidence_added', [])

# Use lexical graph if main graph is empty
if not graph_data['nodes'] and lexical_data['nodes']:
    graph_data = lexical_data

# ── Markdown → HTML ────────────────────────────────────────────────────────────
def md_to_html(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', t, flags=re.MULTILINE)
    t = re.sub(r'^### (.+)$',  r'<h3>\1</h3>', t, flags=re.MULTILINE)
    t = re.sub(r'^## (.+)$',   r'<h2>\1</h2>', t, flags=re.MULTILINE)
    t = re.sub(r'^# (.+)$',    r'<h1>\1</h1>', t, flags=re.MULTILINE)
    t = re.sub(r'^---+$',      r'<hr>', t, flags=re.MULTILINE)
    t = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', t)
    t = re.sub(r'\*\*(.+?)\*\*',     r'<strong>\1</strong>', t)
    t = re.sub(r'\*(.+?)\*',         r'<em>\1</em>', t)
    t = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener">{m.group(1)}</a>',
        t,
    )
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)

    lines = t.split('\n')
    out: list[str] = []
    in_ul = in_ol = False
    for line in lines:
        ul_m = re.match(r'^[-*] (.+)$', line)
        ol_m = re.match(r'^\d+\. (.+)$', line)
        if ul_m:
            if in_ol: out.append('</ol>'); in_ol = False
            if not in_ul: out.append('<ul>'); in_ul = True
            out.append(f'<li>{ul_m.group(1)}</li>')
        elif ol_m:
            if in_ul: out.append('</ul>'); in_ul = False
            if not in_ol: out.append('<ol>'); in_ol = True
            out.append(f'<li>{ol_m.group(1)}</li>')
        else:
            if in_ul: out.append('</ul>'); in_ul = False
            if in_ol: out.append('</ol>'); in_ol = False
            out.append(line)
    if in_ul: out.append('</ul>')
    if in_ol: out.append('</ol>')
    t = '\n'.join(out)

    blocks = re.split(r'\n{2,}', t)
    result = []
    for block in blocks:
        block = block.strip()
        if not block: continue
        if re.match(r'^<(h[1-6]|ul|ol|hr)', block):
            result.append(block)
        else:
            result.append(f'<p>{block}</p>')
    return '\n'.join(result)

# ── Tool styling ───────────────────────────────────────────────────────────────
TOOL_ICONS  = {'search_web':'','search_eu':'️','graphgen':'️','graphrag':'','search_archives':''}
TOOL_COLORS = {'search_web':'#3b82f6','search_eu':'#8b5cf6','graphgen':'#f59e0b','graphrag':'#10b981','search_archives':'#ef4444'}

def tool_badge(name: str) -> str:
    icon  = TOOL_ICONS.get(name, '️')
    color = TOOL_COLORS.get(name, '#64748b')
    return f'<span class="tool-badge" style="background:{color}">{icon} {html.escape(name)}</span>'

def fmt_ms(ms: int) -> str:
    return f'{ms}ms' if ms < 1000 else f'{ms/1000:.1f}s'

def fmt_inputs(inp) -> str:
    if isinstance(inp, str):
        return f'<span class="param-val">{html.escape(inp[:300])}</span>'
    parts = []
    for k, v in (inp or {}).items():
        parts.append(f'<span class="param-key">{html.escape(str(k))}</span>: '
                     f'<span class="param-val">{html.escape(str(v)[:200])}</span>')
    return '<br>'.join(parts) or '—'

# ── D3 graph data ──────────────────────────────────────────────────────────────
d3_nodes: list[dict] = []
seen_ids: set[str] = set()
for n in graph_data['nodes'][:120]:
    nid = str(n.get('id') or n.get('name') or '')
    if not nid or nid in seen_ids: continue
    seen_ids.add(nid)
    d3_nodes.append({'id': nid,
                     'label': str(n.get('name') or n.get('label') or nid)[:50],
                     'type':  str(n.get('type')  or n.get('label') or 'CONCEPT')})

valid_ids = {n['id'] for n in d3_nodes}
d3_edges: list[dict] = []
seen_edges: set[tuple] = set()
for e in graph_data['edges'][:300]:
    src = str(e.get('source') or '')
    tgt = str(e.get('target') or '')
    rel = str(e.get('type')   or e.get('label') or 'RELATED')
    if src not in valid_ids or tgt not in valid_ids: continue
    key = (src, tgt, rel)
    if key in seen_edges: continue
    seen_edges.add(key)
    d3_edges.append({'source': src, 'target': tgt, 'label': rel})

nodes_json = json.dumps(d3_nodes)
edges_json = json.dumps(d3_edges)

# ── Workflow SVG ───────────────────────────────────────────────────────────────
def build_workflow_svg(tcs: list[dict]) -> str:
    steps = ['START'] + [tc['tool'] for tc in tcs] + ['ANSWER']
    box_w, box_h = 130, 40
    gap = 36
    total_w = len(steps) * (box_w + gap) + gap
    svg_h = 110
    parts = [f'<svg viewBox="0 0 {total_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" '
             f'style="width:100%;max-height:110px;">']
    parts.append('<defs><marker id="arr" markerWidth="8" markerHeight="8" '
                 'refX="6" refY="3" orient="auto">'
                 '<path d="M0,0 L0,6 L8,3 z" fill="#64748b"/></marker></defs>')
    cy = svg_h // 2
    cx_prev = None
    for i, step in enumerate(steps):
        x  = gap + i * (box_w + gap)
        cx = x + box_w // 2
        y  = cy - box_h // 2
        color = ('#1e293b' if step == 'START'
                 else '#15803d' if step == 'ANSWER'
                 else TOOL_COLORS.get(step, '#7c3aed'))
        icon  = TOOL_ICONS.get(step, '')
        label = f'{icon} {step}' if icon else step
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" fill="{color}" opacity="0.9"/>')
        parts.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-family="monospace" '
                     f'font-size="11" fill="#fff">{html.escape(label)}</text>')
        if cx_prev is not None:
            x1 = cx_prev + box_w // 2
            parts.append(f'<line x1="{x1}" y1="{cy}" x2="{x}" y2="{cy}" '
                         f'stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)"/>')
        cx_prev = x
    parts.append('</svg>')
    return ''.join(parts)

workflow_svg = build_workflow_svg(tool_calls)

# ── Stats ──────────────────────────────────────────────────────────────────────
total_docs = sum(tc.get('doc_count', 0) for tc in tool_calls)

# ── Build HTML ─────────────────────────────────────────────────────────────────
date_str = ''
try:
    dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
    date_str = dt.strftime('%d %B %Y, %H:%M UTC')
except Exception:
    date_str = started_at[:19] if started_at else ''

# Trace rows
trace_rows = []
for tc in tool_calls:
    step = tc.get('step', '?')
    name = tc.get('tool', '')
    dur  = tc.get('duration_ms', 0)
    inp  = tc.get('inputs', {})
    out  = str(tc.get('output', ''))
    doc_count = tc.get('doc_count', 0)
    doc_badge = f'<span class="doc-badge">{doc_count} docs</span>' if doc_count else ''
    trace_rows.append(f'''
      <div class="trace-step">
        <div class="trace-header" onclick="this.closest(\'.trace-step\').classList.toggle(\'open\')">
          <span class="step-num">Step {step}</span>
          {tool_badge(name)}
          {doc_badge}
          <span class="step-dur">{fmt_ms(dur)}</span>
          <span class="chevron">▼</span>
        </div>
        <div class="trace-body">
          <div class="trace-section">
            <div class="trace-label">Inputs</div>
            <div class="trace-content">{fmt_inputs(inp)}</div>
          </div>
          <div class="trace-section">
            <div class="trace-label">Output (first 800 chars)</div>
            <div class="trace-content output-pre">{html.escape(out[:800])}</div>
          </div>
        </div>
      </div>''')

trace_html   = '\n'.join(trace_rows) if trace_rows else '<p class="muted">No tool calls captured — re-run the investigation.</p>'
report_html  = md_to_html(answer)   if answer     else '<p class="muted">No report generated yet.</p>'
graph_section = f'''
  <div id="graph-container">
    <div id="graph-controls">
      <button onclick="resetZoom()">Reset zoom</button>
      <button onclick="toggleLabels()">Toggle labels</button>
      <span id="graph-stats">{len(d3_nodes)} nodes &middot; {len(d3_edges)} edges</span>
    </div>
    <svg id="graph-svg"></svg>
    <div id="node-tooltip"></div>
  </div>''' if d3_nodes else '<p class="muted">No graph data captured.</p>'

HTML = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GraphKnows — ECB CBDC Intelligence Brief</title>
<style>
:root{{
  --bg:#0f1117;--bg2:#161b22;--bg3:#1c2333;--border:#30363d;
  --text:#e6edf3;--text-muted:#7d8590;
  --blue:#388bfd;--blue-dark:#1f4e8c;--purple:#7c3aed;--accent:#58a6ff;
  --green:#3fb950;--amber:#d29922;--red:#f85149;--card:#1c2333;
  --radius:10px;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;line-height:1.65}}
.container{{max-width:1200px;margin:0 auto;padding:0 24px 80px}}
/* Hero */
.hero{{background:linear-gradient(135deg,var(--bg2) 0%,var(--bg) 60%,#160d2e 100%);border-bottom:1px solid var(--border);padding:40px 24px 32px;margin-bottom:32px}}
.hero-inner{{max-width:1200px;margin:0 auto}}
.hero-logo{{font-size:11px;letter-spacing:2.5px;text-transform:uppercase;color:var(--accent);margin-bottom:10px;font-weight:600}}
.hero-query{{font-size:18px;font-weight:600;color:var(--text);margin-bottom:14px;line-height:1.45;max-width:900px}}
.hero-meta{{display:flex;gap:24px;flex-wrap:wrap}}
.meta-chip{{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--text-muted)}}
.meta-chip strong{{color:var(--text)}}
.status-ok{{color:var(--green);font-weight:600}}
/* Stats */
.stats-bar{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:40px}}
.stat-card{{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:20px;text-align:center}}
.stat-num{{font-size:28px;font-weight:700;color:var(--accent)}}
.stat-label{{font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;margin-top:4px}}
/* Sections */
.section{{margin-bottom:48px}}
.section-title{{font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--text-muted);border-bottom:1px solid var(--border);padding-bottom:10px;margin-bottom:20px}}
/* Workflow */
.workflow-wrap{{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:20px;overflow-x:auto}}
/* Trace */
.trace-step{{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:8px;overflow:hidden}}
.trace-header{{display:flex;align-items:center;gap:10px;padding:12px 16px;cursor:pointer;user-select:none}}
.trace-header:hover{{background:var(--bg3)}}
.step-num{{font-size:12px;font-weight:600;color:var(--text-muted);min-width:50px}}
.step-dur{{font-size:12px;color:var(--text-muted);margin-left:auto}}
.doc-badge{{background:rgba(63,185,80,.1);color:var(--green);border:1px solid rgba(63,185,80,.25);border-radius:20px;padding:2px 10px;font-size:11px}}
.chevron{{color:var(--text-muted);transition:transform .2s;font-size:12px}}
.trace-step.open .chevron{{transform:rotate(180deg)}}
.trace-body{{display:none;border-top:1px solid var(--border);padding:16px;background:var(--bg3)}}
.trace-step.open .trace-body{{display:block}}
.trace-section{{margin-bottom:12px}}
.trace-label{{font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--text-muted);margin-bottom:5px}}
.trace-content{{font-size:13px;color:var(--text)}}
.output-pre{{font-family:'SFMono-Regular',Consolas,'Liberation Mono',monospace;font-size:12px;white-space:pre-wrap;word-break:break-all;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px}}
.param-key{{color:var(--purple);font-weight:600}}
.param-val{{color:var(--text)}}
/* Report */
.report-body{{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:32px 36px;max-width:860px}}
.report-body h1,.report-body h2{{color:var(--accent);margin:24px 0 10px;font-size:17px;font-weight:600}}
.report-body h3{{color:var(--blue);margin:16px 0 8px;font-size:15px}}
.report-body h4{{color:var(--amber);margin:12px 0 6px;font-size:14px}}
.report-body p{{margin:8px 0;color:var(--text)}}
.report-body ul,.report-body ol{{margin:8px 0 8px 24px}}
.report-body li{{margin:4px 0}}
.report-body a{{color:var(--accent);text-decoration:underline}}
.report-body a:hover{{color:var(--blue)}}
.report-body strong{{color:var(--text)}}
.report-body code{{background:rgba(56,139,253,.1);border:1px solid rgba(56,139,253,.2);border-radius:4px;padding:1px 5px;font-size:12px;font-family:monospace}}
.report-body hr{{border:none;border-top:1px solid var(--border);margin:20px 0}}
/* Graph */
#graph-container{{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;position:relative}}
#graph-controls{{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid var(--border);background:var(--bg3)}}
#graph-controls button{{background:var(--bg2);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:5px 12px;font-size:12px;cursor:pointer;transition:border-color .15s}}
#graph-controls button:hover{{border-color:var(--accent);color:var(--accent)}}
#graph-stats{{margin-left:auto;font-size:12px;color:var(--text-muted)}}
#graph-svg{{width:100%;height:520px;display:block}}
#node-tooltip{{position:absolute;background:rgba(15,17,23,.97);border:1px solid var(--border);border-radius:8px;padding:8px 12px;font-size:12px;color:var(--text);pointer-events:none;display:none;max-width:260px;z-index:10}}
/* Badge */
.tool-badge{{display:inline-flex;align-items:center;gap:4px;border-radius:20px;padding:3px 10px;font-size:12px;font-weight:600;color:#fff}}
.muted{{color:var(--text-muted);font-style:italic}}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-inner">
    <div class="hero-logo">GraphKnows &middot; Intelligence Brief</div>
    <div class="hero-query">{html.escape(query)}</div>
    <div class="hero-meta">
      <div class="meta-chip"> <strong>{date_str}</strong></div>
      <div class="meta-chip"> Session <strong>{html.escape(session_id)}</strong></div>
      <div class="meta-chip">️ <strong>{total_seconds}s</strong> total</div>
      <div class="meta-chip status-ok"> Complete</div>
    </div>
  </div>
</div>

<div class="container">

  <div class="stats-bar">
    <div class="stat-card"><div class="stat-num">{len(tool_calls)}</div><div class="stat-label">Tool Calls</div></div>
    <div class="stat-card"><div class="stat-num">{total_docs}</div><div class="stat-label">Docs Retrieved</div></div>
    <div class="stat-card"><div class="stat-num">{len(d3_nodes)}</div><div class="stat-label">Graph Nodes</div></div>
    <div class="stat-card"><div class="stat-num">{len(d3_edges)}</div><div class="stat-label">Graph Edges</div></div>
    <div class="stat-card"><div class="stat-num">{total_seconds}s</div><div class="stat-label">Runtime</div></div>
  </div>

  <div class="section">
    <div class="section-title">Agent Workflow</div>
    <div class="workflow-wrap">{workflow_svg}</div>
  </div>

  <div class="section">
    <div class="section-title">Investigation Trace</div>
    {trace_html}
  </div>

  <div class="section">
    <div class="section-title">Full Intelligence Report</div>
    <div class="report-body">{report_html}</div>
  </div>

  <div class="section">
    <div class="section-title">Knowledge Graph</div>
    {graph_section}
  </div>

</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function(){{
  const nodes={nodes_json};
  const links={edges_json};
  if(!nodes.length)return;
  const TYPE_COLORS={{'PERSON':'#ec4899','ORGANIZATION':'#58a6ff','POLICY':'#3fb950',
    'LOCATION':'#d29922','CONCEPT':'#7c3aed','Document':'#7d8590'}};
  const nodeColor=d=>TYPE_COLORS[d.type]||'#a78bfa';
  const nodeR=d=>d.type==='Document'?7:9;
  const svg=d3.select('#graph-svg');
  const W=document.getElementById('graph-svg').clientWidth||1100,H=520;
  const g=svg.append('g');
  const zoom=d3.zoom().scaleExtent([0.15,10]).on('zoom',e=>g.attr('transform',e.transform));
  svg.call(zoom);
  svg.append('defs').append('marker').attr('id','arr')
    .attr('viewBox','0 -4 8 8').attr('refX',14).attr('markerWidth',6)
    .attr('markerHeight',6).attr('orient','auto')
    .append('path').attr('d','M0,-4L8,0L0,4').attr('fill','#475569');
  const sim=d3.forceSimulation(nodes)
    .force('link',d3.forceLink(links).id(d=>d.id).distance(85).strength(0.4))
    .force('charge',d3.forceManyBody().strength(-220))
    .force('center',d3.forceCenter(W/2,H/2))
    .force('collision',d3.forceCollide(18));
  const link=g.append('g').selectAll('line').data(links).join('line')
    .attr('stroke','#334155').attr('stroke-width',1).attr('marker-end','url(#arr)');
  const node=g.append('g').selectAll('g').data(nodes).join('g')
    .call(d3.drag()
      .on('start',(ev,d)=>{{if(!ev.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y}})
      .on('drag', (ev,d)=>{{d.fx=ev.x;d.fy=ev.y}})
      .on('end',  (ev,d)=>{{if(!ev.active)sim.alphaTarget(0);d.fx=null;d.fy=null}}));
  node.append('circle').attr('r',nodeR).attr('fill',nodeColor)
    .attr('stroke','#0f172a').attr('stroke-width',1.5);
  const labels=node.append('text').attr('dy',d=>nodeR(d)+12)
    .attr('text-anchor','middle').attr('font-size','10px').attr('fill','#cbd5e1')
    .attr('pointer-events','none')
    .text(d=>d.label.length>18?d.label.slice(0,16)+'\u2026':d.label);
  const tooltip=document.getElementById('node-tooltip');
  node.on('mouseenter',(ev,d)=>{{
      tooltip.style.display='block';
      tooltip.innerHTML=`<strong>${{d.label}}</strong><br><em style="color:#94a3b8">${{d.type}}</em>`;
    }})
    .on('mousemove',ev=>{{
      const cr=document.getElementById('graph-container').getBoundingClientRect();
      tooltip.style.left=(ev.clientX-cr.left+12)+'px';
      tooltip.style.top=(ev.clientY-cr.top+12)+'px';
    }})
    .on('mouseleave',()=>{{tooltip.style.display='none'}});
  node.on('click',(ev,d)=>{{
    ev.stopPropagation();
    const connected=new Set([d.id]);
    links.forEach(l=>{{
      if(l.source.id===d.id)connected.add(l.target.id);
      if(l.target.id===d.id)connected.add(l.source.id);
    }});
    node.select('circle').attr('opacity',n=>connected.has(n.id)?1:0.15);
    link.attr('opacity',l=>connected.has(l.source.id)&&connected.has(l.target.id)?1:0.05);
  }});
  svg.on('click',()=>{{node.select('circle').attr('opacity',1);link.attr('opacity',1)}});
  sim.on('tick',()=>{{
    link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
        .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    node.attr('transform',d=>`translate(${{d.x}},${{d.y}})`);
  }});
  window.resetZoom=()=>svg.transition().duration(400).call(zoom.transform,d3.zoomIdentity);
  let labelsOn=true;
  window.toggleLabels=()=>{{labelsOn=!labelsOn;labels.attr('display',labelsOn?'block':'none')}};
}})();
</script>

</body>
</html>'''

OUTPUT_PATH.write_text(HTML, encoding='utf-8')
size_kb = OUTPUT_PATH.stat().st_size // 1024
print(f'Saved {OUTPUT_PATH}  ({size_kb} KB)')
print(f'Nodes: {len(d3_nodes)}, Edges: {len(d3_edges)}, Tool calls: {len(tool_calls)}')
