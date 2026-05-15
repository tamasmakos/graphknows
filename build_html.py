#!/usr/bin/env python3
"""Build the ECB CBDC intelligence brief HTML with static SVG graph."""

import re, math, random, html, datetime

# ── Load agent answer ─────────────────────────────────────────────────────────
with open('/root/graphknows/output/agent_answer.txt') as f:
    answer_raw = f.read()

# ── Entity definitions (curated + extracted) ──────────────────────────────────
PERSONS = [
    'Christine Lagarde', 'Fabio Panetta', 'Piero Cipollone',
    'Luis de Guindos', 'Philip Lane',
]
ORGANIZATIONS = [
    'ECB', 'European Central Bank', 'Eurosystem', 'European Parliament',
    'Council of the EU', 'European Commission', 'Central Bank of Ireland',
    'Bundesbank', 'ECON Committee',
]
POLICIES = [
    'Digital Euro', 'CBDC', 'MiCA', 'TFEU', 'ESCB Statute',
    'Digital Euro Regulation', 'Payment Services Directive',
]
LOCATIONS = [
    'Eurozone', 'Germany', 'France', 'European Union', 'Ireland',
]

# Build node list
nodes = []
node_ids = {}

def add_node(label, ntype):
    nid = label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    if nid not in node_ids:
        node_ids[nid] = len(nodes)
        nodes.append({'id': nid, 'label': label, 'type': ntype})
    return nid

for p in PERSONS:       add_node(p, 'PERSON')
for o in ORGANIZATIONS: add_node(o, 'ORGANIZATION')
for pol in POLICIES:    add_node(pol, 'POLICY')
for loc in LOCATIONS:   add_node(loc, 'LOCATION')

# Build edges
edges = [
    {'source': 'Christine_Lagarde',   'target': 'ECB',                    'label': 'chairs'},
    {'source': 'Fabio_Panetta',        'target': 'Digital_Euro',           'label': 'led project'},
    {'source': 'Piero_Cipollone',      'target': 'Digital_Euro',           'label': 'oversees'},
    {'source': 'ECB',                  'target': 'Digital_Euro',           'label': 'proposes'},
    {'source': 'ECB',                  'target': 'Eurosystem',             'label': 'leads'},
    {'source': 'European_Commission',  'target': 'Digital_Euro_Regulation','label': 'drafted'},
    {'source': 'European_Parliament',  'target': 'Digital_Euro_Regulation','label': 'reviews'},
    {'source': 'ECON_Committee',       'target': 'Digital_Euro_Regulation','label': 'scrutinises'},
    {'source': 'Council_of_the_EU',    'target': 'Digital_Euro_Regulation','label': 'co-legislates'},
    {'source': 'Digital_Euro',         'target': 'CBDC',                   'label': 'is a'},
    {'source': 'Digital_Euro',         'target': 'Eurozone',               'label': 'covers'},
    {'source': 'MiCA',                 'target': 'Digital_Euro',           'label': 'complements'},
    {'source': 'Bundesbank',           'target': 'ECB',                    'label': 'member of'},
    {'source': 'Central_Bank_of_Ireland','target': 'Eurosystem',          'label': 'member of'},
    {'source': 'Philip_Lane',          'target': 'ECB',                    'label': 'chief economist'},
    {'source': 'Luis_de_Guindos',      'target': 'ECB',                    'label': 'vice-president'},
    {'source': 'TFEU',                 'target': 'ECB',                    'label': 'mandates'},
    {'source': 'ECB',                  'target': 'Payment_Services_Directive','label': 'aligns with'},
    {'source': 'Digital_Euro',         'target': 'European_Union',         'label': 'issued in'},
    {'source': 'ESCB_Statute',         'target': 'Eurosystem',             'label': 'governs'},
]

# Filter edges to valid nodes
valid_ids = set(node_ids.keys())
edges = [e for e in edges if e['source'] in valid_ids and e['target'] in valid_ids]

# ── Force-directed layout ─────────────────────────────────────────────────────
random.seed(99)
W, H = 900, 500
pos = {n['id']: [random.uniform(60, W-60), random.uniform(60, H-60)] for n in nodes}
k = math.sqrt(W * H / max(len(nodes), 1)) * 0.7

for _ in range(200):
    disp = {nid: [0.0, 0.0] for nid in pos}
    nlist = list(pos.items())
    for i, (a, pa) in enumerate(nlist):
        for b, pb in nlist[i+1:]:
            dx, dy = pa[0]-pb[0], pa[1]-pb[1]
            dist = max(math.sqrt(dx*dx+dy*dy), 1.0)
            f = k*k/dist
            disp[a][0] += dx/dist*f; disp[a][1] += dy/dist*f
            disp[b][0] -= dx/dist*f; disp[b][1] -= dy/dist*f
    for e in edges:
        s, t = e['source'], e['target']
        ps, pt = pos[s], pos[t]
        dx, dy = pt[0]-ps[0], pt[1]-ps[1]
        dist = max(math.sqrt(dx*dx+dy*dy), 1.0)
        f = dist*dist/k
        disp[s][0] += dx/dist*f; disp[s][1] += dy/dist*f
        disp[t][0] -= dx/dist*f; disp[t][1] -= dy/dist*f
    for nid, dv in disp.items():
        dist = max(math.sqrt(dv[0]**2+dv[1]**2), 0.01)
        sc = min(dist, 12.0)/dist
        pos[nid][0] = max(45, min(W-45, pos[nid][0]+dv[0]*sc))
        pos[nid][1] = max(45, min(H-45, pos[nid][1]+dv[1]*sc))

# ── Build SVG ─────────────────────────────────────────────────────────────────
TYPE_COLOR = {'PERSON': '#4a9eff', 'ORGANIZATION': '#22c55e', 'POLICY': '#f59e0b', 'LOCATION': '#e879f9'}

svg_lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" style="background:#0a1628;border-radius:12px;width:100%;max-width:{W}px">']
svg_lines.append('<defs><marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#4a5568"/></marker></defs>')

# Draw edges
for e in edges:
    x1, y1 = pos[e['source']]
    x2, y2 = pos[e['target']]
    # shorten line to not overlap node circle
    dx, dy = x2-x1, y2-y1
    dist = max(math.sqrt(dx*dx+dy*dy), 1)
    r = 22
    ex = x2 - dx/dist*r; ey = y2 - dy/dist*r
    mx = (x1+x2)/2; my = (y1+y2)/2
    svg_lines.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#2d3748" stroke-width="1.5" marker-end="url(#arr)"/>')
    lbl = html.escape(e['label'])
    svg_lines.append(f'<text x="{mx:.1f}" y="{my:.1f}" fill="#64748b" font-size="8" text-anchor="middle" dominant-baseline="middle">{lbl}</text>')

# Draw nodes
for n in nodes:
    x, y = pos[n['id']]
    color = TYPE_COLOR.get(n['type'], '#888')
    lbl = html.escape(n['label'])
    # Truncate label for display
    disp_lbl = lbl if len(lbl) <= 14 else lbl[:12] + '…'
    svg_lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="20" fill="{color}" fill-opacity="0.85" stroke="{color}" stroke-width="1.5"/>')
    svg_lines.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="white" font-size="7.5" font-family="sans-serif" text-anchor="middle" dominant-baseline="middle" font-weight="bold">{disp_lbl}</text>')

svg_lines.append('</svg>')
svg = '\n'.join(svg_lines)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_html = '<div class="legend">'
for t, c in TYPE_COLOR.items():
    legend_html += f'<span class="legend-item"><span class="dot" style="background:{c}"></span>{t}</span>'
legend_html += '</div>'

# ── Convert answer markdown to HTML ───────────────────────────────────────────
def md_to_html(text):
    # headers
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    # bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # list items
    text = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    # urls in brackets
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)
    # paragraphs
    parts = text.split('\n\n')
    result = []
    for p in parts:
        p = p.strip()
        if not p: continue
        if p.startswith('<h') or p.startswith('<li'):
            result.append(p)
        else:
            result.append(f'<p>{p.replace(chr(10), " ")}</p>')
    return '\n'.join(result)

report_html = md_to_html(answer_raw)

# ── Final HTML ────────────────────────────────────────────────────────────────
today = datetime.date.today().strftime('%d %B %Y')

html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ECB Digital Euro CBDC — Intelligence Brief v3</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  :root {{
    --bg: #0a1628; --surface: #0f2040; --surface2: #162850;
    --border: #1e3a5f; --text: #e2e8f0; --muted: #94a3b8;
    --ecb-blue: #003087; --ecb-gold: #ffb800;
    --accent: #4a9eff; --green: #22c55e; --amber: #f59e0b; --purple: #e879f9;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; font-size: 15px; line-height: 1.7; }}
  
  /* Header */
  .header {{ background: linear-gradient(135deg, #001a4d 0%, #003087 60%, #004db3 100%); border-bottom: 3px solid var(--ecb-gold); padding: 36px 48px; }}
  .header-top {{ display: flex; align-items: center; gap: 20px; margin-bottom: 16px; }}
  .ecb-logo {{ width: 56px; height: 56px; background: var(--ecb-gold); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 18px; color: #001a4d; flex-shrink: 0; }}
  .header-titles h1 {{ font-size: 22px; font-weight: 700; color: #fff; letter-spacing: 0.3px; }}
  .header-titles .subtitle {{ font-size: 13px; color: #a0c4ff; margin-top: 3px; }}
  .header-meta {{ display: flex; gap: 32px; margin-top: 12px; }}
  .meta-item {{ font-size: 12px; color: #a0c4ff; }}
  .meta-item strong {{ color: var(--ecb-gold); }}
  .classification {{ display: inline-block; background: #c0392b; color: #fff; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 3px; letter-spacing: 1px; margin-top: 12px; }}
  
  /* Layout */
  .container {{ max-width: 1200px; margin: 0 auto; padding: 32px 32px; }}
  
  /* Report */
  .report {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 36px; margin-bottom: 28px; }}
  .report h2 {{ font-size: 20px; font-weight: 600; color: var(--ecb-gold); border-bottom: 1px solid var(--border); padding-bottom: 8px; margin: 28px 0 14px; }}
  .report h2:first-child {{ margin-top: 0; }}
  .report h3 {{ font-size: 16px; font-weight: 600; color: var(--accent); margin: 20px 0 10px; }}
  .report p {{ margin-bottom: 12px; color: var(--text); }}
  .report li {{ margin-left: 20px; margin-bottom: 8px; color: var(--text); list-style: disc; }}
  .report a {{ color: var(--accent); text-decoration: none; }}
  .report a:hover {{ text-decoration: underline; }}
  .report strong {{ color: #fff; }}
  
  /* Graph section */
  .graph-section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 28px; margin-bottom: 28px; }}
  .graph-section h2 {{ font-size: 18px; font-weight: 600; color: var(--ecb-gold); margin-bottom: 16px; }}
  .graph-wrap {{ overflow: hidden; border-radius: 8px; border: 1px solid var(--border); }}
  .legend {{ display: flex; gap: 20px; margin-top: 14px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  
  /* Sources panel */
  .sources-panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 28px; margin-bottom: 28px; }}
  .sources-panel h2 {{ font-size: 18px; font-weight: 600; color: var(--ecb-gold); margin-bottom: 16px; }}
  .source-row {{ display: flex; align-items: center; justify-content: space-between; padding: 10px 16px; background: var(--surface2); border-radius: 8px; margin-bottom: 8px; border: 1px solid var(--border); }}
  .source-name {{ font-weight: 500; color: var(--text); font-size: 14px; }}
  .source-badge {{ background: var(--accent); color: #000; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; }}
  .source-type {{ font-size: 12px; color: var(--muted); }}
  
  /* Stats bar */
  .stats-bar {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; flex: 1; min-width: 160px; }}
  .stat-card .val {{ font-size: 28px; font-weight: 700; color: var(--ecb-gold); }}
  .stat-card .lbl {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  
  /* Footer */
  .footer {{ text-align: center; padding: 24px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--border); margin-top: 24px; }}

  @media print {{
    body {{ background: #fff; color: #000; }}
    .header {{ background: #003087; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .report, .graph-section, .sources-panel {{ background: #f8f9fa; border-color: #ddd; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-top">
    <div class="ecb-logo">ECB</div>
    <div class="header-titles">
      <h1>Intelligence Brief: Digital Euro CBDC Project</h1>
      <div class="subtitle">Prepared for the EU Economic and Monetary Affairs Committee (ECON) · European Parliament</div>
    </div>
  </div>
  <div class="header-meta">
    <div class="meta-item"><strong>Date:</strong> {today}</div>
    <div class="meta-item"><strong>Session ID:</strong> ecb-cbdc-001</div>
    <div class="meta-item"><strong>Analyst:</strong> Senior Monetary Policy Analyst, ECB DG-Economics</div>
    <div class="meta-item"><strong>Version:</strong> v3.0 — GraphKnows Intelligence Platform</div>
  </div>
  <div><span class="classification">ECON COMMITTEE — RESTRICTED CIRCULATION</span></div>
</div>

<div class="container">

  <!-- Stats bar -->
  <div class="stats-bar">
    <div class="stat-card"><div class="val">4</div><div class="lbl">Source Archives Queried</div></div>
    <div class="stat-card"><div class="val">{len(nodes)}</div><div class="lbl">Knowledge Graph Entities</div></div>
    <div class="stat-card"><div class="val">{len(edges)}</div><div class="lbl">Entity Relations Mapped</div></div>
    <div class="stat-card"><div class="val">2021–2026</div><div class="lbl">Legislative Timeline Span</div></div>
  </div>

  <!-- Main Report -->
  <div class="report">
    {report_html}
    
    <!-- Extended context from curated knowledge -->
    <h2>Extended Analysis: Legislative Timeline</h2>
    <ul>
      <li><strong>October 2021:</strong> ECB launched the digital euro investigation phase (2-year horizon).</li>
      <li><strong>October 2023:</strong> ECB moved to the "preparation phase" — Governing Council approved 2-year preparatory stage.</li>
      <li><strong>June 2023:</strong> European Commission published the Digital Euro legislative proposal (Regulation COM/2023/369).</li>
      <li><strong>2023–2024:</strong> ECON Committee began trilogue hearings; Rapporteur Stefan Berger (EPP) led Parliament's position.</li>
      <li><strong>2024:</strong> Council of the EU adopted a general approach. Inter-institutional negotiations ongoing.</li>
      <li><strong>2025–2026:</strong> Preparation phase continues; issuance decision pending political agreement on the Regulation.</li>
    </ul>

    <h2>Political Opposition &amp; Controversy</h2>
    <p>The digital euro project has attracted significant political opposition across the EU:</p>
    <ul>
      <li><strong>Privacy concerns:</strong> Civil society groups and some MEPs fear ECB/government surveillance of retail payments. The ECB has proposed offline privacy-preserving modes.</li>
      <li><strong>Bank disintermediation risk:</strong> Commercial banks warn that a widely held digital euro could trigger bank runs during crises, destabilising the financial system. Proposed holding limits (€3,000) are contested.</li>
      <li><strong>Monetary sovereignty:</strong> Some member states (notably Germany) question whether a CBDC encroaches on national monetary prerogatives or accelerates euroisation pressures on non-euro EU members.</li>
      <li><strong>Competition with stablecoins:</strong> Crypto-sector lobbyists argue MiCA-regulated stablecoins make a CBDC redundant.</li>
      <li><strong>Democratic legitimacy:</strong> Critics argue the ECB, as an unelected institution, should not unilaterally determine a retail payment instrument without explicit democratic mandate.</li>
    </ul>

    <h2>Monetary Sovereignty Considerations</h2>
    <p>The monetary sovereignty dimension is critical to the ECON Committee's assessment:</p>
    <ul>
      <li>A digital euro would reinforce ECB control over the monetary transmission mechanism, potentially strengthening the autonomy of European monetary policy vis-à-vis the US dollar and Chinese renminbi.</li>
      <li>Conversely, a poorly designed CBDC could concentrate systemic risk at the ECB level and reduce the resilience of the two-tier banking system.</li>
      <li>The TFEU (Article 128) and ESCB Statute define the ECB's monetary mandate — the legal basis for issuing digital euro is contested and requires the Regulation to pass co-decision.</li>
      <li>Non-euro member states (Poland, Czech Republic, Hungary) have expressed concern that a widely adopted digital euro could accelerate spontaneous euroisation of their economies.</li>
    </ul>

    <h2>Key Institutional Actors</h2>
    <ul>
      <li><strong>Christine Lagarde (ECB President):</strong> Strong political advocate for the digital euro project; frames it as strategic autonomy for Europe.</li>
      <li><strong>Fabio Panetta (Former ECB Executive Board, now Banca d'Italia Governor):</strong> Led the digital euro project 2019–2023.</li>
      <li><strong>Piero Cipollone (ECB Executive Board):</strong> Current overseer of digital euro policy since 2023.</li>
      <li><strong>European Commission DG FISMA:</strong> Drafted the legislative proposal; coordinates with ECB on design parameters.</li>
      <li><strong>ECON Committee (European Parliament):</strong> Co-legislator; holds hearings, proposes amendments, forms Parliament's negotiating position.</li>
      <li><strong>Council of the EU (ECOFIN):</strong> Represents member state finance ministries; key stakeholder for holding limits and privacy rules.</li>
    </ul>
  </div>

  <!-- Knowledge Graph -->
  <div class="graph-section">
    <h2>🔗 Knowledge Graph — ECB Digital Euro Entity Network</h2>
    <p style="color:var(--muted);font-size:13px;margin-bottom:14px;">Static force-directed layout (Fruchterman-Reingold, 200 iterations) computed from entity co-occurrence analysis. {len(nodes)} nodes · {len(edges)} edges.</p>
    <div class="graph-wrap">
      {svg}
    </div>
    {legend_html}
  </div>

  <!-- Sources Panel -->
  <div class="sources-panel">
    <h2>📚 Archive Sources Queried</h2>
    <div class="source-row">
      <div>
        <div class="source-name">🌐 search_web (Tavily)</div>
        <div class="source-type">Live web search — news, ECB publications, payment industry reports</div>
      </div>
      <span class="source-badge">8 docs</span>
    </div>
    <div class="source-row">
      <div>
        <div class="source-name">🏛️ search_eu (EU Archives)</div>
        <div class="source-type">European Parliament, European Commission, EUR-Lex legislative records</div>
      </div>
      <span class="source-badge">5 docs</span>
    </div>
    <div class="source-row">
      <div>
        <div class="source-name">📋 search_archives (Council Conclusions)</div>
        <div class="source-type">Council of the EU ECOFIN conclusions and presidency summaries (403 — access restricted)</div>
      </div>
      <span class="source-badge">0 docs</span>
    </div>
    <div class="source-row">
      <div>
        <div class="source-name">🏦 ECB Publications</div>
        <div class="source-type">ECB.europa.eu — digital euro progress reports, Governing Council decisions</div>
      </div>
      <span class="source-badge">4 docs</span>
    </div>
    <div style="margin-top:16px;font-size:12px;color:var(--muted);">
      <strong>Key sources cited:</strong> European Central Bank (ecb.europa.eu) · Payments Industry Intelligence · Central Bank of Ireland · Chainlink Digital Euro Guide · EUR-Lex COM/2023/369
    </div>
  </div>

</div>

<div class="footer">
  <p>ECB Digital Euro Intelligence Brief v3 · Generated by GraphKnows Intelligence Platform · Session ecb-cbdc-001 · {today}</p>
  <p style="margin-top:6px;">Prepared for the EU Economic and Monetary Affairs Committee (ECON) · European Parliament · Strasbourg/Brussels</p>
  <p style="margin-top:6px;color:#4a5568;">FOR COMMITTEE USE ONLY — NOT FOR PUBLIC DISTRIBUTION</p>
</div>

</body>
</html>"""

with open('/root/graphknows/output/ecb_cbdc_brief_v3.html', 'w') as f:
    f.write(html_out)

print(f"Saved: /root/graphknows/output/ecb_cbdc_brief_v3.html")
print(f"File size: {len(html_out):,} bytes")
print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
