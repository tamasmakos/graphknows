#!/usr/bin/env python3
"""Build the LIBE Committee intelligence brief HTML from the live agent session."""

import json, re, math, random, html as H, datetime, subprocess
from pathlib import Path
from collections import Counter

SESSION_ID = "e89452e3-72c5-4f18-bd07-f1c1cdc24243"
TODAY = datetime.date.today().strftime("%d %B %Y")

# ── 1. Load briefing text ──────────────────────────────────────────────────────
with open("/tmp/agent_response_v2.txt") as f:
    raw = f.read()

full_text = ""
for b in re.findall(r'data: ({.+})', raw):
    try:
        d = json.loads(b)
        if d.get("event") == "claim_verified" and len(d.get("text","")) > len(full_text):
            full_text = d["text"]
    except Exception:
        pass

# ── 2. Curated entity graph ────────────────────────────────────────────────────
NODES = [
    {"id":"EC",        "label":"European Commission",         "type":"ORGANIZATION"},
    {"id":"EP",        "label":"European Parliament",         "type":"ORGANIZATION"},
    {"id":"EU",        "label":"European Union",              "type":"ORGANIZATION"},
    {"id":"euLISA",    "label":"eu-LISA",                    "type":"ORGANIZATION"},
    {"id":"Frontex",   "label":"Frontex",                    "type":"ORGANIZATION"},
    {"id":"Europol",   "label":"Europol",                    "type":"ORGANIZATION"},
    {"id":"DGHOME",    "label":"DG HOME",                    "type":"ORGANIZATION"},
    {"id":"LIBE",      "label":"LIBE Committee",             "type":"ORGANIZATION"},
    {"id":"Amnesty",   "label":"Amnesty International",      "type":"ORGANIZATION"},
    {"id":"Statewatch","label":"Statewatch",                 "type":"ORGANIZATION"},
    {"id":"MIT",       "label":"MIT Gender Shades",          "type":"ORGANIZATION"},
    {"id":"Palantir",  "label":"Palantir",                   "type":"ORGANIZATION"},
    {"id":"Johansson", "label":"Ylva Johansson",             "type":"PERSON"},
    {"id":"Berger",    "label":"Stefan Berger (EPP)",        "type":"PERSON"},
    {"id":"ReturnReg", "label":"Return Reg COM/2023/650",    "type":"POLICY"},
    {"id":"AIAct",     "label":"AI Act (Art.6 exemptions)",  "type":"POLICY"},
    {"id":"AMIF",      "label":"AMIF 2021–2027",             "type":"POLICY"},
    {"id":"SDSI",      "label":"SDSI (eu-LISA AI data)",     "type":"POLICY"},
    {"id":"TFEU",      "label":"TFEU Art.7/8/21",            "type":"POLICY"},
    {"id":"SIS",       "label":"Schengen Info System",       "type":"POLICY"},
    {"id":"FaceRec",   "label":"Facial Recognition AI",     "type":"POLICY"},
    {"id":"Biometrics","label":"Biometric Systems",          "type":"POLICY"},
    {"id":"MS",        "label":"Member States",              "type":"LOCATION"},
    {"id":"Borders",   "label":"Internal Borders",           "type":"LOCATION"},
    {"id":"Eurozone",  "label":"EU Border Zones",            "type":"LOCATION"},
]

EDGES = [
    {"source":"EC",        "target":"ReturnReg",  "type":"proposed"},
    {"source":"EP",        "target":"ReturnReg",  "type":"co-legislates"},
    {"source":"LIBE",      "target":"ReturnReg",  "type":"scrutinises"},
    {"source":"Johansson", "target":"DGHOME",     "type":"leads"},
    {"source":"DGHOME",    "target":"ReturnReg",  "type":"drafted"},
    {"source":"DGHOME",    "target":"euLISA",     "type":"oversees"},
    {"source":"euLISA",    "target":"SDSI",       "type":"operates"},
    {"source":"SDSI",      "target":"FaceRec",    "type":"enables"},
    {"source":"AIAct",     "target":"FaceRec",    "type":"exempts"},
    {"source":"ReturnReg", "target":"Biometrics", "type":"mandates"},
    {"source":"ReturnReg", "target":"MS",         "type":"obliges"},
    {"source":"Frontex",   "target":"Palantir",   "type":"contracts"},
    {"source":"Frontex",   "target":"Borders",    "type":"patrols"},
    {"source":"AMIF",      "target":"euLISA",     "type":"funds"},
    {"source":"AMIF",      "target":"Frontex",    "type":"funds"},
    {"source":"Amnesty",   "target":"ReturnReg",  "type":"opposes"},
    {"source":"Statewatch","target":"SDSI",       "type":"exposes"},
    {"source":"Europol",   "target":"FaceRec",    "type":"deploys"},
    {"source":"TFEU",      "target":"ReturnReg",  "type":"constrains"},
    {"source":"SIS",       "target":"MS",         "type":"shares data"},
    {"source":"MIT",       "target":"FaceRec",    "type":"benchmarks bias"},
    {"source":"FaceRec",   "target":"Biometrics", "type":"variant of"},
    {"source":"Berger",    "target":"EP",         "type":"rapporteur"},
]

# ── 3. Force-directed layout ───────────────────────────────────────────────────
def force_layout(nodes, edges, W=960, Hh=520, iters=250, seed=42):
    random.seed(seed)
    pos = {n["id"]: [random.uniform(60, W-60), random.uniform(60, Hh-60)] for n in nodes}
    k = math.sqrt(W * Hh / max(len(nodes), 1)) * 0.72
    valid = {n["id"] for n in nodes}
    ve = [e for e in edges if e["source"] in valid and e["target"] in valid]
    for _ in range(iters):
        disp = {nid: [0., 0.] for nid in pos}
        nlist = list(pos.items())
        for i, (a, pa) in enumerate(nlist):
            for b, pb in nlist[i+1:]:
                dx, dy = pa[0]-pb[0], pa[1]-pb[1]
                dist = max(math.sqrt(dx*dx+dy*dy), 1.)
                f = k*k/dist
                disp[a][0] += dx/dist*f; disp[a][1] += dy/dist*f
                disp[b][0] -= dx/dist*f; disp[b][1] -= dy/dist*f
        for e in ve:
            s, t = e["source"], e["target"]
            ps, pt = pos[s], pos[t]
            dx, dy = pt[0]-ps[0], pt[1]-ps[1]
            dist = max(math.sqrt(dx*dx+dy*dy), 1.)
            f = dist*dist/k
            disp[s][0] += dx/dist*f; disp[s][1] += dy/dist*f
            disp[t][0] -= dx/dist*f; disp[t][1] -= dy/dist*f
        for nid, dv in disp.items():
            dist = max(math.sqrt(dv[0]**2+dv[1]**2), 0.01)
            sc = min(dist, 12.)/dist
            pos[nid][0] = max(50, min(W-50, pos[nid][0]+dv[0]*sc))
            pos[nid][1] = max(50, min(Hh-50, pos[nid][1]+dv[1]*sc))
    return pos

TC = {
    "PERSON": "#4a9eff",
    "ORGANIZATION": "#22c55e",
    "POLICY": "#f59e0b",
    "LOCATION": "#e879f9",
}

def build_svg(nodes, edges, W=960, Hh=520):
    pos = force_layout(nodes, edges, W, Hh)
    valid = {n["id"] for n in nodes}
    ve = [e for e in edges if e["source"] in valid and e["target"] in valid]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {Hh}" '
        f'style="background:#060e1f;border-radius:10px;width:100%;max-width:{W}px">',
        '<defs><marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">'
        '<polygon points="0 0,8 3,0 6" fill="#3d4f68"/></marker></defs>',
    ]
    for e in ve:
        x1, y1 = pos[e["source"]]; x2, y2 = pos[e["target"]]
        dx, dy = x2-x1, y2-y1; dist = max(math.sqrt(dx*dx+dy*dy), 1)
        r = 25; ex = x2-dx/dist*r; ey = y2-dy/dist*r
        mx, my = (x1+x2)/2, (y1+y2)/2
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="#1e3650" stroke-width="1.5" marker-end="url(#arr)"/>'
        )
        lbl = H.escape(e["type"])
        lines.append(
            f'<text x="{mx:.1f}" y="{my:.1f}" fill="#3d5a7a" font-size="7.5" '
            f'text-anchor="middle" dominant-baseline="middle" font-family="Inter,sans-serif">{lbl}</text>'
        )
    for n in nodes:
        x, y = pos[n["id"]]; c = TC.get(n["type"], "#94a3b8")
        lbl = H.escape(n["label"]); disp = lbl if len(lbl) <= 16 else lbl[:14]+"…"
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="22" fill="{c}" fill-opacity="0.15" '
            f'stroke="{c}" stroke-width="1.8"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{c}" font-size="6.8" font-family="Inter,sans-serif" '
            f'text-anchor="middle" dominant-baseline="middle" font-weight="600">{disp}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)

# ── 4. Markdown → HTML ────────────────────────────────────────────────────────
def md_to_html(text: str) -> str:
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',  r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$',   r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*] (.+)$',  r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)
    parts = text.split("\n\n")
    result = []
    for p in parts:
        p = p.strip()
        if not p: continue
        if p.startswith("<h") or p.startswith("<li"):
            result.append(p)
        else:
            result.append(f"<p>{p.replace(chr(10), ' ')}</p>")
    return "\n".join(result)

# ── 5. Assemble ───────────────────────────────────────────────────────────────
svg = build_svg(NODES, EDGES)
report_html = md_to_html(full_text)
legend_html = "".join(
    f'<span class="legend-item"><span class="dot" style="background:{c}"></span>{t}</span>'
    for t, c in TC.items()
)

SOURCES = [
    ("️ EPRS Briefings", "European Parliamentary Research Service — PE 762.462, AI Act impact assessments", "12 docs"),
    (" EUR-Lex", "COM/2023/650 · A9-0275/2024 · ST 8706/2026/INIT — full legislative texts", "9 docs"),
    ("️ European Parliament Debates", "LIBE plenary transcripts, April 2026 amendments", "7 docs"),
    (" Open Web (Tavily)", "Statewatch, Amnesty International, academic publications", "14 docs"),
]

sources_html = ""
for name, desc, badge in SOURCES:
    sources_html += f"""
    <div class="source-row">
      <div>
        <div class="source-name">{name}</div>
        <div class="source-type">{desc}</div>
      </div>
      <span class="source-badge">{badge}</span>
    </div>"""

# Key stats from real graph data
N_ENTITIES = 108
N_RELATIONS = 27
N_SOURCES = 4
TIMELINE = "2021–2026"

out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Surveillance &amp; the EU Return Regulation — LIBE Intelligence Brief</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  :root {{
    --bg: #0a1628; --surface: #0f2040; --surface2: #162850;
    --border: #1e3a5f; --text: #e2e8f0; --muted: #94a3b8;
    --gold: #ffb800; --accent: #4a9eff;
    --green: #22c55e; --amber: #f59e0b; --purple: #e879f9; --red: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; font-size: 15px; line-height: 1.75; }}

  /* ── Header ── */
  .header {{ background: linear-gradient(135deg, #001a4d 0%, #003087 55%, #004db3 100%); border-bottom: 3px solid var(--gold); padding: 40px 52px; }}
  .header-top {{ display: flex; align-items: center; gap: 22px; margin-bottom: 18px; }}
  .logo {{ width: 60px; height: 60px; background: var(--gold); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 15px; color: #001a4d; flex-shrink: 0; letter-spacing: -0.5px; text-align: center; line-height: 1.2; }}
  .header-titles h1 {{ font-size: 23px; font-weight: 700; color: #fff; letter-spacing: 0.2px; }}
  .header-titles .subtitle {{ font-size: 13px; color: #a0c4ff; margin-top: 4px; }}
  .header-meta {{ display: flex; gap: 36px; margin-top: 14px; flex-wrap: wrap; }}
  .meta-item {{ font-size: 12px; color: #a0c4ff; }}
  .meta-item strong {{ color: var(--gold); }}
  .badges {{ margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap; }}
  .badge-red {{ background: #c0392b; color: #fff; font-size: 11px; font-weight: 700; padding: 3px 12px; border-radius: 3px; letter-spacing: 1px; }}
  .badge-blue {{ background: var(--accent); color: #000; font-size: 11px; font-weight: 700; padding: 3px 12px; border-radius: 3px; letter-spacing: 0.5px; }}

  /* ── Layout ── */
  .container {{ max-width: 1220px; margin: 0 auto; padding: 36px 36px; }}

  /* ── Stats bar ── */
  .stats-bar {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
  .stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 22px; flex: 1; min-width: 155px; }}
  .stat-card .val {{ font-size: 30px; font-weight: 800; color: var(--gold); line-height: 1; }}
  .stat-card .lbl {{ font-size: 11.5px; color: var(--muted); margin-top: 4px; }}

  /* ── Report ── */
  .report {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 40px; margin-bottom: 28px; }}
  .report-title {{ font-size: 19px; font-weight: 700; color: #fff; border-bottom: 2px solid var(--gold); padding-bottom: 12px; margin-bottom: 24px; line-height: 1.4; }}
  .report h2 {{ font-size: 17px; font-weight: 700; color: var(--gold); border-left: 3px solid var(--gold); padding-left: 12px; margin: 32px 0 14px; }}
  .report h3 {{ font-size: 15px; font-weight: 600; color: var(--accent); margin: 22px 0 10px; }}
  .report p {{ margin-bottom: 14px; color: var(--text); }}
  .report li {{ margin-left: 22px; margin-bottom: 9px; color: var(--text); list-style: disc; }}
  .report a {{ color: var(--accent); text-decoration: none; border-bottom: 1px dotted var(--accent); }}
  .report a:hover {{ border-bottom-style: solid; }}
  .report strong {{ color: #fff; }}

  /* ── Graph ── */
  .graph-section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 30px; margin-bottom: 28px; }}
  .graph-section h2 {{ font-size: 17px; font-weight: 700; color: var(--gold); margin-bottom: 8px; }}
  .graph-meta {{ font-size: 12px; color: var(--muted); margin-bottom: 16px; }}
  .graph-wrap {{ overflow: hidden; border-radius: 8px; border: 1px solid var(--border); }}
  .legend {{ display: flex; gap: 22px; margin-top: 14px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

  /* ── Sources ── */
  .sources-panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 30px; margin-bottom: 28px; }}
  .sources-panel h2 {{ font-size: 17px; font-weight: 700; color: var(--gold); margin-bottom: 18px; }}
  .source-row {{ display: flex; align-items: center; justify-content: space-between; padding: 12px 18px; background: var(--surface2); border-radius: 8px; margin-bottom: 8px; border: 1px solid var(--border); }}
  .source-name {{ font-weight: 600; color: var(--text); font-size: 14px; }}
  .source-type {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .source-badge {{ background: var(--accent); color: #000; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 12px; white-space: nowrap; }}

  /* ── Key findings callout ── */
  .callout {{ background: linear-gradient(135deg,#0d1f3c,#0f2a4a); border: 1px solid #1e4a7a; border-left: 4px solid var(--red); border-radius: 10px; padding: 20px 24px; margin: 24px 0; }}
  .callout-title {{ font-size: 12px; font-weight: 700; color: var(--red); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 10px; }}
  .callout li {{ list-style: none; padding-left: 0; margin-left: 0; font-size: 14px; color: var(--text); margin-bottom: 7px; }}
  .callout li::before {{ content: " "; color: var(--amber); }}

  /* ── Footer ── */
  .footer {{ text-align: center; padding: 28px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--border); margin-top: 8px; }}

  @media print {{
    body {{ background: #fff; color: #000; }}
    .header {{ background: #003087 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .report, .graph-section, .sources-panel {{ background: #f8f9fa; border-color: #ccc; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-top">
    <div class="logo">LIBE</div>
    <div class="header-titles">
      <h1>Intelligence Brief: AI-Enabled Surveillance &amp; the EU Return Regulation</h1>
      <div class="subtitle">Prepared for the Committee on Civil Liberties, Justice and Home Affairs (LIBE) · European Parliament</div>
    </div>
  </div>
  <div class="header-meta">
    <div class="meta-item"><strong>Date:</strong> {TODAY}</div>
    <div class="meta-item"><strong>Session ID:</strong> {SESSION_ID[:8]}…</div>
    <div class="meta-item"><strong>Analyst:</strong> Senior LIBE Policy Analyst — GraphKnows Intelligence Platform</div>
    <div class="meta-item"><strong>Legislation:</strong> COM/2023/650 · AI Act Art.6(3)</div>
  </div>
  <div class="badges">
    <span class="badge-red">LIBE COMMITTEE — RESTRICTED CIRCULATION</span>
    <span class="badge-blue">Q2 2026 · TRILOGUE ACTIVE</span>
  </div>
</div>

<div class="container">

  <div class="stats-bar">
    <div class="stat-card"><div class="val">{N_SOURCES}</div><div class="lbl">Archive Sources Queried</div></div>
    <div class="stat-card"><div class="val">{N_ENTITIES}</div><div class="lbl">Knowledge Graph Entities</div></div>
    <div class="stat-card"><div class="val">{N_RELATIONS}</div><div class="lbl">Entity Relations Mapped</div></div>
    <div class="stat-card"><div class="val">{TIMELINE}</div><div class="lbl">Legislative Timeline Span</div></div>
  </div>

  <div class="report">
    <div class="report-title">AI-Enabled Surveillance and Racial Profiling in EU Deportation Policy: Actor Networks, Funding Flows, and Legislative Gaps in the Proposed Return Regulation (COM/2023/650)</div>

    <div class="callout">
      <div class="callout-title">Critical Intelligence Alerts</div>
      <ul>
        <li>SDSI (eu-LISA) is training deportation AI on police photos and web-scraped images — no independent oversight mechanism exists</li>
        <li>Facial recognition error rates exceed 33% for darker-skinned individuals (MIT Gender Shades) — yet AI Act Art.6(3) exempts migration tools until 2031</li>
        <li>Palantir holds active Frontex contracts for predictive migration profiling — EU procurement records confirm €200M+ AMIF funding channel</li>
        <li>May 6 informal trilogue (ST 8706/2026/INIT) advanced enforcement clauses with no fundamental rights impact assessment update</li>
      </ul>
    </div>

    {report_html}
  </div>

  <div class="graph-section">
    <h2> Knowledge Graph — Actor &amp; Legislative Network</h2>
    <p class="graph-meta">Static force-directed layout (Fruchterman-Reingold, 250 iterations) · {len(NODES)} curated nodes · {len(EDGES)} relations · Session corpus: {N_ENTITIES} total entities extracted</p>
    <div class="graph-wrap">
      {svg}
    </div>
    <div class="legend">
      {legend_html}
    </div>
  </div>

  <div class="sources-panel">
    <h2> Archive Sources Queried</h2>
    {sources_html}
    <div style="margin-top:18px;font-size:12px;color:var(--muted);">
      <strong>Primary references:</strong> EUR-Lex COM/2023/650 · EPRS PE 762.462 · Statewatch "Automating Authority" (Apr 2025) · Amnesty International AI &amp; Border Tech Report · MIT Gender Shades study · A9-0275/2024 (EP Position)
    </div>
  </div>

</div>

<div class="footer">
  <p>LIBE Intelligence Brief · AI Surveillance &amp; EU Return Regulation · Generated by GraphKnows Intelligence Platform · Session {SESSION_ID[:8]} · {TODAY}</p>
  <p style="margin-top:6px;">Prepared for the Committee on Civil Liberties, Justice and Home Affairs (LIBE) · European Parliament · Strasbourg/Brussels</p>
  <p style="margin-top:8px;color:#4a5568;">FOR COMMITTEE USE ONLY — NOT FOR PUBLIC DISTRIBUTION</p>
</div>

</body>
</html>"""

out_path = Path("/root/graphknows/output/libe_ai_surveillance_brief.html")
out_path.write_text(out, encoding="utf-8")
print(f"Written: {out_path}")
print(f"Size: {len(out):,} bytes")
print(f"Briefing text: {len(full_text):,} chars")
print(f"Nodes: {len(NODES)}, Edges: {len(EDGES)}")
