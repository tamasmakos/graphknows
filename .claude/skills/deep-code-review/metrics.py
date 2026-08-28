"""Scoreboard for /deep-code-review.

Computes the verified metric set from a graphify graph.json
(see docs/research/2026-08-10-graphify-code-metrics.md for feasibility audit).

Usage: python metrics.py [path/to/graph.json]
"""

import collections
import json
import math
import subprocess
import sys

import networkx as nx

GRAPH = sys.argv[1] if len(sys.argv) > 1 else "graphify-out/graph.json"
TOP = 10


def nmi(xs: list, ys: list) -> float:
    n = len(xs)
    cx, cy, cxy = collections.Counter(xs), collections.Counter(ys), collections.Counter(zip(xs, ys))
    hx = -sum(c / n * math.log(c / n) for c in cx.values())
    hy = -sum(c / n * math.log(c / n) for c in cy.values())
    mi = sum(c / n * math.log((c / n) / ((cx[a] / n) * (cy[b] / n))) for (a, b), c in cxy.items())
    return mi / math.sqrt(hx * hy) if hx and hy else 0.0


g = json.load(open(GRAPH, encoding="utf-8"))
nodes = {n["id"]: n for n in g["nodes"]}
# hygiene: structural metrics use AST-extracted code nodes only (34% of nodes are rationale)
code = {i for i, n in nodes.items() if n.get("file_type") == "code" and n.get("_origin") == "ast"}
ast_edges = [
    e
    for e in g["links"]
    if e.get("_origin") == "ast" and e["source"] in code and e["target"] in code
]
src_file = lambda nid: nodes[nid].get("source_file", "")

print(f"=== 0. COVERAGE (trust envelope) — {GRAPH} @ {g.get('built_at_commit', '?')[:9]} ===")
print("relations:", dict(collections.Counter(e["relation"] for e in ast_edges).most_common()))
print("confidence:", dict(collections.Counter(e.get("confidence") for e in g["links"])))
calls = [e for e in ast_edges if e["relation"] == "calls"]
xfile = sum(1 for e in calls if src_file(e["source"]) != src_file(e["target"]))
print(f"calls: {len(calls)}, cross-file {xfile / len(calls):.0%}" if calls else "calls: 0")
print(
    "NOT measurable from this graph: CBO, feature envy (member calls unresolved), LCOM4 (no field nodes),"
)
print("cyclomatic/cognitive complexity (use radon on source).")

print("\n=== 1. FILE IMPORT GRAPH: Ca/Ce/Instability, SDP violations ===")
imp = nx.DiGraph()
for e in ast_edges:
    if e["relation"] in ("imports", "imports_from"):
        a, b = src_file(e["source"]), src_file(e["target"])
        if a and b and a != b:
            imp.add_edge(a, b)
inst = {f: imp.out_degree(f) / d for f in imp if (d := imp.in_degree(f) + imp.out_degree(f))}
print("top Ca (ripple risk):", sorted(imp.in_degree, key=lambda x: -x[1])[:TOP])
sdp = sorted(((inst[b] - inst[a], a, b) for a, b in imp.edges if inst[a] < inst[b]), reverse=True)
print(f"SDP violations (depends on something less stable): {len(sdp)}")
for d, a, b in sdp[:TOP]:
    print(f"  {a} (I={inst[a]:.2f}) -> {b} (I={inst[b]:.2f})  severity {d:.2f}")

print("\n=== 2. CYCLES / LAYERING / PROPAGATION ===")
sccs = [c for c in nx.strongly_connected_components(imp) if len(c) > 1]
print(f"non-trivial SCCs: {len(sccs)}", [sorted(c) for c in sccs[:3]])
n = imp.number_of_nodes()
desc = {f: len(nx.descendants(imp, f)) for f in imp}
print(f"propagation cost: {sum(desc.values()) / (n * n):.3f}  ({n} files)")
if not sccs:
    print(f"Lakos depth (longest import chain): {nx.dag_longest_path_length(imp)}")
blast = sorted(((len(nx.ancestors(imp, f)), f) for f in imp), reverse=True)
print("blast radius (files transitively importing it):", [(f, b) for b, f in blast[:TOP]])

print("\n=== 3. COMMUNITIES vs FOLDERS ===")
ug = nx.Graph((e["source"], e["target"]) for e in ast_edges)
part = collections.defaultdict(set)
for i in ug:
    part[nodes[i].get("community", i)].add(i)
print(f"modularity Q: {nx.community.modularity(ug, part.values()):.3f}  ({len(part)} communities)")
files = sorted({src_file(e["source"]) for e in ast_edges if e["relation"] == "contains"})
fnode = {src_file(e["source"]): e["source"] for e in ast_edges if e["relation"] == "contains"}
folder = lambda f: f.rsplit("/", 1)[0] if "/" in f else "."
print(
    f"community-vs-folder NMI: {nmi([nodes[fnode[f]].get('community') for f in files], [folder(f) for f in files]):.3f}"
)
by_file_comm = collections.defaultdict(set)
for i in code:
    if src_file(i):
        by_file_comm[src_file(i)].add(nodes[i].get("community"))
print(
    "tangled files (members in 3+ communities):",
    sorted(((f, len(c)) for f, c in by_file_comm.items() if len(c) > 2), key=lambda x: -x[1])[:TOP],
)

print("\n=== 4. HUBS / GOD NODES / RFC ===")
deg = collections.Counter()
for e in ast_edges:
    deg[e["source"]] += 1
    deg[e["target"]] += 1
print("top degree:", [(nodes[i]["label"], d) for i, d in deg.most_common(TOP)])
dep = nx.DiGraph(
    (e["source"], e["target"])
    for e in ast_edges
    if e["relation"] in ("calls", "imports", "imports_from", "indirect_call")
)
pr = nx.pagerank(dep.reverse(copy=False))
print(
    "top PageRank (de-facto core API):",
    [(nodes[i]["label"], round(v, 4)) for i, v in sorted(pr.items(), key=lambda x: -x[1])[:TOP]],
)
methods_of = collections.defaultdict(set)
for e in ast_edges:
    if e["relation"] == "method":
        methods_of[e["source"]].add(e["target"])
callees = collections.defaultdict(set)
for e in calls:
    callees[e["source"]].add(e["target"])
rfc = {c: len(ms) + len(set().union(*(callees[m] for m in ms))) for c, ms in methods_of.items()}
print(
    "top RFC (testability burden, >50 flag):",
    [(nodes[c]["label"], r) for c, r in sorted(rfc.items(), key=lambda x: -x[1])[:TOP]],
)

print("\n=== 5. EVOLUTION OVERLAY (git, last 6 months) ===")
# ponytail: single git-log pass, last 500 commits; Code-Maat-grade analysis if this proves useful
log = subprocess.run(
    ["git", "log", "--since=6.months", "-500", "--format=@%H", "--numstat"],
    capture_output=True,
    text=True,
).stdout
churn, commit_files, cur = collections.Counter(), [], set()
for line in log.splitlines():
    if line.startswith("@"):
        if cur:
            commit_files.append(cur)
        cur = set()
    elif line.strip():
        parts = line.split("\t")
        if len(parts) == 3 and parts[2].endswith(".py"):
            churn[parts[2]] += 1
            cur.add(parts[2])
if cur:
    commit_files.append(cur)
hotspots = sorted(((churn[f] * (imp.in_degree(f) + 1), f) for f in imp if churn[f]), reverse=True)
print("hotspots (churn x (Ca+1)):", [(f, s) for s, f in hotspots[:TOP]])
co = collections.Counter(
    pair
    for fs in commit_files
    if len(fs) < 20
    for pair in __import__("itertools").combinations(sorted(fs), 2)
)
static = {frozenset(e) for e in imp.edges}
hidden = [
    (p, c)
    for p, c in co.most_common()
    if c >= 5 and frozenset(p) not in static and p[0] in imp and p[1] in imp
]
print(f"hidden coupling (co-changed >=5x, no import edge): {len(hidden)}")
for p, c in hidden[:TOP]:
    print(f"  {p[0]} <-> {p[1]}  ({c}x)")
