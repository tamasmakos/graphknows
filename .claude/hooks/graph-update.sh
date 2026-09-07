#!/usr/bin/env bash
# post-commit: refresh graphify-out after every commit (AST only, no LLM), snapshot the
# previous graph, and append one growth line per commit. Silent no-op without a graph,
# so worktrees (graphify-out is gitignored) and fresh clones skip it.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 0
[ -f graphify-out/graph.json ] || exit 0
sha=$(git rev-parse --short HEAD)
mkdir -p graphify-out/snapshots
# ponytail: ~7MB per commit; prune to last N or keep only prev.json if it grows
cp graphify-out/graph.json "graphify-out/snapshots/$sha.json"
graphify update . >/dev/null 2>&1 || { echo "graphify update failed" >&2; exit 0; }
python - "$sha" <<'PY' >> graphify-out/growth.jsonl
import json, sys, datetime
sha = sys.argv[1]
old = json.load(open(f"graphify-out/snapshots/{sha}.json"))
new = json.load(open("graphify-out/graph.json"))
key = lambda g: {n["id"] for n in g["nodes"]}
ekey = lambda g: {(e["source"], e["target"]) for e in g["links"]}
o, n, oe, ne = key(old), key(new), ekey(old), ekey(new)
print(json.dumps({"sha": sha, "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
  "nodes": len(n), "edges": len(ne), "communities": len({x.get("community") for x in new["nodes"]}),
  "nodes_added": len(n - o), "nodes_removed": len(o - n), "edges_added": len(ne - oe), "edges_removed": len(oe - ne)}))
PY
