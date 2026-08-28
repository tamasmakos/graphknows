#!/usr/bin/env python3
"""Architecture metrics over a graphify graph.

Reads ``graphify-out/graph.json`` and reports the metrics that are genuinely
derivable from a reference graph: centrality, cycles, communities, coupling,
package stability, architectural smells, change propagation.

Direction matters for every one of these, and graphify writes
``"directed": false`` unless built with ``--directed``. The orientation is not
lost, though: each link keeps ``source`` -> ``target`` in its semantic sense
(caller -> callee, importer -> imported, container -> contained), so the
digraph is rebuilt from the links rather than trusted from the flag.

Not implemented here, deliberately — see SKILL.md:
cyclomatic/cognitive complexity, clone detection and AST diffing need the parse
tree, which graphify does not persist; slicing, taint and points-to need a
CFG/PDG that no reference graph can stand in for.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

# Edges that mean "A depends on B". Containment (`contains`, `method`) is
# structure, not dependency — rolling it into the dependency graph would make
# every file depend on its own symbols and flatten instability to 0.
DEPENDENCY = frozenset(
    {"calls", "imports", "imports_from", "uses", "inherits", "indirect_call", "re_exports"}
)
CONTAINMENT = frozenset({"contains", "method"})


# Labels carry arrows and other non-ASCII; a Windows console defaults to cp1252
# and dies on them mid-report.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _git(root: Path, *argv: str) -> str | None:
    """Best-effort git; None outside a repo, or without git on PATH."""
    try:
        done = subprocess.run(
            ("git", "-C", str(root), *argv), capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip()


def warn_if_stale(data: dict[str, Any], root: Path) -> None:
    """A graph outlives the tree it was built from, and degrades quietly when it does.

    Every path-based number here — LOC, blast radius, the changed-file join —
    reads the wrong tree once the graph falls behind. Warn on stderr so the
    report on stdout stays parseable.
    """
    built, head = data.get("built_at_commit"), _git(root, "rev-parse", "HEAD")
    if not built or not head or built == head:
        return
    behind = _git(root, "rev-list", "--count", f"{built}..HEAD")
    drift = f"{behind} commits behind" if behind else "at an unknown distance"
    print(
        f"! graph built at {built[:8]}, HEAD is {head[:8]} — {drift}. Rerun /graphify.",
        file=sys.stderr,
    )


def load(graph_path: Path) -> dict[str, Any]:
    if not graph_path.exists():
        sys.exit(f"no graph at {graph_path} — run /graphify first")
    return json.loads(graph_path.read_text(encoding="utf-8"))


def symbol_graph(data: dict[str, Any]) -> nx.DiGraph:
    """Symbol-level dependency digraph."""
    g = nx.DiGraph()
    for node in data["nodes"]:
        g.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for link in data["links"]:
        if link.get("relation") in DEPENDENCY:
            g.add_edge(link["source"], link["target"], relation=link["relation"])
    return g


def file_graph(data: dict[str, Any]) -> nx.DiGraph:
    """File-level dependency digraph, with symbol edges lifted to their files.

    Self-loops are dropped: a file calling its own functions is not a
    dependency, and leaving them in inflates both Ca and Ce for every file.
    """
    owner = {n["id"]: n.get("source_file") for n in data["nodes"]}
    g = nx.DiGraph()
    for path in filter(None, owner.values()):
        g.add_node(path)
    for link in data["links"]:
        if link.get("relation") not in DEPENDENCY:
            continue
        src, dst = owner.get(link["source"]), owner.get(link["target"])
        if src and dst and src != dst:
            g.add_edge(src, dst)
    return g


def _package(path: str) -> str:
    parent = str(Path(path).parent).replace("\\", "/")
    return parent or "."


def _pagerank(
    g: nx.DiGraph, damping: float, tol: float = 1e-8, max_iter: int = 100
) -> dict[str, float]:
    """Power iteration, O(k*E).

    Hand-rolled rather than nx.pagerank because that dispatches to scipy, which
    the graphify tool env does not ship — and one function is not worth a
    numerical stack. Dangling nodes redistribute uniformly; without that their
    rank leaks out and the vector stops summing to 1.
    """
    nodes = list(g)
    if not nodes:
        return {}
    n = len(nodes)
    rank = dict.fromkeys(nodes, 1.0 / n)
    out_degree = {v: g.out_degree(v) for v in nodes}
    dangling = [v for v in nodes if not out_degree[v]]

    for _ in range(max_iter):
        leaked = damping * sum(rank[v] for v in dangling) / n
        nxt = dict.fromkeys(nodes, (1.0 - damping) / n + leaked)
        for v in nodes:
            if out_degree[v]:
                share = damping * rank[v] / out_degree[v]
                for w in g.successors(v):
                    nxt[w] += share
        if sum(abs(nxt[v] - rank[v]) for v in nodes) < tol * n:
            return nxt
        rank = nxt
    return rank


def cmd_pagerank(data: dict[str, Any], args: argparse.Namespace) -> None:
    """Rank architecturally significant symbols. Power iteration, O(k*E)."""
    g = symbol_graph(data)
    # Edges already point dependent -> dependency, so plain PageRank sends score
    # to what is depended ON. Reversing it ranks entry points instead — the
    # in-degree-0 callers — which is the opposite of what a reader needs first.
    ranks = _pagerank(g, args.damping)
    label = {n["id"]: n.get("label", n["id"]) for n in data["nodes"]}
    where = {n["id"]: n.get("source_file", "") for n in data["nodes"]}
    print(f"{'score':>8}  {'symbol':<40} file")
    for node, score in sorted(ranks.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"{score:8.5f}  {label[node][:40]:<40} {where[node]}")


def cmd_cycles(data: dict[str, Any], args: argparse.Namespace) -> None:
    """Circular dependencies. Tarjan's SCC, O(V+E)."""
    g = file_graph(data) if args.level == "file" else symbol_graph(data)
    label = {n["id"]: n.get("label", n["id"]) for n in data["nodes"]}
    cycles = [c for c in nx.strongly_connected_components(g) if len(c) > 1]
    if not cycles:
        print(f"no {args.level}-level cycles")
        return
    print(f"{len(cycles)} cyclic component(s), largest first\n")
    for members in sorted(cycles, key=len, reverse=True)[: args.top]:
        names = sorted(label.get(m, m) for m in members)
        print(f"  [{len(members)}] {', '.join(names[:8])}{' …' if len(names) > 8 else ''}")


def cmd_communities(data: dict[str, Any], args: argparse.Namespace) -> None:
    """Module decomposition. Louvain, near-linear.

    networkx ships Louvain, so this needs no python-louvain/leidenalg. Louvain
    can leave internally disconnected communities where Leiden would not; for a
    remodularisation you intend to act on, check the split by hand.
    """
    g = file_graph(data).to_undirected()
    if g.number_of_nodes() == 0:
        sys.exit("no file-level dependency edges — was the graph built from code?")
    parts = nx.community.louvain_communities(g, seed=args.seed)
    modularity = nx.community.modularity(g, parts)
    print(f"{len(parts)} communities, modularity {modularity:.3f}\n")
    for i, members in enumerate(sorted(parts, key=len, reverse=True)[: args.top]):
        packages = sorted({_package(m) for m in members})
        print(f"  community {i} [{len(members)} files] {', '.join(packages[:5])}")


def cmd_coupling(data: dict[str, Any], args: argparse.Namespace) -> None:
    """Fan-in/fan-out and Henry-Kafura information flow, length*(in*out)^2."""
    g = file_graph(data)
    loc = _loc_by_file(g.nodes, args.root)
    rows = []
    for path in g.nodes:
        fan_in, fan_out = g.in_degree(path), g.out_degree(path)
        rows.append((loc.get(path, 1) * (fan_in * fan_out) ** 2, fan_in, fan_out, path))
    print(f"{'flow':>12} {'in':>5} {'out':>5}  file")
    for flow, fan_in, fan_out, path in sorted(rows, reverse=True)[: args.top]:
        print(f"{flow:12d} {fan_in:5d} {fan_out:5d}  {path}")


def cmd_martin(data: dict[str, Any], args: argparse.Namespace) -> None:
    """Package instability I = Ce/(Ca+Ce), and distance from the main sequence.

    Abstractness needs a notion of abstract type that a reference graph does not
    carry, so D is reported against A=0. Read D as "how unstable is this package
    relative to how much depends on it", not as Martin's full metric.
    """
    g = file_graph(data)
    packages: dict[str, set[str]] = defaultdict(set)
    for path in g.nodes:
        packages[_package(path)].add(path)

    rows = []
    for pkg, files in packages.items():
        efferent = {_package(dst) for f in files for dst in g.successors(f) if _package(dst) != pkg}
        afferent = {
            _package(src) for f in files for src in g.predecessors(f) if _package(src) != pkg
        }
        ca, ce = len(afferent), len(efferent)
        if not (ca + ce):
            # Nothing depends on it and it depends on nothing, so I is undefined.
            # Scoring it 0 would rank docs/ as the most painful package in the
            # repo, which is how this metric earns a reputation for noise.
            continue
        instability = ce / (ca + ce)
        rows.append((abs(instability - 1.0), instability, ca, ce, len(files), pkg))

    print(f"{'I':>6} {'D':>6} {'Ca':>4} {'Ce':>4} {'files':>6}  package")
    for distance, instability, ca, ce, n, pkg in sorted(rows, reverse=True)[: args.top]:
        print(f"{instability:6.2f} {distance:6.2f} {ca:4d} {ce:4d} {n:6d}  {pkg}")


def cmd_smells(data: dict[str, Any], args: argparse.Namespace) -> None:
    """Hub-like and God components, by degree and size thresholds."""
    g = file_graph(data)
    if g.number_of_nodes() == 0:
        sys.exit("no file-level dependency edges")
    degrees = sorted(g.in_degree(n) + g.out_degree(n) for n in g.nodes)
    cutoff = degrees[int(len(degrees) * 0.95)] if degrees else 0

    symbols: dict[str, int] = defaultdict(int)
    for link in data["links"]:
        if link.get("relation") in CONTAINMENT:
            symbols[link["source"]] += 1
    owner = {n["id"]: n.get("source_file") for n in data["nodes"]}
    per_file: dict[str, int] = defaultdict(int)
    for node_id, count in symbols.items():
        if (path := owner.get(node_id)) is not None:
            per_file[path] += count
    loc = _loc_by_file(g.nodes, args.root)

    print(f"hub-like (degree > p95 = {cutoff}, both directions non-trivial)")
    hubs = [
        (g.in_degree(n) + g.out_degree(n), n)
        for n in g.nodes
        if g.in_degree(n) + g.out_degree(n) > cutoff and g.in_degree(n) and g.out_degree(n)
    ]
    for degree, path in sorted(hubs, reverse=True)[: args.top]:
        print(f"  {degree:4d}  in={g.in_degree(path):3d} out={g.out_degree(path):3d}  {path}")

    print(f"\ngod components (LOC > {args.god_loc} or > {args.god_symbols} symbols)")
    gods = [
        (loc.get(p, 0), per_file.get(p, 0), p)
        for p in g.nodes
        if loc.get(p, 0) > args.god_loc or per_file.get(p, 0) > args.god_symbols
    ]
    for lines, count, path in sorted(gods, reverse=True)[: args.top]:
        print(f"  {lines:6d} LOC  {count:4d} symbols  {path}")


def cmd_propagation(data: dict[str, Any], args: argparse.Namespace) -> None:
    """Propagation cost: density of the transitive visibility matrix.

    Textbook transitive closure is O(V^3). Reachability per node over a bitset
    is O(V*E) and finishes in seconds on graphs this size, which is why the
    closure is never materialised as a matrix.
    """
    g = file_graph(data)
    order = list(g.nodes)
    if not order:
        sys.exit("no file-level dependency edges")
    index = {n: i for i, n in enumerate(order)}
    reach = 0
    for start in order:
        seen = {start}
        stack = [start]
        while stack:
            for nxt in g.successors(stack.pop()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        reach += len(seen) - 1  # a file reaching itself is not propagation

    n = len(order)
    cost = reach / (n * (n - 1)) if n > 1 else 0.0
    print(f"files                {n}")
    print(f"direct dependencies  {g.number_of_edges()}")
    print(f"transitive pairs     {reach}")
    print(f"propagation cost     {cost:.2%}")
    print(f"\n{'reaches':>8}  file  (blast radius of a change)")
    radius = [(len(nx.descendants(g, p)), p) for p in order]
    for count, path in sorted(radius, reverse=True)[: args.top]:
        print(f"{count:8d}  {path}")
    _ = index


def _flags(g: nx.DiGraph, path: str, loc: dict[str, int], args: argparse.Namespace) -> str:
    degree = g.in_degree(path) + g.out_degree(path)
    marks = []
    if degree > args.hub_degree and g.in_degree(path) and g.out_degree(path):
        marks.append("hub")
    if loc.get(path, 0) > args.god_loc:
        marks.append("god")
    return ",".join(marks)


def _changed_files(root: Path, base: str) -> list[str] | None:
    """Files differing between the merge-base with `base` and the working tree.

    Diffing from the merge-base rather than `base` itself keeps commits landed
    on the base branch out of the review; diffing to the working tree rather
    than HEAD keeps uncommitted work in it.
    """
    merge_base = _git(root, "merge-base", base, "HEAD")
    if merge_base is None:
        return None
    names = _git(root, "diff", "--name-only", merge_base)
    if names is None:
        return None
    return [line.strip() for line in names.splitlines() if line.strip()]


def _communities(g: nx.DiGraph, seed: int) -> dict[str, int]:
    parts = sorted(
        nx.community.louvain_communities(g.to_undirected(), seed=seed), key=len, reverse=True
    )
    return {member: i for i, part in enumerate(parts) for member in part}


def cmd_review(data: dict[str, Any], args: argparse.Namespace) -> None:
    """Risk surface of a diff: what the changed files carry, and what they can break.

    The regression surface is the union of reverse-dependency closures — every
    file that can reach something in the diff, so every file whose behaviour the
    diff can move. That is the set a reviewer has to hold in mind, and it is
    routinely an order of magnitude wider than the diff itself.
    """
    g = file_graph(data)
    changed = _changed_files(args.root, args.base)
    if changed is None:
        sys.exit(f"cannot diff against {args.base!r} — not a git repo, or unknown ref")

    known = [p for p in changed if p in g]
    unknown = [p for p in changed if p not in g]
    print(f"{len(changed)} changed vs merge-base with {args.base}, {len(known)} in the graph")
    if unknown:
        tail = " …" if len(unknown) > 5 else ""
        print(
            f"  {len(unknown)} absent from the graph (new files, or a stale graph): "
            f"{', '.join(unknown[:5])}{tail}"
        )
    if not known:
        return

    loc = _loc_by_file(known, args.root)
    community = _communities(g, args.seed)
    print(f"\n{'in':>4} {'out':>4} {'reach':>6} {'comm':>5}  {'file':<52} flags")
    ranked = sorted(((len(nx.descendants(g, p)), p) for p in known), reverse=True)
    for reach, path in ranked[: args.top]:
        print(
            f"{g.in_degree(path):4d} {g.out_degree(path):4d} {reach:6d} "
            f"{community.get(path, -1):5d}  {path[:52]:<52} {_flags(g, path, loc, args)}"
        )

    if len(ranked) > args.top:
        print(f"  … {len(ranked) - args.top} more, --top to widen")

    surface: set[str] = set()
    for path in known:
        surface |= nx.ancestors(g, path)
    surface -= set(known)
    touched = {community.get(p) for p in known}
    print(
        f"\nregression surface  {len(surface)} files reach the diff "
        f"({len(surface) / g.number_of_nodes():.1%} of the repo)"
    )
    print(f"communities touched {len(touched)} — a diff spanning several crosses module seams")
    for count, path in sorted(((g.out_degree(p), p) for p in surface), reverse=True)[: args.top]:
        print(f"  {count:4d} deps out  {path}")


def cmd_plan(data: dict[str, Any], args: argparse.Namespace) -> None:
    """Impact of touching the named files, and which of them are parallel-safe.

    Two targets conflict when their dependency closures overlap: work on one
    moves ground the other stands on. Conflict is transitive, so the batches are
    the connected components of the conflict graph — serialise inside a batch,
    run separate batches concurrently.
    """
    g = file_graph(data)
    targets = [p.replace("\\", "/") for p in args.paths]
    if not targets:
        sys.exit("plan takes one or more file paths")
    if missing := [p for p in targets if p not in g]:
        print(f"not in the graph: {', '.join(missing)}")
    targets = [p for p in targets if p in g]
    if not targets:
        return

    closure = {p: nx.descendants(g, p) | nx.ancestors(g, p) | {p} for p in targets}
    print(f"{'dependents':>10} {'deps':>6}  file")
    for path in sorted(targets, key=lambda p: -len(nx.ancestors(g, p))):
        dependents = sorted(nx.ancestors(g, path))
        print(f"{len(dependents):10d} {len(nx.descendants(g, path)):6d}  {path}")
        for dependent in dependents[: args.top]:
            print(f"{'':10}         <- {dependent}")

    conflict = nx.Graph()
    conflict.add_nodes_from(targets)
    for i, a in enumerate(targets):
        for b in targets[i + 1 :]:
            if closure[a] & closure[b]:
                conflict.add_edge(a, b)
    print("\nparallel-safe batches (batches are independent; serialise within a batch)")
    for i, batch in enumerate(nx.connected_components(conflict)):
        print(f"  batch {i}: {', '.join(sorted(batch))}")


def _loc_by_file(paths: Any, root: Path) -> dict[str, int]:
    """Physical line counts, for the metrics that are size-weighted.

    Missing files count 0 rather than raising: a graph outlives the tree it was
    built from, and a moved file should not take the whole report down with it.
    """
    counts: dict[str, int] = {}
    for path in paths:
        try:
            with (root / path).open("rb") as handle:
                counts[path] = sum(1 for _ in handle)
        except OSError:
            counts[path] = 0
    return counts


COMMANDS = {
    "pagerank": cmd_pagerank,
    "cycles": cmd_cycles,
    "communities": cmd_communities,
    "coupling": cmd_coupling,
    "martin": cmd_martin,
    "smells": cmd_smells,
    "propagation": cmd_propagation,
    "review": cmd_review,
    "plan": cmd_plan,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("paths", nargs="*", help="plan: the files you intend to change")
    parser.add_argument("--graph", type=Path, default=Path("graphify-out/graph.json"))
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="tree the graph was built from"
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--level", choices=("file", "symbol"), default="file")
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0, help="Louvain is randomised; pin it")
    parser.add_argument("--god-loc", type=int, default=750)
    parser.add_argument("--god-symbols", type=int, default=40)
    parser.add_argument("--hub-degree", type=int, default=20, help="review: hub-flag cutoff")
    parser.add_argument("--base", default="main", help="review: branch to diff against")
    args = parser.parse_args()
    data = load(args.graph)
    warn_if_stale(data, args.root)
    COMMANDS[args.command](data, args)


if __name__ == "__main__":
    main()
