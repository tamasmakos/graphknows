# What this graph can and cannot answer

A graphify graph is a **reference graph**: symbols, files, and edges between
them, extracted syntactically. That fixes what is derivable from it.

## Tier 1 — implemented here

A reference graph is genuinely sufficient for these.

| Method | Command | Note |
|---|---|---|
| PageRank centrality | `pagerank` | Power iteration, O(k·E). Hand-rolled: `nx.pagerank` needs scipy, which graphify's env lacks. |
| Tarjan SCC | `cycles` | O(V+E). `--level symbol` for call cycles, `file` for module cycles. Self-loops are filtered, so direct recursion does not register. |
| Louvain communities | `communities` | networkx built-in, no extra dependency. |
| Fan-in/fan-out | `coupling` | Henry–Kafura `LOC × (fan_in × fan_out)²`. |
| Martin metrics | `martin` | I and D only; see the A-free caveat in SKILL.md. |
| Architectural smells | `smells` | Hub-like at degree > p95 with both directions non-trivial; god by LOC or symbol count (Arcan definitions). |
| Propagation cost | `propagation` | Reachability per node over a visited set, O(V·E) — the textbook O(V³) closure is never materialised. |
| Diff risk surface | `review` | Reverse-reachability from the changed set. |
| Impact + parallel batching | `plan` | Closure overlap, then connected components. |

## Tier 2 — needs name resolution, which graphify does not do

Tree-sitter extraction binds no names: no cross-file symbol resolution, no type
inference, no call-target resolution. So a *resolved* call graph — and anything
built on one — is out of reach without adding a resolution layer (stack graphs,
or a SCIP indexer such as `scip-python`).

The sharpest consequence: **reachability-based SCA**. Intersecting a call graph
with a CVE function list is the right way to cut vulnerability noise, but
graphify's `calls` edges are syntactic and would **under-report** reachable
vulnerabilities — the one direction a security tool must never be wrong in. Use
`osv-scanner` for advisories and do not filter them with this graph.

## Tier 3 — needs a CFG/PDG, out of scope for any graph tool

Slicing, taint and points-to need control and data dependence, not references.
Building that is a static analyser, not a skill.

| Method | Use instead |
|---|---|
| Program slicing (backward/forward) | `joern` — builds a real CPG/SDG, has `.slice` |
| Taint analysis (source → sink) | `semgrep --config=auto`, CodeQL, or joern |
| Points-to (Andersen / Steensgaard) | Internal to joern/CodeQL — not a CLI you drive directly |
| Dominator tree | `networkx.immediate_dominators` is one call, but it needs a **CFG**; graphify has none |
| CFL/Dyck-reachability | This is what SCIP and stack graphs *are*. Index with `scip-python`, don't reimplement |

## Needs the parse tree, which graphify does not persist

`.graphify_ast.json` holds extracted nodes and edges, not ASTs. Use the
existing tool; do not write your own.

| Method | Use instead |
|---|---|
| Cyclomatic complexity | `uvx lizard <path>` (polyglot) or `uvx radon cc -s <path>` (Python) |
| Cognitive complexity | `uvx complexipy <path>`, or lizard's `--Ecognitive` |
| AST subtree hashing / clones | `npx jscpd <path>`, or `uvx pylint --disable=all --enable=duplicate-code` |
| GumTree AST diff | `difft <old> <new>` (difftastic), or the `gumtree` CLI |

## Testing and QA, all off-the-shelf

| Method | Use instead |
|---|---|
| Graph coverage (node/edge) | `pytest --cov --cov-branch`; node coverage = statement, edge = branch |
| Basis path testing | The basis path count *is* cyclomatic complexity — `uvx lizard` gives it |
| Regression test selection | `uvx pytest-testmon`, or `pytest --picked` against a diff |

`review`'s regression surface is the graph-side half of test selection: it names
the files that reach the diff. Intersect that with your test paths to pick a
subset by hand.
