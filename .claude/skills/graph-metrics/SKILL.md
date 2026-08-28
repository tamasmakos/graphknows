---
name: graph-metrics
description: "Architecture metrics over an existing graphify graph. Use to review a diff (what the changed files can break), to plan work (impact of a target file, and which targets are parallel-safe), or to answer what is architecturally central, cyclic, god-like, or expensive to change in a repo that has graphify-out/."
---

# /graph-metrics

Metrics over `graphify-out/graph.json`. graphify builds the graph; this reads it.

Everything here is **tier 1**: derivable from a reference graph alone, no name
resolution. What is not, and what to run instead, is in
[`METHODS.md`](METHODS.md) — read it before writing analysis code of your own,
or when asked for a method this skill lacks.

## Setup

`graphify-out/graph.json` must exist. If it does not, say so and offer
`/graphify` — do not build it unasked, extraction is expensive.

The script needs only **networkx**, which graphify's own env has. That env's
interpreter is recorded by graphify itself:

```bash
PY=$(python -c "print(open('graphify-out/.graphify_python', encoding='utf-8-sig').read().strip())")
M=~/.claude/skills/graph-metrics/metrics.py
"$PY" "$M" propagation --top 5     # smoke test: prints counts, not an empty report
```

`utf-8-sig` is not optional — graphify writes that file with a BOM, and a plain
read yields a path that will not execute.

Every command warns on stderr when the graph has fallen behind HEAD. A stale
graph reads a tree that has moved: **LOC, blast radius and the changed-file join
all go quietly wrong**. Report the warning; offer a rebuild.

## Reviewing a diff

```bash
"$PY" "$M" review --base main
```

Joins the diff against the graph and answers what a reviewer cannot see in the
patch itself:

- **regression surface** — every file that transitively reaches the diff, so
  every file whose behaviour it can move. Routinely an order of magnitude wider
  than the diff, and it is what to regression-test.
- **blast radius** (`reach`) per changed file, plus `hub` / `god` flags. A hub
  in the diff means the change lands where everything funnels through.
- **communities touched** — a diff spanning several crosses module seams, which
  is either the refactor you asked for or scope creep.
- files **absent from the graph** — new files, or a stale graph. Distinguish the
  two before drawing conclusions; the count alone does not.

Takes `--base` (default `main`), `--top`, `--hub-degree`, `--god-loc`. Diffs
from the merge-base to the **working tree**, so uncommitted work is included.

## Planning work

```bash
"$PY" "$M" plan graphknows/memory.py graphknows/settings.py
```

Per target: how many files depend on it (the cost of getting it wrong) and how
many it depends on (the cost of understanding it), then the named dependents.

Then **parallel-safe batches**. Two targets conflict when their dependency
closures overlap — work on one moves ground the other stands on. Conflict is
transitive, so batches are the connected components of the conflict graph:
serialise inside a batch, run separate batches concurrently. One batch
containing everything is the honest answer that this work does not parallelise.

## The seven metrics

```
pagerank      # architecturally significant symbols, for context budgets
cycles        # circular dependencies (Tarjan SCC), --level file|symbol
communities   # module decomposition (Louvain)
coupling      # fan-in/fan-out + Henry-Kafura information flow
martin        # package instability I = Ce/(Ca+Ce)
smells        # hub-like and god components
propagation   # transitive visibility density + per-file blast radius
```

All take `--top N` (default 20), `--graph PATH`, `--root PATH`. `smells` takes
`--god-loc` / `--god-symbols`. With no argument at all, run `smells` then
`propagation` — together they answer "where is this codebase hard to change",
which is what people mean when they ask for metrics without naming one.

**Report the numbers, then say what they mean for the question actually asked.**
A table of Ce/Ca values is not an answer to "can I split this module".

## Direction is reconstructed, not trusted

graphify writes `"directed": false` unless built with `--directed`, but the
links keep their semantic orientation (caller → callee, importer → imported).
The script rebuilds the digraph from the links. **Do not "fix" this by
rebuilding the graph with `--directed` and assuming the numbers change** — they
should not.

Edges point dependent → dependency, so plain PageRank scores what is depended
**on**. Reversing the graph ranks in-degree-0 entry points instead — the
opposite of what a reader needs first.

Containment edges (`contains`, `method`) are excluded from the dependency graph.
Including them makes every file depend on its own symbols, which drives every
instability score to 0 and every fan-out up by the file's symbol count.
`references` is excluded too: on a graph carrying docs it is mostly prose
citing prose, and it manufactures cycles between markdown files.

## Honest limits

- **`calls` edges are syntactic** — AST extraction without name resolution, so
  dynamic dispatch, decorators and re-exports are missed or mis-bound. Good
  enough for centrality and clustering; not for security reachability.
- **Louvain is randomised.** `--seed` is pinned to 0 so runs are comparable.
  Louvain can leave a community internally disconnected where Leiden would not;
  check a split before acting on it. Leiden needs `leidenalg`, not installed.
- **Martin's D is A-free.** Abstractness is not derivable from a reference
  graph, so D is computed against A=0 and collapses to `1 − I`. Read it as
  relative instability, not Martin's metric.
- **Test files are in the graph.** They inflate fan-in on whatever they exercise
  and often top the blast-radius list. That is not a defect in the code under
  test.
