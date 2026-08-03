# Topics

The topic layer groups the graph into themes, describes each one, and grades the
result. It is a **stage, not a flush step**: rebuilding topics at new settings
re-runs clustering, description and grading only — never ingestion, entity
resolution, PageRank or consolidation.

```bash
# rebuild in place
graphknows-topics --namespace mytenant --mode extractive --min-size 3

# compare settings before committing to one (never persists)
graphknows-topics --namespace mytenant --sweep-param min_size --sweep 2,3,5,10

# write a read-only JSON artifact alongside the DB write
graphknows-topics --namespace mytenant --export topics.json
```

Also available as `Memory.build_topics(...)` and the `memory_topics` MCP tool.
All three share one implementation, which `flush_memory` also calls — a rebuild
is never a different code path from the flush.

## Two engine facts that shaped this

Both verified against ArcadeDB 26.5.1, and both worth knowing before you tune:

**1. `algo.louvain` ignores its parameters.** `{resolution: 0.1}` and
`{resolution: 5.0}` return byte-identical partitions. The community layer
therefore has no granularity knob; `min_size` is the only lever over it.

**2. `algo.modularityScore` is not Newman modularity and must not be optimised.**
Measured on LoCoMo conv-26:

| partition | ArcadeDB score |
|---|---|
| real Louvain | 0.247 |
| **all-one-cluster** | **0.750** |
| random 7-way | 0.095 |

Newman's Q for a single all-encompassing cluster is 0 by definition. The native
score appears to omit the expected-edge term, so it rewards ever-coarser
partitions — a sweep optimising it converges on one giant topic. It *does*
separate real structure from random, so it is reported as
`arcadedb_modularity_untrustworthy` for continuity, and never tuned against.
The `newman_modularity` field is computed in Python and is the real objective.

## Choosing a source

`source` trades coverage against tunability.

| | `community` (default) | `graph_embedding` |
|---|---|---|
| Basis | Louvain partition | node2vec entity vectors, average-linkage |
| Granularity knob | none (only `min_size`) | `distance`, plus the node2vec params |
| Coverage | ~every entity | only entities with ≥3 relation edges |

The coverage caveat is not hypothetical: on conv-26, **33 of 1049 entities** have
a `graph_embedding`, because node2vec skips sparsely-connected nodes and a
conversational graph is sparse (327 REL edges over 1049 entities). Clustering
that source there produces topics describing 3% of the corpus. `build_topics`
logs a warning below 25% coverage, and `entity_coverage` is in the metrics —
check it before trusting a `graph_embedding` build.

node2vec's walk parameters are exposed (`n2v_walk_length`, `n2v_walks`,
`n2v_dimensions`, with `recompute_embeddings` to re-run it). Note what they do
and do not affect: they shape the embedding space, so they drive PPR seed
expansion, entity similarity, and `graph_embedding` topic granularity. They do
**not** influence the Louvain communities — nothing feeds node2vec output back
into community detection.

## Outputs

Per topic: `title` (top entities by PageRank), `summary`, `keywords` (TF-IDF
against the other topics — the terms that *separate* this topic), `entities`
with their PageRank, `chunk_ids`, `lead_chunk_id` (member chunk nearest the
centroid), and `coherence`.

Persistence writes one edge: `ENTITY-[:IN_TOPIC]->TOPIC` carrying `pagerank`,
the cluster membership itself — the `entities` field is the actual output of the
clustering, not a byproduct. `modularity` on every `TOPIC` row is the build's
real Newman modularity (`newman_modularity`), shared across all topics from the
same build.

The centroid embedding and the `CHUNK-[:HAS_TOPIC]->TOPIC` evidence edge are
**not** persisted. Both existed only for `TopicChannel`, and that channel is
gone (see below), which made them write-only. The centroid is still computed in
the builder — `coherence` is the mean cosine of members to it — it is simply not
stored.

Global metrics — the numbers a sweep compares:

| metric | meaning |
|---|---|
| `newman_modularity` | partition quality. **The objective.** |
| `mean_coherence` | mean cosine of member chunks to their topic centroid |
| `chunk_coverage` / `entity_coverage` | fraction of the corpus that landed in a topic |
| `topics`, `mean_size`, `largest_topic`, `singleton_topics` | shape of the partition |
| `arcadedb_modularity_untrustworthy` | reported only — see above |

A real sweep on conv-26 shows the tension these expose:

```
  min_size      topics   mean size     largest    newman Q   coherence     ent cov
         2          50       20.96          46      0.1963      0.8016       0.999
         3          41       25.12          46      0.1804      0.8556      0.9819
         5          39       26.23          46      0.1719      0.8765      0.9752
        10          36       27.83          46      0.1743      0.8751      0.9552
```

Modularity prefers `min_size=2`; coherence prefers `5`. Pick against the metric
that matches your use, rather than by eye.

## How topics reach retrieval

**They do not.** `TopicChannel` — which vector-matched the query against topic
centroids, traversed `CHUNK-[:HAS_TOPIC]->TOPIC` to their evidence chunks, and
injected one summary as a synthetic passage — has been removed. Nothing on the
read path consumes what this stage writes.

The layer is therefore reporting and analysis only: `Memory.ltm_topics` walks
`ENTITY-[:IN_TOPIC]->TOPIC` to show what the corpus clustered into, and
`newman_modularity` grades the partition for a sweep. Retrieval is unaffected by
whether topics are built.

Re-adding a read path means re-adding the centroid embedding and the evidence
edge alongside it — and measuring the channel before trusting it, because a
topic is coarse (on conv-26 the mean topic spanned 17 of 63 chunks), so a broad
topic matching the query returns most of the conversation. The saturation guard
that made the old channel survivable dropped any topic covering more than half a
session's chunks.

## Modes

`GRAPHKNOWS_TOPICS` (or `--mode`) selects how summaries are written:

- `extractive` — title from PageRank, summary led by the centroid-nearest member
  chunk. Fully deterministic and network-free, so it is valid under `llm_free`.
- `llm` — prose summaries. Degrades to the extractive summary on any provider
  failure; topic summarisation must never fail a flush.
- `none` — the layer is not built. The default for both presets.

## The JSON artifact

`--export` writes topics, metrics and the config that produced them. It is
**read-only by intent** — there is no import path, so a hand-edited file can
never silently overwrite a computed state. Re-run the stage to change what is
stored. Embeddings are omitted unless explicitly requested, so the file stays
readable and diffable across runs; `config_fingerprint` identifies the
parameter set that produced it.
