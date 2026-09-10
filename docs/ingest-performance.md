# Ingest performance: what is actually slow, and what is not running at all

Date: 2026-09-10. Branch `004-neurosymbolic-memory-core` @ `aa4206a`. Method: six parallel
investigations (end-to-end profile, local extraction, language-model extraction, container and
GPU environment, storage writes, orchestration), each measuring rather than reading. Every
number below was taken in the workspace container against the live ArcadeDB.

## 1. The headline

Two different problems got tangled together under "ingest is slow".

**Ingest is slower than it should be**, but not catastrophically, and not where the previous
diagnosis said. Warm, per segment: **0.49 s on the local path, 2.05 s on the language-model
path.** Cost is linear in segment count; there is no blow-up.

**Ingest is also not doing most of the work it claims**, which is why it is cheaper than
expected and why the results look thin. Three capabilities are built, unit-tested, and never
connected to the path that would use them. Making the current path faster would be optimising
a pipeline that does not extract relations and does not resolve identity.

Fix the wiring first. Otherwise the speed work is measured against a pipeline doing a fraction
of its job, and every number has to be retaken the moment the wiring lands.

## 2. Issue #337 is obsolete

It was measured before the cutover, on the old core. Re-tested item by item:

| #337 claim | Now |
|---|---|
| Stale CPU-only torch image; GPU reserved but unused | **Fixed.** Container rebuilt 4 Sep: torch 2.12.0+cu126, `cuda.is_available()` true, Quadro RTX visible. Compose documents why 12.6 not 12.8 |
| Three relex forward passes per chunk | **Wrong now.** One pass. Two only with deixis *and* a relation vocabulary. The old sub-windowing was removed |
| DeBERTa verifier recomputing embeddings per relation | **Never loads.** It short-circuits on an empty relation list |
| GLiNER2 frame role filling, one schema pass per trigger | **Gone.** `frame_srl.py` no longer exists |
| Prompt of ~39 KB / 10k tokens, 1437 class + 264 relation labels | **~2.6 KB, 700-800 tokens.** Partly earlier legitimate work, partly defect (see 3.1) |
| 541 s model warmup | Still real, still unpaid by warmup: spaCy 3.75 s, embedder 3.8-4.0 s, language-model client 7.15 s |
| 4-hour eval request timeout hides slowness as a hang | **Unchanged.** `evaluation/common/config.py:30` |

A GPU is present and working, but it is **not obviously a win**: a benchmark of the embedding
model was *slower* on GPU (0.56 s) than CPU (0.22 s), because kernel launch overhead dominates
a small model on a small batch. Nobody benchmarked the large relation-extraction model both
ways. **That is the single highest-value unmeasured question**, and it is cheap to settle.

## 3. Built, tested, never wired

The real finding. Three instances of one pattern, each filed.

### 3.1 Neither decoder passes relation labels ([#413](https://github.com/tamasmakos/graphknows/issues/413))

`_DecoderExtractor._decode` (`ingestion/extraction/protocol.py:132-138`) and
`LLMExtractor._decode` (`:203-210`) both forward `entity_labels` and never
`extra_relation_labels` or `frame_candidates`. So `relation_spec` is always `{}`:

- the local path always takes the entity-only branch (`entities/extractor.py:1013`);
- the language-model path has both prompt sections deleted (`llm/decoder.py:143-149`);
- the verifier filters an empty list, having loaded for nothing.

**Neither path can produce a fact.** A pack without its own extractor yields mentions only, and
since the atom of recall is a fact with evidence, recall over that content returns nothing. The
two shipped packs mask this: both carry deterministic extractors. The dialogue pack, the next
spec, would land straight on this path.

### 3.2 Identity resolution is never called ([#415](https://github.com/tamasmakos/graphknows/issues/415))

Nothing in `graphknows/` imports `ingestion/consolidation/`. The only importer in the repository
is `evaluation/scenarios.py`. A live ingest writes entities on a normalised-name upsert and
never merges, never proposes a candidate, never writes the merge log.

Both identity scenarios say in their own docstrings that they need no namespace: `merge-replay`
runs over three hand-built objects, `resolve-scaling` times a pure function over 10,000
synthetic dicts. **The 421.7 ms / 438.0 ms scaling result measures an algorithm, not the graph.**
It shows blocking keys bound the work, which is worth having, and it is not evidence about
ingest.

### 3.3 Batching exists and nothing calls it

`GLiNER2EntityExtractor.extract_batch` batches spaCy through `nlp.pipe()`.
`LLMDecoder.extract_batch` fans out over a thread pool sized by `decode_concurrency` (default
8). `IngestPipeline.ingest` calls neither — it loops segments and calls `extract()` one at a
time (`pipeline.py:125-136`). **`decode_concurrency` is dead configuration.**

## 4. Measured cost

Warm, DDL excluded, per segment: **local 0.49 s, language-model 2.05 s.** Linear in segment
count. Declining per-segment cost at larger N is fixed per-call overhead amortising, not
economies inside the loop.

Ranked sinks (warm steady state):

| Rank | Sink | Cost | Where |
|---|---|---|---|
| 1 | Language-model call, serial, blocking | 1.85-2.05 s/segment | `llm/decoder.py:419` |
| 2 | Relex forward pass, serial, global lock | 0.29-0.32 s/segment | `entities/gliner_model.py:38` |
| 3 | Per-mention embedding, one call per segment | 0.05-0.25 s/segment | `pipeline.py:39` via `:160` |
| 4 | Sequential ArcadeDB round trips | 0.04-0.06 s/segment | `arcadedb/client.py:191` |
| 5 | Cold model load, one-time | spaCy 3.75 s, embedder 3.9 s, LM client 7.15 s | `nlp.py:30`, `embedder.py:37` |

**One disagreement, stated rather than smoothed:** the storage investigation measured a
50-segment ingest at 3.36 s wall with **76.6% in database round trips** (276 calls, 5.52 per
segment), which ranks the store far above position 4. The two runs used different inputs and
neither repeated. Settle it with a repeated measurement on one agreed input before trusting
either ranking.

**Everything expensive runs on the event loop.** `Extractor.extract` is synchronous, `_embed_texts`
calls the embedder directly, and there is no `asyncio.to_thread` anywhere between `Memory.ingest`
and a model call. `nlp.py`'s docstring claims ingest workers reach it via `to_thread`; that claim
is stale. Database writes are async but issued one at a time, never gathered.

## 5. Ideas, ranked

### Free, no behaviour change

1. **Instrument first.** `IngestReport`/`Counters` carry no timings. Add per-stage milliseconds
   and one log line per document. Slowness currently presents as a hang, and the next
   optimisation has nothing to measure against.
2. **Warm what ingest uses.** `stdio_server._startup_warmup` touches settings and the embedder
   only, not spaCy, relex or the language-model client. Up to ~100 s per process, one-time.
3. **Drop the unused spaCy lemmatizer** via `exclude=["lemmatizer"]`. Nothing reads `.lemma_` on
   this pipeline.
4. **Cut the 4-hour eval timeout** so slow stops looking like hung.
5. **Surface abstentions.** The decoder counts them; `memory.py:451` hardcodes `{}`.

### Cheap, small risk

6. **Call `extract_batch`.** Both already exist. Replaces N spaCy calls with one `nlp.pipe()`,
   and on the language-model path replaces N serial network calls with a bounded fan-out —
   roughly 7x wall clock at the default concurrency, identical requests, same cost.
7. **Offload model work with `to_thread`** so a decode stops blocking the event loop.
8. **Embed mentions once per source**, not once per segment.
9. **Batch the schema DDL.** `sqlscript` was tested live and works (`cypherscript` does not
   exist). 108 round trips and ~2.6 s per cold connect collapse to a handful.
10. **Remove the read-before-write** in `EntityWriter._histogram`: merge the histogram
    server-side, saving one round trip per mention.

### Structural, real risk, needs tests first

11. **Batch the per-segment write loops** with `asyncio.gather`, keeping `_settle_currency`
    ordered per functional predicate.
12. **Rewrite the hot writers in SQL** and batch each segment's writes into one script: about
    half the round trips, but upsert semantics differ and a wrong one silently duplicates
    vertices.
13. **Narrow the per-namespace lock** to writes and resolution, leaving embed and extract
    outside it. Largest structural win, and it needs the concurrent-ingest test that FR-047
    implies and nobody wrote.
14. **Pipeline across documents** once the lock is narrowed.
15. **Reconsider the model.** A 467M-parameter joint entity-and-relation model is doing
    entity-only work. Either wire relations (3.1) or use a smaller entity model.

## 6. Measure these next

- The relation-extraction model on GPU against CPU, batched and unbatched. Settles whether the
  GPU matters at all.
- One agreed input, repeated runs, to reconcile the two cost rankings in section 4.
- Segment length varied independently of count. Only count was varied.
- Segment-repetition rate on a real corpus, to size a text-keyed extraction cache. Note the
  language-model cache is off deliberately: disk caching would put ingested text outside the
  configured database. Respect that constraint rather than flipping the flag.
