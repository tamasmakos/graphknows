# Graphknows Evaluation

> **Internal benchmarking — not part of the installed `graphknows` package.**
> This directory is excluded from the wheel and sdist and depends on the `eval`
> dependency group (`datasets`, `pandas`, `tabulate`). The memory side is
> configured with `GRAPHKNOWS_*` variables (see [docs/configuration.md](../docs/configuration.md));
> answer generation and LLM judging additionally read `OPENROUTER_API_KEY` and
> `LLM_MODEL` (optional `JUDGE_MODEL` overrides the judge). Datasets are **not
> redistributed** here — they are downloaded/converted on first use; check each
> upstream license for redistribution terms.

The LoCoMo memory benchmark over one shared pipeline
(**ingest → retrieve → generate → LLM-judge**):

| Benchmark | What it measures | Dataset |
|---|---|---|
| `locomo` | Long-term conversational QA over 10 multi-session two-person dialogues (single-hop, multi-hop, temporal, open-ended) | converted JSONL (see below) |

## Run

```bash
# inside the workspace container
python -m evaluation --limit 20 --mode llm_free
python -m evaluation --limit 20 --mode llm_assisted
python -m evaluation --limit 20 --compare          # llm_free vs llm_assisted table
python -m evaluation --limit 20 --no-ingest        # reuse the graph, tweak retrieval only
python -m evaluation --reset                       # drop this mode's namespace, then run
```

### Namespaces, isolation, and the reuse loop

Every **conversation** runs in its own physical database pair —
`eval_locomo_<mode>_<conv_id>` (e.g. `eval_locomo_llm_free_conv_26`). This is the
LoCoMo protocol: each question is tied to one conversation and memory is built
per conversation, so conversations must not share a graph. Physical separation
guarantees it — entities, topics, PageRank and communities never merge across
conversations, and the two modes never contaminate each other either.

Because each conversation is isolated, ingest and answer are **decoupled** and
each is bounded by the resource it actually stresses:

- **Ingest is a bounded queue.** `EVAL_INGEST_WORKERS` (default 3) — how many
  conversations run the embedding + NER + vector-index build *at once*. This is
  the memory knob: each concurrent ingest costs ~2 GB RSS in the MCP-server
  process, so an unbounded fan-out OOMs the box. Raise it only as far as your RAM
  allows.
- **Answering is parallel.** `EVAL_QUESTION_CONCURRENCY` (default 8) — a *global*
  cap on total in-flight questions across every conversation. Answering is
  read-only + a remote LLM call (no local models), so it's cheap; this knob
  tracks the LLM provider's rate limit, not memory. A conversation answers as
  soon as it finishes ingest, while later conversations are still queued to
  ingest.
- `EVAL_INGEST_CONCURRENCY` (default 1) — documents *within* one conversation
  stay sequential; they share that conversation's graph (the `ENTITY(name)` MERGE
  would race otherwise).

Data persists per conversation and session ids are deterministic, so
`--no-ingest` re-runs retrieval + scoring against the already-ingested graphs
(the "just tweak retrieval" loop, no reingest cost). `--reset` drops every
per-conversation namespace in the run window for a clean re-ingest
(`--compare --reset` does both modes). See
[docs/multi-tenancy.md](../docs/multi-tenancy.md).

## Metrics

The **headline metric is accuracy under an LLM judge** — the fraction of cases
scoring ≥ 0.5. The judge is a **`dspy.Predict`** over a signature whose inputs are
`(question, gold, generated)` and whose output is a `0.0–1.0` correctness score,
fired with the same model used everywhere (`LLM_MODEL`, or `JUDGE_MODEL` if set)
through OpenRouter and called concurrently via DSPy's async `acall`. Grading is
lenient (partial credit on lists, paraphrase, superset, date-format tolerance).
Lexical `locomo_f1` / `token_f1` (official, paper-comparable) are still computed
as free diagnostic columns; they punish verbose-but-correct answers and are
**not** the headline.

Judge verdicts are cached on disk (`evaluation/results/.judge_cache/`) keyed by
(question, gold, answer, prompt-version), so unchanged answers contribute
exactly 0 noise to A/B deltas. Disable with `GRAPHKNOWS_JUDGE_CACHE=0`.

## Triplet audit — graph quality without retrieval

`evidence_recall` cannot see graph-quality work. A run that added an entire
symbolic plane (1240 frame instances gaining polarity/modality/epistemic, 2735
REL edges gaining a stable predicate id) reproduced
`evidence_recall_at_probe=0.9759` / `evidence_recall_in_context=0.9255` —
identical to four decimals to two prior runs. The system is generation-bound, so
retrieval metrics are saturated and blind to precision. The audit is the
instrument that does not go through retrieval at all:

```bash
# 1. draw a stratified sheet of REL edges (deterministic given --seed)
python -m evaluation audit sample --namespace eval_locomo_llm_free_conv_30 -n 150 --seed 0

# 2. a human fills judge_span / judge_predicate / judge_direction /
#    judge_polarity_modality with 1 or 0 (blank = not judged), then
python -m evaluation audit report --sheet evaluation/results/audit/<sheet>.csv
```

Evidence is joined, not assumed. A reified REL edge is an *entity-adjacency
projection* of a frame instance and carries no `evidence` of its own — 2579 of
2735 edges on conv-30 look evidence-less until you take the hop
`FRAME_INSTANCE -[FRAME_EVOKED_IN]-> CHUNK`, which the sampler does. Only edges
where both routes come up empty are marked unjudgeable and counted in the
report's `no evidence` column.

`verifier_score` coverage is **partial by construction**: `RelationVerifierGate`
runs only in the relex write path, so the frame projection (~90% of edges) is
never scored. The report names the covered and blind tiers in its own caveats so
no mean is mistaken for a whole-graph health number.

Stratified by `canonical_predicate` tier — `fn:` (frame projection), `wn30:`
(WordNet; **no channel emits these yet**, so the tier reports zero rather than
disappearing), an ontology property label/IRI, and `lemma:` (unanchored). Each
sheet row ships with its evidence chunk text, so judging needs no queries.
`report` writes per-tier precision on each axis with **Wilson 95% intervals**,
plus mean/median `verifier_score` and its NULL count per tier — the continuous
monitor between audits, since that score is computed at flush and otherwise read
by nothing.

150 edges over four tiers is ~37 per tier, where the interval is roughly ±15pp
near p=0.5. The report says so in its own output; read the cells as directions,
not measurements, until n is much larger.

### Baseline

`evaluation/audit/baseline/` holds a committed conv-30 baseline
(`conv-30-seed0.csv`, `conv-30-seed0.json`, `conv-30-seed0-report.{json,md}`),
deliberately outside the gitignored `evaluation/results/` tree so it survives a
clone and later audits have something to compare against. It was drawn with
`--namespace eval_locomo_llm_free_conv_30 -n 150 --seed 0`; a comparable later
audit must use the same namespace, size and seed.

The four `judge_*` columns ship blank — the sample is the baseline, not a
judged result — so every precision cell in the committed report is an em dash
until a human completes the judging pass. See `conv-30-seed0-report.md` for
the current population/quota/`no evidence`/`verifier_score` figures rather
than this paragraph, so redrawing the baseline can't leave stale numbers here
disagreeing with the regenerated report.

Pass `--baseline` to `report` to diff against it instead of eyeballing two
Markdown tables:

```bash
python -m evaluation audit report --sheet <sheet>.csv \
    --baseline evaluation/audit/baseline/conv-30-seed0-report.json
```

This adds a `## Change from baseline` section giving per-tier, per-axis
precision change in points, with a marker on any change whose Wilson interval
overlaps the baseline's — so the "audit up &ge;10 points on two tiers" half of
spec #135's abandonment criterion is readable from the report instead of by
eye. (The other half, `evidence_recall` flat, comes from the RAG eval
pipeline, not this report.) The committed baseline's judgement columns are
blank until a human completes a judging pass, so every delta against it reads
as an em dash until then.

## Generation pipeline

Answer generation is a plain chat completion (httpx → OpenRouter, temp 0):
retrieved passages are deduped, **date-extracted, and presented chronologically
with no ranks or scores** (score anchoring biases the answerer; temporal
questions need the timeline). The reasoning prompt lives in `locomo/prompts.py`;
the final answer is extracted after the `ANSWER:` marker. Retrieval diagnostics
(`gold_context_coverage`, reachability, hit counts) separate retrieval misses
from generation failures per case.

## Layout

```
evaluation/
├── __main__.py       # dispatcher: python -m evaluation [flags]
├── common/           # shared: datamodels, ChatLLM, memories, judge cache,
│                     # lexical metrics, MCP ingestion/retrieval, pipeline, reporting
├── locomo/           # config, dataset, prompts, adapter, cli
├── audit/            # stratified triplet audit: core (pure) + cli (db, csv, report)
├── audit/baseline/   # committed conv-30 baseline sheet + report
├── data/locomo/      # dataset (gitignored)
├── results/locomo/   # JSON + CSV + Markdown reports per run (gitignored)
├── results/audit/    # audit sheets, manifests and reports (gitignored)
└── scripts/          # convert_locomo.py, retrieval_probe.py
```

The locomo module implements a `BenchmarkAdapter` (answer + judge) plus a
dataset loader that emits `IngestUnit`s; everything else — MCP lifecycle,
ingestion, retrieval, checkpointing, purging, reporting — is shared in
`evaluation/common/pipeline.py`.

## Datasets

* **locomo** — regenerate `evaluation/data/locomo/questions.jsonl` with
  `python -m evaluation.scripts.convert_locomo` (adversarial category excluded
  by default).

## Operational notes

* Each conversation runs in its own namespace (database pair), so conversations
  and modes never contaminate each other and data persists between runs (reuse
  with `--no-ingest`, reset with `--reset`). Within a namespace, ingest is
  idempotent per unit (deterministic session id + chunk-id MERGE).
* Ingest is a bounded queue (`EVAL_INGEST_WORKERS`, memory-bound) and answering
  is globally parallel (`EVAL_QUESTION_CONCURRENCY`, rate-limit-bound) — physical
  per-conversation isolation is what makes this safe. Lower both to 1 for a
  strictly serial run.
* Runs checkpoint every 30 cases and after every unit — a crash never loses
  completed work. Credit exhaustion (OpenRouter 402) stops the run cleanly and
  keeps partial results.
* Ingestion is sequential by default (`EVAL_INGEST_CONCURRENCY=1`); concurrent
  ingest races the `ENTITY(name)` UNIQUE MERGE and degrades retrieval quality.
