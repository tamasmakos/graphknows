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

Because each conversation is isolated, they ingest and answer **concurrently**
(no shared `ENTITY(name)` MERGE to race):

- `EVAL_UNIT_CONCURRENCY` (default 3) — conversations processed in parallel.
- `EVAL_QUESTION_CONCURRENCY` (default 5) — questions answered in parallel within
  a conversation (retrieve→generate→judge is read-only, no shared writes).
- `EVAL_INGEST_CONCURRENCY` (default 1) — documents within one conversation stay
  sequential; they share that conversation's graph.

Data persists per conversation and session ids are deterministic, so
`--no-ingest` re-runs retrieval + scoring against the already-ingested graphs
(the "just tweak retrieval" loop, no reingest cost). `--reset` drops every
per-conversation namespace in the run window for a clean re-ingest
(`--compare --reset` does both modes). See
[docs/multi-tenancy.md](../docs/multi-tenancy.md).

## Metrics

The **headline metric is accuracy under an LLM judge** — the fraction of cases
scoring ≥ 0.5. `locomo` uses a binary CORRECT/WRONG judge with lenient grading
(partial credit on lists, paraphrase, superset, ±14-day date tolerance). Lexical
`locomo_f1` / `token_f1` (official, paper-comparable) are still computed as free
diagnostic columns; they punish verbose-but-correct answers and are **not** the
headline.

Judge verdicts are cached on disk (`evaluation/results/.judge_cache/`) keyed by
(question, gold, answer, prompt-version), so unchanged answers contribute
exactly 0 noise to A/B deltas. Disable with `GRAPHKNOWS_JUDGE_CACHE=0`.

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
├── data/locomo/      # dataset (gitignored)
├── results/locomo/   # JSON + CSV + Markdown reports per run (gitignored)
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
* Conversations run in parallel (`EVAL_UNIT_CONCURRENCY`), and questions within a
  conversation run in parallel (`EVAL_QUESTION_CONCURRENCY`) — physical
  per-conversation isolation is what makes this safe. Raise them to stress-test
  ArcadeDB concurrency; lower to 1 for a strictly serial run.
* Runs checkpoint every 30 cases and after every unit — a crash never loses
  completed work. Credit exhaustion (OpenRouter 402) stops the run cleanly and
  keeps partial results.
* Ingestion is sequential by default (`EVAL_INGEST_CONCURRENCY=1`); concurrent
  ingest races the `ENTITY(name)` UNIQUE MERGE and degrades retrieval quality.
