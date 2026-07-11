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

### Namespaces and the reuse loop

Each **mode** runs in its own physical database pair — namespace
`eval_locomo_llm_free` / `eval_locomo_llm_assisted` — so the two modes coexist in
one ArcadeDB server without contaminating each other's graph (ingesting one mode
never merges entities into the other). Because data persists in the namespace and
session ids are deterministic, `--no-ingest` re-runs retrieval + scoring against
the already-ingested graph (the "just tweak retrieval" loop, no reingest cost).
`--reset` drops the mode's namespace for a clean re-ingest (`--compare --reset`
drops both). See [docs/multi-tenancy.md](../docs/multi-tenancy.md).

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

* Each mode runs in its own namespace (database pair), so llm_free and
  llm_assisted never contaminate each other and data persists between runs
  (reuse with `--no-ingest`, reset with `--reset`). Within a namespace, ingest is
  idempotent per unit (deterministic session id + chunk-id MERGE).
* Runs checkpoint every 30 cases and after every unit — a crash never loses
  completed work. Credit exhaustion (OpenRouter 402) stops the run cleanly and
  keeps partial results.
* Ingestion is sequential by default (`EVAL_INGEST_CONCURRENCY=1`); concurrent
  ingest races the `ENTITY(name)` UNIQUE MERGE and degrades retrieval quality.
