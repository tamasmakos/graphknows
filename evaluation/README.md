# Graphknows Evaluation

> **Internal benchmarking — not part of the installed `graphknows` package.**
> This directory is excluded from the wheel and sdist and depends on the `eval`
> dependency group (`datasets`, `pandas`, `tabulate`). The memory side is
> configured with `GRAPHKNOWS_*` variables (see [docs/configuration.md](../docs/configuration.md));
> answer generation and LLM judging additionally read `OPENROUTER_API_KEY` and
> `LLM_MODEL` (optional `JUDGE_MODEL` overrides the judge). Datasets are **not
> redistributed** here — they are downloaded/converted on first use; check each
> upstream license for redistribution terms.

Three memory benchmarks over one shared pipeline
(**ingest → retrieve → generate → LLM-judge**):

| Benchmark | What it measures | Dataset |
|---|---|---|
| `locomo` | Long-term conversational QA over 10 multi-session two-person dialogues (single-hop, multi-hop, temporal, open-ended) | converted JSONL (see below) |
| `longmemeval` | Assistant memory over per-question haystacks: temporal reasoning, knowledge updates, multi-session synthesis, abstention | auto-download (500 q) |
| `beam` | Very-long-conversation abilities (10 types incl. contradiction resolution, event ordering, instruction/preference following) at 100K–10M tokens | auto-download via HF `datasets` |

## Run

```bash
# inside the workspace container
python -m evaluation locomo --limit 20 --mode llm_free
python -m evaluation locomo --limit 20 --compare          # llm_free vs llm_assisted table
python -m evaluation longmemeval --per-type 5             # 30-question stratified sample
python -m evaluation longmemeval --all-questions
python -m evaluation beam --chat-sizes 100K --conversations 0-4
```

Bare flags (`python -m evaluation --limit 5`) run locomo for backward
compatibility.

## Metrics

The **headline metric is accuracy under an LLM judge** — the fraction of cases
scoring ≥ 0.5:

* `locomo` — binary CORRECT/WRONG judge with lenient grading (partial credit on
  lists, paraphrase, superset, ±14-day date tolerance). Lexical `locomo_f1` /
  `token_f1` (official, paper-comparable) are still computed as free diagnostic
  columns; they punish verbose-but-correct answers and are **not** the headline.
* `longmemeval` — binary yes/no judge (semantic equivalence with an explicit
  anti-"no" bias check; abstention gold matches any refusal).
* `beam` — per-rubric-nugget judge (0 / 0.5 / 1, question score = nugget mean);
  `event_ordering` blends in a Kendall tau-b ordering term.

Judge verdicts are cached on disk (`evaluation/results/.judge_cache/`) keyed by
(question, gold, answer, prompt-version), so unchanged answers contribute
exactly 0 noise to A/B deltas. Disable with `GRAPHKNOWS_JUDGE_CACHE=0`.

## Generation pipeline

Answer generation is a plain chat completion (httpx → OpenRouter, temp 0):
retrieved passages are deduped, **date-extracted, and presented chronologically
with no ranks or scores** (score anchoring biases the answerer; temporal
questions need the timeline). Each benchmark has its own reasoning prompt in
`<benchmark>/prompts.py`; the final answer is extracted after the `ANSWER:`
marker. Retrieval diagnostics (`gold_context_coverage`, reachability, hit
counts) separate retrieval misses from generation failures per case.

## Layout

```
evaluation/
├── __main__.py       # dispatcher: python -m evaluation <benchmark>
├── common/           # shared: datamodels, ChatLLM, memories, judge cache,
│                     # lexical metrics, MCP ingestion/retrieval, pipeline, reporting
├── locomo/           # config, dataset, prompts, adapter, cli
├── longmemeval/      # config, dataset (auto-download), prompts, adapter, cli
├── beam/             # config, dataset (HF datasets), prompts, adapter, cli
├── data/<benchmark>/ # datasets (gitignored)
├── results/<benchmark>/  # JSON + CSV + Markdown reports per run (gitignored)
└── scripts/          # convert_locomo.py, retrieval_probe.py
```

Each benchmark module implements a `BenchmarkAdapter` (answer + judge) plus a
dataset loader that emits `IngestUnit`s; everything else — MCP lifecycle,
ingestion, retrieval, checkpointing, purging, reporting — is shared in
`evaluation/common/pipeline.py`.

## Datasets

* **locomo** — regenerate `evaluation/data/locomo/questions.jsonl` with
  `python -m evaluation.scripts.convert_locomo` (adversarial category excluded
  by default).
* **longmemeval** — `longmemeval_s_cleaned.json` downloads automatically on
  first run into `evaluation/data/longmemeval/`.
* **beam** — splits download via the `datasets` package on first run and cache
  as JSON into `evaluation/data/beam/` (the 10M bucket is its own dataset).

## Operational notes

* Each ingest unit gets a fresh session; sessions are purged after their
  questions run, and all LTM is wiped before a run (no cross-run contamination).
* Runs checkpoint every 30 cases and after every unit — a crash never loses
  completed work. Credit exhaustion (OpenRouter 402) stops the run cleanly and
  keeps partial results.
* Ingestion is sequential by default (`EVAL_INGEST_CONCURRENCY=1`); concurrent
  ingest races the `ENTITY(name)` UNIQUE MERGE and degrades retrieval quality.
* `longmemeval` ingests one haystack per question and `beam` one 100K–10M-token
  conversation per unit — widen `--per-type` / `--conversations` gradually.
