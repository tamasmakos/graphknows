# Graphknows Evaluation

> **Internal benchmarking — not part of the installed `graphknows` package.**
> This directory is excluded from the wheel and sdist and depends on the `eval`
> dependency group (`datasets`, `pandas`, `tabulate`). The memory side is
> configured with `GRAPHKNOWS_*` variables (see [docs/configuration.md](../docs/configuration.md));
> the SemanticF1 scorer and answer generator additionally read `OPENROUTER_API_KEY`.
> The LoCoMo dataset is **not redistributed** here — regenerate it with the
> converter below and check the upstream license for redistribution terms.

A LoCoMo evaluation framework with one job: measure graphknows ingestion and
retrieval through a DSPy answer-generation pipeline, scored with **DSPy
SemanticF1** as the single headline metric and lexical token-F1 as a cheap
diagnostic secondary column.

There is **one workflow**, not a flag matrix. The pipeline ingests each dialogue
once, then for every question: retrieve → DSPy generate → SemanticF1 (via
OpenRouter) + token-F1.

## Run

```bash
# Single-mode run (default: llm_free)
python -m evaluation --limit 152
python -m evaluation --limit 10 --offset 0 --mode llm_assisted

# Systematic comparison: run both modes, get a per-category SemanticF1 table
python -m evaluation --limit 152 --compare
```

`--limit` / `--offset` / `--mode` are the basic knobs. `--compare` runs
*both* `llm_free` and `llm_assisted` over the same questions and writes a
side-by-side per-category SemanticF1 table — the primary systematic-review
surface. Everything else — the MCP command, retrieval depth, recall scope,
relation-extraction mode, generation model, promote-to-LTM, purge behaviour —
lives in [`config.py`](config.py) as a single frozen `EvalConfig`.

Requires `OPENROUTER_API_KEY` (SemanticF1 scoring and generation both call
OpenRouter) and a running graphknows MCP server.

## Scoring surface

| Column | Type | Role |
|--------|------|------|
| `semantic_f1` | float 0-1 | **Headline** — DSPy LLM-estimated F1 |
| `token_f1` | float 0-1 | Diagnostic — SQuAD lexical F1 (no LLM) |

Per-category breakdown (single_hop / multi_hop / temporal / open_domain /
adversarial) is included in both the Markdown report and the `--compare` table.

## Structure

```
evaluation/
  config.py          EvalConfig — the single hardcoded workflow
  datamodels.py      Pydantic models crossing every stage boundary
  dataset.py         LoCoMo loading + session→document normalization
  pipeline.py        orchestrator: ingest → retrieve → generate → score
  cli.py             python -m evaluation  (--limit / --offset / --mode / --compare)
  client/            MCP stdio client
  ingestion/         Ingestor — memory_ingest + memory_flush
  retrieval/         Retriever — memory_query → RetrievedPassage
  generation/        DSPy signature + AnswerGenerator
  scoring/           SemanticF1 (dspy_metrics.py) + token-F1 / retrieval diagnostics (metrics.py)
  reporting/         JSON + CSV + Markdown writers; compare_modes() for --compare
  data/locomo/       questions.jsonl
  scripts/           convert_locomo.py, retrieval_probe.py
  results/           run artifacts (git-ignored)
```

## Data

`data/locomo/questions.jsonl` is produced from the official HuggingFace dataset:

```bash
python -m evaluation.scripts.convert_locomo \
  --output evaluation/data/locomo/questions.jsonl
```

Each row is one question:

```json
{
  "id": "conv-26-q0001",
  "category": "single_hop",
  "question": "Which city did Alex visit after the Berlin conference?",
  "answer": "Prague",
  "sessions": [
    {"session_id": "conv-26-s1", "messages": [
      {"role": "Alex", "content": "Flying to Berlin.", "timestamp": "1 May 2023"},
      {"role": "Alex", "content": "Then two days in Prague."}
    ]}
  ]
}
```

## Diagnostics

`retrieval_probe.py` classifies retrieval failures (OK / RANKING /
VOCAB_MISMATCH / MISSING / EMBED_FAIL) for one dialogue:

```bash
python -m evaluation.scripts.retrieval_probe \
  --dialogue conv-26 --out evaluation/results/locomo/probe-conv-26.json
```

## Outputs

Each run writes `{run_id}-locomo.{json,csv,md}` into `config.output_dir`
(`evaluation/results/locomo/`, git-ignored). The CSV is one flat row per case
for statistical analysis. `--compare` additionally writes `compare-{ts}.md`
with the side-by-side per-category table.
