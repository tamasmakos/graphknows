# Graphknows Evaluation

A LoCoMo evaluation framework with one job: measure graphknows ingestion and
retrieval through a DSPy answer-generation pipeline, scored with paper-comparable
token F1 (arXiv 2402.17753).

There is **one workflow**, not a flag matrix. The pipeline ingests each dialogue
once, then for every question: retrieve → DSPy generate → score token F1 vs gold.

## Run

```bash
python -m evaluation --limit 152          # run 152 questions
python -m evaluation --limit 10 --offset 0
```

`--limit` / `--offset` are the only flags. Everything else — the MCP command,
retrieval depth, recall scope, relation-extraction mode, generation model,
promote-to-LTM, purge behaviour — lives in [`config.py`](config.py) as a single
frozen `EvalConfig`. Edit that file to change the workflow.

Requires `OPENROUTER_API_KEY` (generation is always on) and a running graphknows
MCP server.

## Structure

```
evaluation/
  config.py          EvalConfig — the single hardcoded workflow
  datamodels.py      Pydantic models crossing every stage boundary
  dataset.py         LoCoMo loading + session→document normalization
  pipeline.py        orchestrator: ingest → retrieve → generate → score
  cli.py             python -m evaluation  (--limit / --offset)
  client/            MCP stdio client
  ingestion/         Ingestor — memory_ingest + memory_flush
  retrieval/         Retriever — memory_query → RetrievedPassage
  generation/        DSPy signature + AnswerGenerator
  scoring/           token-F1 + retrieval diagnostics
  reporting/         JSON + CSV + Markdown writers
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
(`evaluation/results/locomo/`, git-ignored). The CSV is one flat row per case for
statistical analysis.
