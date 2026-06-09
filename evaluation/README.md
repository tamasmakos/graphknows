# Graphknows Evaluation

This package is the production evaluation surface for graphknows memory systems.
It intentionally exposes only the three benchmarks that define the current
measurement landscape:

- `locomo`: LoCoMo, 1,540 questions across single-hop, multi-hop, open-domain,
  and temporal memory recall.
- `longmemeval`: LongMemEval, 500 questions across single-session user recall,
  single-session assistant recall, single-session preference recall, knowledge
  update, temporal reasoning, and multi-session recall.
- `beam`: BEAM at 1M and 10M token scales across preference following,
  instruction following, information extraction, knowledge update,
  multi-session reasoning, summarization, temporal reasoning, event ordering,
  abstention, and contradiction resolution.

The launcher is designed so these can be run together or separately.

## Entry Points

```bash
python -m evaluation list
python -m evaluation run locomo
python -m evaluation run longmemeval
python -m evaluation run beam
python -m evaluation run all
```

The individual benchmark names are shorthand for `run`:

```bash
python -m evaluation locomo
python -m evaluation longmemeval
python -m evaluation beam
python -m evaluation all
```

Multiple benchmarks can be selected explicitly:

```bash
python -m evaluation run locomo beam
```

## Infrastructure

Start graphknows first:

```bash
make infra-up
```

From the host machine, prefer connecting to the singleton MCP container:

```bash
python -m evaluation run longmemeval \
  --command "docker exec -i graphknows-mcp graphknows-mcp"
```

The default command is `graphknows-mcp`, which is convenient inside a local venv
or workspace container.

## LongMemEval

LongMemEval uses the existing LongMemEval-V2 adapter. Missing official files are
downloaded into `evaluation/data/longmemeval-v2`.

```bash
python -m evaluation longmemeval \
  --longmemeval-max-questions 1 \
  --longmemeval-max-trajectories 25 \
  --longmemeval-max-states 4 \
  --longmemeval-max-chars-per-trajectory 4000 \
  --longmemeval-ingest-strategy batched_capsules
```

## LoCoMo And BEAM Local Data

LoCoMo and BEAM are file-backed adapters so local benchmark exports can be
plugged in without locking this repo to one external distribution format.

Defaults:

- LoCoMo: `evaluation/data/locomo/questions.jsonl`
- BEAM 1M: `evaluation/data/beam/1m/questions.jsonl`
- BEAM 10M: `evaluation/data/beam/10m/questions.jsonl`

Each row may be JSONL or part of a JSON array. The runner accepts these common
fields:

- Identity: `id`, `question_id`, or `case_id`
- Prompt: `question`, `query`, or `prompt`
- Gold answer: `answer`, `gold`, or `target`
- Category: `category`, `type`, `task_type`, or `question_type`
- Memory context: `sessions`, `messages`, `turns`, `conversation`,
  `documents`, `contexts`, `evidence`, `memories`, `haystack`, `context`,
  `memory`, or `transcript`

Example LoCoMo row:

```json
{
  "id": "locomo-dev-001",
  "category": "temporal memory recall",
  "question": "Which city did Alex visit after the Berlin conference?",
  "answer": "Prague",
  "sessions": [
    {
      "session_id": "s1",
      "messages": [
        {"role": "user", "content": "I am flying to Berlin for the conference."},
        {"role": "user", "content": "After Berlin I will spend two days in Prague."}
      ]
    }
  ]
}
```

Example BEAM commands:

```bash
python -m evaluation beam --beam-scale 1m --limit 1
python -m evaluation beam --beam-scale 10m --beam-max-documents 500 --limit 1
```

## Shared Controls

All runnable commands accept:

```bash
--command
--run-id
--output-dir
--top-k
--offset
--limit
--recall-scope stm|ltm|both
--relation-extraction-mode gliner2|llm_assisted|off
--no-purge
```

## Outputs

Each run writes JSON and Markdown artifacts into:

```text
evaluation/results/<benchmark>/
```

Large benchmark data and outputs are intentionally ignored by git under:

- `evaluation/results/`
- `evaluation/data/`
- `evaluation/external/`
