# Graphknows Evaluation

This package is the production evaluation surface for graphknows. It is organized around one official entrypoint, `run_eval.py`, with benchmark-specific modules underneath it.

## Entry Points

Run the unified CLI with either:

```bash
python -m evaluation.run_eval --help
```

or:

```bash
python -m evaluation --help
```

## Supported Evaluation Suites

- `native`: direct MCP contract and health validation
- `memoryarena`: memory retrieval slices and optional prediction-backed scoring
- `longmemeval`: official LongMemEval-V2 haystack retrieval slices with capsule ingestion support
- `hotpotqa`: support-retrieval evaluation over HotpotQA distractor contexts
- `research-memory`: deterministic retrieval-only experiments with per-case isolation,
  taxonomy injection, mandatory LTM promotion, mandatory topic summarization, and
  three headline scores

## Examples

Start the graphknows infrastructure first:

```bash
make infra-up
```

Native MCP harness:

```bash
python -m evaluation.run_eval native --config evaluation/config.example.yaml
```

MemoryArena retrieval slice:

```bash
python -m evaluation.run_eval memoryarena \
  --domain bundled_shopping \
  --offset 0 \
  --limit 2 \
  --subtask-index 1 \
  --relation-extraction-mode gliner2
```

LongMemEval capsule slice:

```bash
python -m evaluation.run_eval longmemeval \
  --question-id 00aa905a \
  --max-questions 1 \
  --max-trajectories 25 \
  --max-states 4 \
  --max-chars-per-trajectory 4000 \
  --ingest-strategy batched_capsules \
  --relation-extraction-mode gliner2
```

HotpotQA support retrieval slice:

```bash
python -m evaluation.run_eval hotpotqa \
  --offset 0 \
  --limit 3 \
  --max-docs 10 \
  --relation-extraction-mode gliner2
```

Research-grade retrieval-only memory experiment:

```bash
python -m evaluation.run_eval research-memory \
  --suite hotpotqa \
  --offset 0 \
  --limit 3 \
  --taxonomy-file taxonomy.generic.yaml \
  --relation-extraction-mode gliner2
```

The research-memory command does not use an agentic answering workflow. Each
case is purged before and after measurement, ingested through `corpus_ingest`,
promoted to LTM with topic summarization, and scored on:

- `answer_support_score`
- `evidence_recall_score`
- `topic_entity_score`

Any benchmark slice can also capture the live STM footprint before cleanup:

```bash
python -m evaluation.run_eval hotpotqa \
  --offset 0 \
  --limit 1 \
  --relation-extraction-mode gliner2 \
  --inspect-stm
```

When inspection is enabled, each case artifact includes:

- per-ingest `chunks` and `entities` counts returned by `memory_ingest`
- Memgraph STM node counts by label
- pgvector `stm_chunks` / `stm_turns` counts
- distinct STM entity node count
- STM relation edge count
- sampled entities and relations

Prediction-backed MemoryArena scoring:

```bash
python -m evaluation.run_eval memoryarena \
  --domain bundled_shopping \
  --limit 2 \
  --predictions-json predictions.json \
  --relation-extraction-mode gliner2
```

## Outputs

Each run writes JSON and Markdown artifacts into the benchmark-specific output directory. Native runs report tool-level latency and MCP quality metrics. Benchmark runs report per-case memory context, benchmark metadata, and aggregated retrieval metrics.

Large benchmark data and outputs are intentionally ignored by git under:

- `evaluation/results/`
- `evaluation/data/`
- `evaluation/external/`
