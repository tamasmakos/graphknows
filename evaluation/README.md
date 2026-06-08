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
  dynamic schema extraction, mandatory LTM promotion, mandatory topic summarization, and
  three headline scores

## Examples

Start the graphknows infrastructure first:

```bash
make infra-up
```

Native MCP harness:

```bash
# Inside the workspace container or local venv:
python -m evaluation.run_eval native --command "graphknows-mcp"

# From the host machine (recommended for singleton reuse):
python -m evaluation.run_eval native --command "docker exec -i graphknows-mcp graphknows-mcp"
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
  --relation-extraction-mode gliner2
```

The research-memory command does not use an agentic answering workflow. Each
case is purged before and after measurement, ingested through `corpus_ingest`,
promoted to LTM with topic summarization, and scored on:

- `answer_support_score`
- `evidence_recall_score`
- `topic_entity_score`

## The Singleton MCP

To ensure consistency and avoid container multiplicity, the stack is configured to run a single `graphknows-mcp` instance. When running evaluations from your host machine, always connect to the existing container using `docker exec`:

```bash
# Recommended for host-side evaluation (singleton reuse)
python -m evaluation.run_eval native --command "docker exec -i graphknows-mcp graphknows-mcp"
```

If you must use `docker run`, the evaluation client will automatically inject the `--rm` flag to prevent container leakage, but `exec` is preferred to maintain a single hot instance.

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
