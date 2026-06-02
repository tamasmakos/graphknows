# Graphknows MCP Evaluation

This folder contains a lightweight evaluation harness for measuring graphknows MCP tools through the real `graphknows-mcp` stdio contract.

## Native Evaluation

Start the graphknows infrastructure first:

```bash
make infra-up
```

Run the native MCP evaluation:

```bash
python -m evaluation.native_harness --config evaluation/config.example.yaml
```

Reports are written to `evaluation/results/native/` as JSON and Markdown. The harness creates scoped sessions named `eval-<timestamp>-<case>` and only purges those scoped sessions during cleanup.

## Metrics

The report includes operational metrics:

- success rate and returned-error rate
- p50 and p95 latency per tool
- cold-start latency for the first health call
- throughput in calls per second

It also includes quality metrics for recall-like scenarios:

- expected text found
- expected entity found
- recall@k
- precision@k
- answer-support hit rate

## Optional Benchmarks

`longmemeval_adapter.py` provides a LongMemEval-V2-style memory backend with `insert()` and `query()` methods backed by graphknows MCP.

`memoryarena_adapter.py` contains domain scoring helpers and a limited dataset loader for MemoryArena. Install the optional eval dependency group before using external benchmark adapters:

```bash
uv sync --group eval
```

To score a MemoryArena domain while exercising graphknows recall/store calls, pass a predictions JSON file shaped as a list of task prediction lists:

```bash
python -m evaluation.memoryarena_adapter --domain bundled_shopping --limit 2 --predictions-json predictions.json
```

Large benchmark data and outputs are intentionally ignored by git under:

- `evaluation/results/`
- `evaluation/data/`
- `evaluation/external/`
