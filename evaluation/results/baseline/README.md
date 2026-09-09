# Baseline (FR-041, SC-003)

What the **unmodified core** scores, one file per panel row plus `internal.json` for the
once-per-run internal metrics (FR-040). Every later change is judged against these numbers
by `panel --compare`, so the files are committed — the only thing under
`evaluation/results/` that is.

Each metric carries the contract's `median` / `band` / `runs` / `commit`
([contracts/panel-report.md](../../../.claude/specs/004-neurosymbolic-memory-core/contracts/panel-report.md)),
recorded over three runs against `commit`.

## How it is recorded

```bash
python -m evaluation baseline --rows locomo,longmem,beam --repeats 3
```

## Current state

Every row and every internal metric is recorded `not_run`, with the reason in the file: the
recording run above has not been executed against a provisioned environment yet. A `not_run`
record is the contract's way of writing down a number that was not produced — it is never
omitted and never estimated — and the merge gate does not gate a `not_run` row. Until the
run happens, the panel gates nothing.
