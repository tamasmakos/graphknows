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

Recorded 2026-09-09 against commit `2d50eba` (the last commit before any new core code),
local extractor, `--limit 10 --repeats 3`, repeats two and three reusing the first repeat's
ingest:

| Row | State | accuracy median (band) | evidence recall median (band) | n |
|---|---|---|---|---|
| locomo | recorded | 0.9 (0.1) | 0.6 (0.0) | 10 |
| longmem | not_run | — | — | — |
| beam | not_run | — | — | — |
| internal | not_run | — | — | — |

The first LoCoMo repeat ingested 19 documents in 8656 s (7.6 min per document, CPU only).
At that rate a limit-10 LongMemEval row is roughly 500 documents and a BEAM row hundreds,
so both are recorded `not_run` with that reason rather than estimated, and so are the
internal metrics. A `not_run` record is never omitted and never gated (FR-042); the LoCoMo
row is the only gated row until the others are recorded.
