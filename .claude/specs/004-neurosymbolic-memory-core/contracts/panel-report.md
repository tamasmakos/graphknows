# Contract — Panel report and baseline

**Surface**: `python -m evaluation …`, writing under `evaluation/results/`. The panel is the
merge gate for every change in this slice (FR-042), so its file shape is a contract, not an
output format.

---

## One harness, all rows (FR-039)

Rows: `locomo`, `longmem`, `beam`, `repo` (new — FR-044). All four run through
`evaluation/common/pipeline.py`. Rows whose data has turns are fed **turn by turn**, not as
one blob per session, and the report states the feeding mode it used per row.

Datasets are provisioned by `evaluation/common/download.py` from the setup script and the
image — never lazily from library or test code (Constitution III). A row that cannot be
provisioned is recorded as `not_run` with the reason. It is never omitted and never
estimated.

---

## Report shape

Per run, per row:

```json
{
  "row": "locomo",
  "feeding_mode": "turn_by_turn",
  "extractor": "local",
  "accuracy": 0.0,
  "evidence_recall": 0.0,
  "by_category": {"...": 0.0},
  "n": 0,
  "not_run": null
}
```

`extractor` is `local` or `llm`. Both paths appear as **separate columns** in every run
(FR-043); neither is deferred and both are gated.

Per run, once — the internal metrics (FR-040):

```json
{
  "identity": {"pairwise_precision": 0.0, "pairwise_recall": 0.0, "n_clusters": 0},
  "fact_precision": {"...": 0.0},
  "dead_weight": {"unread_types": [], "passed": true}
}
```

- **Identity** — pairwise precision and recall against a hand-labelled sample of at least
  200 clusters, on a corpus that is not LoCoMo. `evaluation/identity.py`.
- **Fact precision** — the existing triplet audit (`evaluation/audit/`), generalised beyond
  dialogue.
- **Dead weight** — `written − read` over the run's namespace. A non-empty `unread_types`
  sets `passed: false` and **fails the run** (FR-024, SC-005). This is a run outcome, not an
  advisory line in a report.

---

## Baseline (FR-041, SC-003)

Written to `evaluation/results/baseline/`, committed, **before the first line of new core
code exists**. This is the gate that makes US1 block every other story.

Each row and each internal metric runs **three times**. Recorded per metric:

```json
{"median": 0.0, "band": 0.0, "runs": [0.0, 0.0, 0.0], "commit": "<baseline sha>"}
```

`median` is the headline; `band` is the observed range (max − min) and *is* the run noise
(research.md R7). Three runs is the smallest *n* that yields a range at all, and the
LLM-judge path is the dominant variance source and is expensive.

---

## Merge gate (FR-042, SC-004)

A change merges only when **all** of the following hold:

1. No panel row falls below `median − band` on accuracy or evidence recall. A row
   introduced after the baseline is gated against its own first recorded run, named as
   such (FR-041); a row that cannot run yet reports `not_run` and is not gated (FR-042).
2. Identity pairwise precision is greater than or equal to the baseline median. No band —
   SC-004 says it must not drop.
3. `dead_weight.passed` is true.

And two rules about how the result is read:

- **No single row is the headline** (FR-042). A row that moves alone moves alone.
- A drop confined to the dialogue row is accepted where the panel as a whole holds (SC-004,
  and the spec's own Assumptions: a dialogue drop is an accepted cost).

---

## The repository row (FR-044, SC-011)

`evaluation/repo/`, built from this repository's own agent transcripts, with gold answers
derived from git history and the transcripts themselves. At least three question kinds:

| Kind | Gold source |
|---|---|
| Decision location | where a decision was stated, in a transcript record |
| Change rationale | why a line changed, from the commit that changed it |
| Test coverage of a symbol | which test guards a given function |

Same metric shape as every other row (SC-011). It is one row among four and is not the
headline either.

## Verify

```bash
python -m evaluation baseline --rows locomo,longmem,beam --repeats 3   # Phase A, once
python -m evaluation panel --compare evaluation/results/baseline       # exits non-zero on regression
python -m evaluation deadweight --namespace <run-namespace>
```
