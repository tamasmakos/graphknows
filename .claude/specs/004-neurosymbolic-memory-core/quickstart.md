# Quickstart — validating the slice

**Feature**: Universal Neurosymbolic Memory Core — First Vertical Slice

One runnable scenario per user story, in the order the phases ship. Each scenario is the
acceptance for its phase: it fails against the tip before the phase exists and passes after
(Constitution I). Implementation detail lives in [contracts/](./contracts/) and
[data-model.md](./data-model.md), not here.

---

## Prerequisites

```bash
make setup          # creates .env
make up             # ArcadeDB + workspace
make shell          # everything below runs inside the workspace container
```

Panel datasets are provisioned by the image and the setup script, never by a code path under
test (Constitution III):

```bash
python -m evaluation download --rows longmem,beam
```

A missing dataset is an **environment** defect. Fix it here; do not let a test fetch it.

Sanity check before anything else:

```bash
make gate           # ruff, mypy, lint-imports, bandit, unit suite, coverage floor
```

---

## Phase A — Baseline (US1)

Runs against the **unmodified** tree. No new core code may exist when this is recorded.

```bash
git rev-parse HEAD                                        # record this sha in the baseline
python -m evaluation baseline --rows locomo,longmem,beam --repeats 3
```

**Expect**: `evaluation/results/baseline/` gains one file per row and one for the internal
metrics, each with `median`, `band`, `runs` and `commit` ([contracts/panel-report.md](./contracts/panel-report.md)).

**Passes when**:

- Every row reports accuracy, evidence recall and a per-category breakdown, or an explicit
  `not_run` with a reason (SC-003).
- Every row whose data has turns reports `feeding_mode: turn_by_turn` — LongMemEval and BEAM
  currently feed a session as one blob, so this is a real change, not a re-run.
- `identity.pairwise_precision` and `pairwise_recall` are numbers, from a hand-labelled
  sample of at least 200 clusters that is not LoCoMo.
- `dead_weight.passed` is reported. Against the old core it is expected **false** — that is
  the audit's four write-only planes, measured rather than asserted.

---

## Phase B — A core that does not know the domain (US2)

```bash
pytest tests/packs tests/models tests/storage -q
lint-imports
```

Then the scenario itself — two unrelated packs, one namespace:

```bash
python -m evaluation scenario two-packs --namespace scratch
```

**Passes when**:

- Both packs load, both sources ingest, both symbols recall facts **with evidence** (SC-002).
- `lint-imports` proves the `core-knows-no-domain` contract: no core module imports
  `graphknows.packs` (SC-001, FR-003).
- The grep test finds no benchmark, dialogue, speaker or turn-regex term in core modules
  (SC-001).
- Re-running the same ingest reports `sources_deduplicated` and writes nothing (SC-008).
- A query resolving to nothing returns `no_evidence: true`, not an empty list (FR-015).
- Opening a namespace stamped by the old schema fails immediately with a version mismatch
  (SC-013):

  ```bash
  python -m evaluation scenario schema-refusal      # exits non-zero, names both versions
  ```

- Two concurrent ingests into one namespace produce the same entity and fact set as running
  them in sequence, and neither fails (SC-015):

  ```bash
  python -m evaluation scenario concurrent-ingest --namespace scratch
  ```

## Phase B′ — Cutover

One change. Old core deleted, its tests deleted with it, dropped planes no longer written or
declared.

```bash
make gate
python -m evaluation panel --compare evaluation/results/baseline
python -m evaluation deadweight --namespace <run-namespace>
```

**Passes when**: the panel holds against the Phase A baseline (SC-004) **and**
`deadweight` exits zero — every declared type that was written has a reader (SC-005). The
dead-weight check going from false in Phase A to true here is the measurable result of the
cutover.

---

## Phase C — Identity that does not guess (US3)

```bash
pytest tests/ingestion/consolidation -q
python -m evaluation identity --sample evaluation/data/identity/labelled.jsonl
```

**Passes when**:

- Pairwise precision and recall are reported as numbers and precision is at or above the
  baseline median.
- Every merge is reconstructible and undoable from its `MERGED_INTO` log alone (SC-007):

  ```bash
  python -m evaluation scenario merge-replay --namespace scratch
  ```

- Resolution time per batch does not grow linearly with namespace size over a tenfold
  increase (SC-006):

  ```bash
  python -m evaluation scenario resolve-scaling --sizes 10000,100000
  ```

  Report the per-batch wall clock at both sizes. A near-flat curve is the pass; a tenfold
  rise means a scan survived somewhere.

---

## Phase D — The repository and its transcripts become memory (US4)

```bash
python -m graphknows.cli.memory ingest --namespace repo --uri file://$(pwd)/graphknows
python -m graphknows.cli.memory ingest --namespace repo --uri file://<a transcript>.jsonl
python -m graphknows.cli.memory recall --namespace repo --query "schema_version"
python -m graphknows.cli.memory recall --namespace repo --query "<a decision stated in that transcript>"
```

**Passes when**:

- The function recall returns facts whose evidence resolves to the right file and byte range
  — open it and check the range actually contains the definition.
- The decision recall returns facts citing the right transcript record.
- Re-ingesting the transcript adds nothing; the report shows `records_skipped_duplicate`
  (SC-008).
- A deliberately drifted record is counted, not fatal (SC-014):

  ```bash
  python -m evaluation scenario transcript-drift
  ```

- Neither ingest required a core change — the diff for this phase touches
  `graphknows/packs/` and `graphknows/ingestion/parsers/` only.

---

## Phase E — Recall arrives in the agent's loop (US5)

Install the settings block from
[contracts/agent-hooks.md](./contracts/agent-hooks.md), then:

```bash
pytest tests/integrations/claude_code -q

echo '{"prompt":"zzz nothing here resolves zzz"}' \
  | python -m graphknows.integrations.claude_code.hooks recall
```

**Passes when**:

- The unresolvable prompt prints **nothing** (SC-009, FR-031).
- A prompt naming a known symbol prints an `additionalContext` block of facts, newest first.
- The hook process loads no ML dependency — the subprocess test asserts `torch`,
  `transformers`, `spacy`, `sentence_transformers` and `gliner` are absent from
  `sys.modules`, and `lint-imports` holds the stdlib-only contract (SC-010).
- Recall p95 is under 1 s and the hard ceiling is 5 s: `python -m evaluation scenario hook-latency`.
- A forgotten fact stops appearing, while its tombstone and merge log stay readable
  (SC-012):

  ```bash
  python -m evaluation scenario forget-roundtrip --namespace repo
  ```

- Running a real session in this repository ends with its own records ingested.

---

## Phase F — A yardstick that includes this repository's own work (US6)

```bash
python -m evaluation panel --rows repo
python -m evaluation panel --compare evaluation/results/baseline
```

**Passes when**:

- The `repo` row runs end to end and reports the same metric shape as every other row, with
  at least three question kinds represented (SC-011).
- Local and LLM extraction appear as separate columns (FR-043).
- The comparison exits zero: no row below `median − band`, identity precision not dropped,
  dead weight clean (SC-004).

---

## The merge gate, in one command

Everything above collapses to this before a merge:

```bash
make gate \
  && python -m evaluation panel --compare evaluation/results/baseline \
  && python -m evaluation deadweight --namespace <run-namespace>
```

A green `make gate` alone is **not** the acceptance for this feature. The panel is
([contracts/panel-report.md](./contracts/panel-report.md)), and no single row is the
headline.
