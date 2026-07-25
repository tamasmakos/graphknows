# Handoff — LongMemEval benchmark: wiring + full-capacity feasibility

**Date:** 2026-07-25
**Repo:** `C:\Users\User\Documents\dev\graphknows` (branch `main`)
**Runtime:** everything runs via `docker exec graphknows-workspace ...`. ArcadeDB is a
separate container reached at `http://arcadedb:2480` (user `root`, password in
`/app/.env` as `GRAPHKNOWS_ARCADEDB_PASSWORD` — do not echo it). One GPU: Quadro RTX 3000, 6 GB.

---

## What this session did

### 1. Wired LongMemEval to score like LoCoMo (DONE, tested)
- `evaluation/longmem/dataset.py` — added `_evidence_turns()`: extracts turn-level gold from
  each record's per-message `has_answer` flags into `case.extras["evidence"]`. This activates
  the shared `evidence_recall` metric (pipeline reads `extras["evidence"]` generically).
  479/500 questions carry evidence; the ~21 without are mostly the 30 abstention (`*_abs`) qs.
- `evaluation/longmem/adapter.py` — swapped single-stage judge for `judge.judge_two_stage`
  (matches LoCoMo). Headline `score` is now the stage-2 verdict; `semantic_f1` + `token_f1`
  recorded alongside. Bumped `_PROMPT_VERSION` to `longmem-judge-two-stage-v2`.
- `evaluation/scripts/run_overnight_longmem.sh` — new, mirrors the LoCoMo overnight script
  (`--compare` both modes; defaults to 500q). **Note its default assumes feasibility that
  does NOT hold — see feasibility below.**
- `tests/evaluation/test_datasets.py` — added `TestLongMemDataset` (evidence extraction +
  reference date + empty-evidence). **All test suites pass:** `tests/evaluation` +
  `tests/ingestion` = 192 passed; `tests/retrieval` = 96 passed.
- Verified end-to-end on 1 question: accuracy 1.0, evidence_recall 0.5, all new metrics land.

### 2. GPU wiring for GLiNER (DONE, kept — pure benefit)
- `graphknows/ingestion/extraction/entities/gliner_model.py` — added `_resolve_device()`;
  `load_gliner2_model()` now moves the relex model to CUDA when available
  (`GRAPHKNOWS_RELEX_DEVICE` overrides). Confirmed model loads on `cuda:0`.
- Impact: full-capacity ingest dropped **42.7 min → 24.4 min per question** (1.75×), because
  GPU accelerates both the entity-extraction and frame-SRL GLiNER passes.

### 3. `infer=False` plumbing (DONE, but NOT the chosen path — see below)
- Threaded an `infer` flag through the text ingest path (it was only honored on the
  turn/messages path): `graphknows/memory.py`, `graphknows/ingestion/stm/service.py`,
  `graphknows/ingestion/stm/ingest.py` (gates extraction in `_ingest_parsed`), and
  `evaluation/common/ingestion.py` (env knob `EVAL_INFER=0`).
- **Defaults are unchanged / full-capacity:** `infer` defaults True; `EVAL_INFER` only does
  anything if explicitly set. No capability was removed from the default path.
- The user **explicitly rejected vector-only** ("I want every capacity"). Keep this plumbing
  (harmless, defaults off) but do NOT use `EVAL_INFER=0` for the real run.

### Debug instrumentation — REMOVED. `graphknows/server/mcp/tools/stm.py` and
`evaluation/common/ingestion.py` are back to clean production code (verified, grep clean).

---

## Key findings (the important stuff)

### Why LongMemEval is ~88× more ingest work than LoCoMo (structural, not a regression)
- LoCoMo: 10 dialogues, each ingested ONCE, ~154 questions reuse it → **272 total session-ingests**.
- LongMemEval: 500 questions, each its OWN ~48-session haystack → **23,867 total session-ingests**.
- Per-haystack size is comparable (~27 vs ~48 sessions); the blow-up is zero amortization.
- Dedup won't save it: only **1.2×** overlap across the 500 haystacks (19,195 unique sessions).

### Full-capacity feasibility
- **24.4 min/question** (full graph, GPU) × 500 = **~203 h ≈ ~8.5 days SERIAL, one mode.**
- Profile: ingest is ~75% ArcadeDB write round-trips, ~25% GLiNER compute. `write_relations`
  does **one HTTP call per relation** — batching those (UNWIND) is the main remaining code lever.
- **Parallel GPU sharding is BROKEN on this card:** 2 concurrent GLiNER models = 100% GPU util,
  only ~447 MB free, they thrash → docs hang → `TimeoutError` (900s MCP timeout) → half-built
  graphs. So the "shard for 2–3×" plan is off the table without a bigger/second GPU.
- Single serial runs are clean and **crash-safe**: `_checkpoint()` rewrites the full report
  after every question (`evaluation/common/pipeline.py:495`). Resume by re-running remaining
  `--offset` ranges (no built-in skip-done).

### Ingestion output QUALITY (user's last question)
Inspected a clean full-capacity graph (`mem_eval_longmem_llm_free_e47becba`, post-consolidation):
- **Clean:** chunks 289/289 embedded (100%); frames sensible; temporal present; provenance +
  entity-resolution consolidation working (7,872 raw entities → 6,647 merged).
- **Noisy graph layer:** entities 23% untyped, dominated by `ACTIVITY` (1,440 — generic gerunds
  like 'planning','steps'), article-dupes ('farmer'/'The farmer'), lots from distractor sessions.
  Relations (1,271) use generic predicates (USE/FOLLOW/INCORPORATE) and store the **entire chunk
  text** in the `fact` field, with heavy duplication. Communities 262 (noisy). **Topics: 0**
  (topic layer produces nothing — matches the known "topic layer dead" finding).
- Takeaway: the vector layer does the retrieval work (identical accuracy 1.0 / evidence_recall
  0.5 with graph on vs off); the expensive graph layer is largely noise. User was told this.

---

## Open decision (asked, NOT yet answered by user)
The last question put to the user:
- (a) proceed with the **serial full-capacity run as-is** (~8.5 days), or
- (b) first improve graph quality (entity-junk filter / relation `fact` trimming) before
  committing a week of compute.

Also unresolved: how to make ~8.5 days practical at all. Realistic options to raise with user:
batch the ArcadeDB relation writes (biggest single-node lever, keeps full capacity), run a
representative **subset** instead of all 500, get a second/bigger GPU for real sharding, or
accept the serial multi-day run.

## Gotchas learned (avoid re-discovering)
- **Contamination trap:** never launch overlapping eval runs. `kill -9` on an eval parent
  ORPHANS its `graphknows-mcp` child, which keeps writing to the namespace. Always kill by
  matching both `m evaluation` AND `graphknows-mcp` in `/proc/*/cmdline`, verify zero
  survivors, then drop the namespace before a fresh run. Trust the tool's returned
  `entities` / eval's own `ingest done:` line, NOT live DB polls (they catch orphan writes).
- Windows→Docker: prefix `docker exec` cmds with `MSYS_NO_PATHCONV=1` when passing container
  paths like `/tmp/...`, or Git Bash mangles them.
- `container has no `ps`/`free`` — enumerate processes via `/proc/*/cmdline`; GPU via `nvidia-smi`.
- `onnxruntime` in the image is CPU-only (no CUDAExecutionProvider); GLiNER uses torch, which
  has CUDA — that's why the `.to("cuda")` fix works.

## Suggested skills for the next session
- **eval-locomo** — the tuning/verification harness (per-profile SemanticF1, pre-screen,
  keep/revert rule). Directly applicable to running/scoring LongMemEval the same way.
- **diagnose** — if the multi-day run wedges or ArcadeDB OOMs (heap-OOM history), use the
  disciplined repro→instrument→fix loop.
- **caveman** — the user prefers terse output (see their memory: caveman/less-tokens mode).

## Verify state before continuing
```
docker exec graphknows-workspace bash -c 'for p in /proc/[0-9]*; do c=$(tr "\0" " " < $p/cmdline 2>/dev/null); case "$c" in *"m evaluation"*|*graphknows-mcp*) echo RUNNING $c;; esac; done'
git -C C:/Users/User/Documents/dev/graphknows status   # my changes are unstaged, not committed
```
Leftover namespaces from testing exist (e.g. `mem_eval_longmem_llm_free_e47becba`,
`..._0a995998`, `..._gpt4_59149c77`, `mem_mcpchk_*`, `mem_inftchk_*`) — drop them before a
clean run. Nothing has been committed.
