# Phase 0 — Research

**Feature**: Universal Neurosymbolic Memory Core — First Vertical Slice
**Date**: 2026-09-08
**Status**: all open questions resolved. No `NEEDS CLARIFICATION` remains.

This run was unattended, so every question below was decided here rather than deferred.
Each decision names what would overturn it, because a decision taken without the owner in
the room should be cheap to reverse.

The spec's own `## Clarifications` settled five questions already (FR-013 one anchor per
segment, FR-014 episodes unmaterialised, FR-043 both extractors first-class, FR-047 ingest
serialised per namespace, FR-037 forget as tombstone). Those are inputs here, not
re-litigated.

---

## R1 — Code parsing: how does a source file become segments and facts?

**Decision**: the stdlib `ast` module. One segment per module-level `FunctionDef`,
`AsyncFunctionDef` and `ClassDef` (plus nested definitions, qualified by dotted path), with
`path` = qualified symbol and byte offsets computed from `lineno`/`col_offset` /
`end_lineno`/`end_col_offset` against the file's decoded lines. Facts come from the same
walk: `imports` from `Import`/`ImportFrom`, `calls` from `Call` nodes resolved to the
nearest enclosing definition, `defines` from the containment tree, `tests` from a test file
naming convention plus the symbols a test body calls.

**Rationale**: the corpus this slice must prove itself on is *this repository*, which is
Python. `ast` is exact where a grammar is approximate, ships with the interpreter, and
gives end positions natively since 3.8 — the repo requires 3.11. FR-030 demands facts
without a machine-learning model, and this is the shortest thing that satisfies it.
Principle IV stops the ladder here: the stdlib does it. Principle VII agrees — no new
third-party import, no new wheel, no grammar bundle to declare to the build backend.

**Alternatives considered**:

- *tree-sitter* (what `docs/generalisation-audit.md` §3.1 proposes). Buys other languages,
  which no requirement in this slice asks for, at the cost of a new dependency plus
  per-language grammar packages that are platform wheels. Deferred, not rejected: the
  `Parser` protocol is what makes the swap a pack change rather than a core change.
- *A regex or indentation scanner*. Cheaper to write, wrong on decorators, multi-line
  signatures and nested classes — and byte offsets that are subtly wrong poison every
  evidence claim downstream (SC-002).
- *Reusing the repository's existing `graphify-out/` graph*. It is a build artefact of a
  different tool with its own lifecycle; depending on it makes ingest non-reproducible from
  sources.

**Overturned if**: a non-Python language pack enters scope. The upgrade is a second
`Parser` implementation, recorded as a ceiling in plan.md.

---

## R2 — Agent transcript format: what exactly is parsed?

**Decision**: newline-delimited JSON, one record per line, read with stdlib `json`.
Sessions live one file per session under the harness's per-project transcript directory
(this repository's own runs are visible at
`~/.claude/projects/<path-slug>/<session-uuid>.jsonl`). The parser keeps records whose
`type` is `user` or `assistant` and whose `isMeta` is falsy, walks `message.content` blocks
by their `type` (`text`, `thinking`, `tool_use`, `tool_result`), and maps block type to
segment kind: `text`/`thinking` → `turn`, `tool_use` → `tool_call`, `tool_result` → **not a
segment** (FR-028; recorded as a `Source` a fact may cite). Dedup key is the record `uuid`.
Retained as segment metadata: `timestamp`, `cwd`, `gitBranch`, `isSidechain`.

**Rationale**: this is the format the integration must actually read, and it is documented
by its vendor as internal and version-dependent — which is precisely why FR-027 demands a
tolerant, versioned parser rather than a schema-validating one. Stdlib `json` per line
means a malformed trailing record (a transcript still being written, per the spec's edge
cases) costs one counted skip, not an aborted ingest.

**Alternatives considered**:

- *Validate each record against a pydantic model*. Turns format drift into a hard failure,
  which FR-027 forbids outright.
- *Parse the whole file with `json.load` after wrapping in brackets*. A single bad line
  loses the entire session, and a file being appended to is never valid.
- *Ingest via the harness's own hook payloads only, no file parsing*. Loses backfill of
  existing transcripts, which US6's panel row is built from.

**Counters this creates** (Principle V): `records_skipped_unknown_type`,
`blocks_skipped_unknown_type`, `records_skipped_malformed`, `records_skipped_duplicate`.
All four surface on `IngestReport`.

---

## R3 — Identity candidate generation: what replaces the full-table scan?

**Decision**: two-stage, both inside the database.

1. **Blocking.** `ENTITY.block_key`, a stored string property with a plain index. Key is
   `f"{name_norm[:4]}|{dominant_type}"` where `name_norm` is the existing normalisation and
   `dominant_type` is the mode of the type histogram. Candidates are fetched by equality on
   the key, bounded by an explicit `candidate_limit`; truncation increments
   `candidates_truncated` and is reported (the spec's "very large candidate set" edge case).
2. **Vector neighbours.** An `LSM_VECTOR` index over `ENTITY.embedding`, where the
   embedding is of the entity's *definition* (canonical name plus dominant type plus one
   context sentence) — the same "a symbol is indexed by the embedding of its definition"
   rule `graphknows/symbolic/index.py` already implements for concepts. Top-*k* neighbours
   are unioned with the block, then gated by the existing floor-plus-margin acceptance in
   `graphknows/symbolic/match.py`.

**Rationale**: the store already declares `LSM_VECTOR` indexes on `CHUNK`, `FRAME` and
`ONTOLOGY_CLASS`, so this is a proven pattern in this codebase with no new dependency —
Principle IV rung 2, not rung 5. Doing both stages in SQL is what makes SC-006 true by
construction: nothing loads the entity table into Python, so batch cost tracks the block
and the neighbour list, not the namespace.

**Alternatives considered**:

- *faiss / hnswlib in-process*. A new dependency, a second copy of the vectors, and a
  synchronisation problem between it and the graph. Rejected on Principle IV and VII.
- *Blocking alone*. The audit records that the current precision rests on a person-only
  3-char prefix (E7); blocking alone cannot find "one thing under two names", which US3
  explicitly requires.
- *Vector neighbours alone*. E2 records that embedding similarity was measured unsafe on
  nicknames. It is evidence, not a decision — hence the floor-plus-margin gate and the
  candidate edge rather than a merge (FR-018).

**Overturned if**: the labelled sample shows block-key recall below the baseline. The fix
is a second blocking key (phonetic), not a scan.

---

## R4 — Per-namespace ingest serialisation (FR-047)

**Decision**: an `asyncio.Lock` per namespace, held in the memory service process, acquired
around the whole ingest transaction and released before recall paths are touched. Waiting
requests queue on the lock rather than failing; wait time is recorded as
`ingest_queue_wait_ms` on `IngestReport`. Recall takes no lock and stays available
throughout.

**Rationale**: the spec's clarification says "the memory service serialises ingest for a
namespace and queues the rest" — a single service process is the deployment model
(`docker-compose.yaml` runs one), so the in-process lock is the whole mechanism. `asyncio`
is stdlib; the code is roughly a `defaultdict` of locks. SC-015 (two concurrent sessions
produce the same set as running them in sequence) is exactly what a lock guarantees.

**Alternatives considered**:

- *A lease row in ArcadeDB*. Correct across processes, and needs lease expiry, heartbeats
  and crash recovery — a distributed lock nobody asked for. Recorded as the upgrade path in
  plan.md's ceiling table.
- *Optimistic retry on conflict*. Identity resolution is the thing being protected; a retry
  after a half-observed merge does not undo the wrong merge.
- *No lock, resolve idempotently*. Makes every writer's correctness depend on every other
  writer's ordering — the failure mode SC-015 exists to catch.

---

## R5 — Schema version refusal (FR-005, SC-013)

**Decision**: already implemented; carried over unchanged, with its tests.
`graphknows/storage/arcadedb/_schema.py:schema_version(dims)` digests the DDL statement text
plus the embedding dimension; `graph_store._check_schema_stamp` reads the `SCHEMA_STAMP`
singleton on open and raises `SchemaVersionMismatchError` when it differs — and treats an
*unstamped* pre-existing database as a mismatch rather than assuming compatibility.

**Rationale**: Principle IV rung 2 — it is already here, and it already does the whole job
including the pre-existing-unstamped case. Replacing the golden DDL changes the digest,
which strands every old database. That is the intended behaviour, not a regression: FR-045
says no database is migrated.

**Work this slice actually needs**: none in the mechanism. One documentation line, and the
existing test kept green through the DDL replacement (its assertion is that a differing
stamp refuses, which survives the new schema unchanged).

---

## R6 — The dead-weight check (FR-024, FR-040, SC-005)

**Decision**: a runtime written-versus-read diff, run at the end of a panel run.

- **Written** — enumerate the vertex and edge types the golden DDL declares, then
  `SELECT count(*)` each one against the namespace the run used. Non-zero means written.
- **Read** — every store query method declares the types it reads, as a `frozenset`
  attribute on the method's own module-level registry entry, and the store records them in
  a per-connection `read_types` set as queries execute. The panel harness snapshots it.
- **Check** — `written - read` must be empty. Non-empty fails the run and names the types.

**Rationale**: the audit's four write-only planes all survived review, CI and a benchmark;
the only thing that catches that class of defect is an observation of the run itself. A
runtime set is honest where static analysis is not — SQL is built as strings, so an AST
scan of the store would both miss types and invent them. The declaration-per-query-method
is a few lines beside code that already exists.

**Alternatives considered**:

- *Static grep of type names across `graphknows/`*. A type named in a comment or a
  write path counts as a reader. The audit's own evidence shows this is the exact mistake:
  the WordNet reader was deleted in `9cec590` and the write path stayed, and grep still hit.
- *Coverage of the store's reader methods*. Measures whether the method ran, not whether
  the type has any reader at all.

**Consequence for FR-014**: this check is why episodes are not materialised. A stored
`EPISODE` plane with no traversal reading it would fail its own panel run on day one.

---

## R7 — Run noise and the merge gate (FR-042, SC-004)

**Decision**: the Phase A baseline runs each panel row **three times**. The recorded noise
band per metric is the observed range (max − min) of those three runs, stored in the
baseline file beside the median, which is the headline value. A later run regresses when it
falls below `median − band` on any row. Identity pairwise precision is compared against the
baseline median with no band — SC-004 says it must not drop.

**Rationale**: SC-004 says "beyond its recorded run noise", so the noise must be a recorded
measurement, not a tolerance someone picked. Three runs is the smallest *n* that yields a
range at all; the LLM-judge path in `evaluation/common/dspy_judge.py` is the dominant
variance source and is expensive, which is what keeps *n* small. Median over mean because
three samples with one outlier is exactly the case the mean handles badly.

**Alternatives considered**:

- *One run plus a fixed tolerance* (e.g. 2%). The number is invented, which is the thing
  Principle VI exists to stop.
- *Bootstrap confidence intervals over cases within a single run*. Measures case sampling
  noise, not the run-to-run variance introduced by the judge and by non-deterministic
  extraction — the wrong variance.
- *Five or more runs*. Better statistics, and the baseline is a blocking prerequisite for
  every other phase. Three is the cost that does not stall the slice.

---

## R8 — Panel data availability (FR-039, FR-041)

**Decision**: LoCoMo data is in the tree and its results are committed. LongMemEval and
BEAM datasets are fetched by `evaluation/common/download.py`, invoked from the setup script
and baked into the evaluation image — never from library code and never lazily from a test.
Turn-feeding is added to the LongMemEval and BEAM adapters (both currently feed a session as
one blob, per `evaluation/longmem/dataset.py` and `evaluation/beam/dataset.py`), and the
report states the feeding mode per row, per US1 acceptance scenario 2.

**Rationale**: Constitution III, verbatim — a missing corpus is an environment defect and is
fixed in the image or the setup script. A baseline row that cannot be provisioned is
recorded as *not run*, with the reason; it is never silently omitted and never estimated.

**Alternatives considered**:

- *Skip the rows without committed data*. FR-041 requires a baseline for every panel row,
  and the audit's headline finding is that the "did terribly" LongMemEval result is
  currently unreproducible — leaving it unmeasured preserves exactly that.
- *A lazy download inside the adapter*. Prohibited by Principle III and by the offline
  guarantee.

---

## R9 — Counters: what must be named rather than silent (Principle V, FR-027, edge cases)

**Decision**: one counter set, carried on `IngestReport` and `RecallResult`, never logged
and dropped. Enumerated so the requirement is checkable rather than aspirational:

*Ingest*: `sources_deduplicated`, `segments_written`, `files_zero_segments`,
`records_skipped_unknown_type`, `blocks_skipped_unknown_type`, `records_skipped_malformed`,
`records_skipped_duplicate`, `time_anchor_inferred`, `candidates_truncated`,
`merges_committed`, `candidate_edges_written`, `merges_vetoed`, `ingest_queue_wait_ms`.

*Recall*: `symbols_resolved`, `symbols_unresolved`, `facts_returned`,
`facts_truncated_by_budget`, `facts_excluded_tombstoned`, `pool_below_floor`.

**Rationale**: the recurring defect this repository names for itself is "code that runs,
passes, and does nothing". A counter that reaches the caller is the difference between a
channel that is working and one that degrades quietly. Every spec edge case maps to exactly
one counter above, which is how the edge-case list becomes testable.

---

## R10 — Pack conflict detection (FR-025)

**Decision**: at load, the loader builds one dict keyed by concept URI and one by predicate
id, recording the owning pack. A second pack claiming a key already owned raises
`PackConflictError` naming both packs and the key. Loading is all-or-nothing: a conflict
leaves no pack loaded.

**Rationale**: FR-025 says detected at load, not resolved by last-writer-wins, and
Principle V says a lookup must raise or count. Two dicts and a raise is the whole
implementation — Principle IV rung 6.

**Alternatives considered**:

- *Namespace every pack's identifiers by pack name*. Makes conflicts impossible and also
  makes two packs unable to agree on a shared concept, which is the point of aligning to a
  standard schema. Rejected.
- *Warn and take the first*. Silent enough to be indistinguishable from success.

---

## R11 — Forget as tombstone: where is the exclusion enforced (FR-037, SC-012)?

**Decision**: a `state` property on `ENTITY` and `FACT` set to `forgotten`, and a **single**
SQL predicate fragment in `graphknows/storage/arcadedb/_sql.py` that every read query
composes. Enforcement is that the fragment exists in one place, plus one test that asserts
every store read method's query text contains it.

**Rationale**: the honest failure mode for a tombstone is one retrieval path that forgot the
filter, and SC-012 says *no* subsequent recall. One choke point plus a test over the method
registry is smaller than auditing each query, and it fails loudly when someone adds a
twelfth read path.

**Alternatives considered**:

- *Physical delete*. Explicitly out of scope, and it destroys the merge log FR-017 requires
  to stay replayable.
- *A filtering wrapper around results in Python*. Moves the filter after the budget, so a
  forgotten fact can consume a budget slot and silently shrink the answer.

---

## R12 — Superseding on functional predicates (FR-012)

**Decision**: on writing a fact whose predicate carries `functional = true`, the writer
looks up the current fact for the same `(subject, predicate)` pair. If the incoming fact's
segment observation time is newer, the existing fact keeps its row with `is_current` set
false and a `SUPERSEDES` edge is written from new to old. If it is older, the incoming fact
is written with `is_current` false and the edge points the other way. Equal times leave both
current and write a `CONTRADICTS` edge — the honest state, not an arbitrary winner.

**Rationale**: FR-013 fixes the time model to one observation time per segment, so "newer"
is unambiguous and needs no per-fact anchor. Retaining the old row is what makes
knowledge-update questions answerable at all; deleting it would be the same
information-destroying move as the current delete-merge.

**Alternatives considered**:

- *Overwrite the object in place*. Loses the history a temporal panel row is scored on.
- *Break ties by confidence*. Manufactures a decision from two numbers produced by
  different extractor runs; `CONTRADICTS` says what is actually true.

---

## R13 — Hook budget and dependency isolation (FR-036, SC-010)

**Decision**: target p95 under 1 s for a recall hook, hard ceiling 5 s, enforced by the hook
process's own timeout. Isolation is asserted structurally, not by inspection: a test imports
`graphknows.integrations.claude_code.hooks` in a subprocess and asserts that `torch`,
`transformers`, `spacy`, `sentence_transformers` and `gliner` are absent from
`sys.modules`; `.importlinter`'s `client-stdlib-only` contract gains the hook package as a
second source module.

**Rationale**: the harness's own hook timeout default is far larger than anything useful —
a hook that takes ten seconds is a hook the user turns off, so the budget is a product
constraint, not a platform one. The `sys.modules` assertion is the check that actually
catches the regression, because an accidental ML import arrives through a transitive
`from graphknows.something import X` that no reviewer sees.

**Alternatives considered**:

- *Trust the import contract alone*. `lint-imports` catches a `graphknows` import, not a
  direct `import torch` in a new hook file.
- *Measure cold start in CI*. Wall-clock in CI is too noisy to gate on; the dependency
  assertion is deterministic and is the real cause of a slow start.

---

## R14 — Which existing modules survive the cutover (FR-045)

**Decision**, following the spec's Assumptions, which names the sound modules. Recorded here
so the cutover change has an inventory rather than a judgement call per file.

**Carried over unchanged (copy, no edit)**: `ingestion/extraction/llm/` (decoder, anchor,
response gates, retry, schema), `temporal.py`, `ranking/` (`rrf.py`, `frame_boost.py`),
`symbolic/ontology/` and the semantic-web stack, `symbolic/index.py`, `symbolic/match.py`,
the memory service and `integrations/client/`, `evaluation/common/`,
`storage/arcadedb/client.py` and `_base.py`, the `SCHEMA_STAMP` mechanism, `storage/embedder.py`,
`storage/namespace.py`, `settings.py`, `llm.py`.

**Rewritten**: `memory.py`, `retrieval/retriever.py`,
`ingestion/consolidation/entity_resolution.py`, `channels/registry.py`,
`storage/arcadedb/_schema.py`, and `graph_store.py` split by responsibility.

**Deleted**: `topics/`, `cli/topics.py`, the topics MCP tools, `ingestion/stm/`, the
`Role` enum in `models/message.py`, the FE/SEMTYPE graph mirror, the `FRAME` versus
`FRAME_INSTANCE` duplication, free-string relations, `SESSION`/`TURN` vertex types,
`ENTITY.pagerank` / `community_id` / `graph_embedding`, the WordNet write path (unless a
pack ships a `collect()` that reads it), `_SURFACE_ALIASES`, `REL.reified`, `REL.anchor`.

**Rationale**: FR-046 lists the planes that must stop being written *and declared*, and the
dead-weight check (R6) is what proves the list is complete rather than remembered. Each
deletion travels with the tests that import it, under the Principle II rule recorded in
plan.md.

**Undecided by design**: `bounds.py`, `worth.py`, `nlp.py`, `linguistics.py`. Each is
audited against the dead-weight check during Phase B′ and either carried or deleted with a
named reason. Guessing now would be the "delete adjacent code" mistake Principle IV forbids.
