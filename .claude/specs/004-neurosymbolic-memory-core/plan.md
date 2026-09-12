# Implementation Plan: Universal Neurosymbolic Memory Core — First Vertical Slice

**Branch**: `004-neurosymbolic-memory-core` | **Date**: 2026-09-08 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `.claude/specs/004-neurosymbolic-memory-core/spec.md`

## Summary

Make the memory core domain-neutral by construction and prove it on an agent working in a
repository. Three seams get a neutral data model — **Source → Segment** at input,
**Mention / Fact** at extraction, **DomainPack** at domain knowledge — and everything the
core currently knows about dialogue moves out of code and into pack data. Identity
resolution gets a mention layer, blocking, a vector-index candidate step and a replayable
merge log. Two packs ship: a code pack (deterministic, no model) and an agent pack (session, prompt,
decision, file and tool concepts, fed by the transcript parser); recall re-enters the agent's loop through stdlib-only hooks talking to the running
memory service.

Nothing is judged by feel: a baseline is recorded against the current core before the first
line of new core code exists, and every later change is gated on the whole panel plus
identity precision plus a dead-weight check that fails the run when a written type has no
reader. FR-001's representation half — the embedder and the fused candidate set from rank
fusion — is carried over unchanged; only the symbolic half is rebuilt.

The rewrite is in place. The dropped planes (topics, the STM buffer, the frame mirror) are
deleted first, with their tests, so the tree never writes a type the schema no longer
declares; the remaining modules (schema, store, retriever, channel registry, facade) are
then rewritten under their current names; the cutover phase deletes whatever old core is
left and records the interim exact-key-only identity state (FR-048). At no point do two
write paths ship — which is what FR-045's "no shims, no flags, no migration" actually
requires. The tree is not green between the first in-place rewrite and cutover; the gate
sits at cutover.

## Technical Context

**Language/Version**: Python 3.11+ (`requires-python = ">=3.11"`), async throughout the
storage and service paths.

**Primary Dependencies**: no new third-party dependency is introduced by this slice.
Existing and reused: `pydantic` / `pydantic-settings` (models, settings), `numpy`,
`sentence-transformers` + `torch` + `transformers` (embeddings), `gliner` / `gliner2`
(local extraction), `spacy` (linguistics), `rank-bm25` (sparse channel), `nltk`
(WordNet/FrameNet), `dspy` + `litellm` (LLM decoder), `mcp` (server), `httpx` (ArcadeDB
client), `dateparser` (temporal). New code uses stdlib only: `ast` (code parser), `json`
(transcript parser), `asyncio` (per-namespace ingest serialisation), `hashlib` (content
hash, schema digest).

**Storage**: ArcadeDB, one database per namespace, reached over HTTP by
`graphknows.storage.arcadedb`. Golden schema replaces the current DDL wholesale;
`SCHEMA_STAMP` version refusal already exists and is carried over unchanged.

**Testing**: `pytest` (unit + integration, `tests/`), `ruff`, `mypy`, `lint-imports`
(`.importlinter`), `bandit`, a coverage floor — all via `make gate`. Panel measurement is
`evaluation/` (`evaluation/common` harness commons, per-row adapters, `evaluation/audit`
triplet audit).

**Target Platform**: Linux container (`docker-compose.yaml`, `Dockerfile`,
`deploy/Dockerfile`) plus local dev on Windows/macOS. The library also installs standalone
from PyPI; the in-loop hook process must run outside the container with no ML dependency.

**Project Type**: distributed Python library + MCP service + evaluation harness. Modular
monolith, layers enforced by `.importlinter`.

**Performance Goals**: identity resolution per flush must not grow linearly with namespace
entity count across a tenfold size increase (SC-006). Hook recall round trip p95 under 1 s,
hard ceiling 5 s (SC-010). Panel throughput no worse than the recorded baseline's
wall-clock per case.

**Constraints**: no domain-conditional branch anywhere in the core (FR-003, enforced by an
import contract plus a grep test); recall always returns facts with evidence, never bare
segments (FR-009); no migration path (FR-005); no network at runtime on the default path
except to ArcadeDB; `integrations.client` and the new `integrations.claude_code` stay
stdlib-only; every third-party import declared with a lower bound; non-`.py` pack data
declared to the build backend.

**Scale/Scope**: one repository's source (~600 Python files) and its agent transcripts as
the first real corpus; panel rows LoCoMo (~191 cases committed), LongMemEval, BEAM, plus a
new repository-transcript row. Identity ground truth: one hand-labelled sample of at least
200 clusters. Six user stories, of which US1 must complete before any new core code exists.

## Constitution Check

*GATE: checked before Phase 0 research, re-checked after Phase 1 design. Both passes below.*

| Principle | Gate | Pre-Phase-0 | Post-Phase-1 |
|---|---|---|---|
| **I. Acceptance Is a Runnable Command** | Every task carries one runnable `## Verify` that fails at tip and passes after. | PASS — [quickstart.md](./quickstart.md) defines the runnable scenario per user story; `/speckit-tasks` inherits them. | PASS — each contract in `contracts/` names the command that exercises it. |
| **II. Never Weaken a Check** | No test deleted, retargeted, skipped, or loosened to go green. | **Tension, resolved — see note below.** | PASS with the recorded rule. |
| **III. Fix the Defect, Not the Symptom** | Corpora, models, services, credentials fixed in image/compose/setup, never in library code. | PASS — LongMemEval and BEAM datasets are provisioned by `evaluation/common/download.py` from the setup script and the image, never lazily by a code path under test. | PASS — no contract permits a library-side fetch. |
| **IV. Build Only What Was Asked For** | Ladder: needed at all → already here → stdlib → installed dep → one line. | PASS — code parser is stdlib `ast`, not tree-sitter; candidate search is the store's existing `LSM_VECTOR` index, not a new ANN library; schema refusal already exists and is reused verbatim. | PASS — `DomainPack` has two shipped implementations (code pack, agent pack) plus a third specified next slice, so it is not an abstraction with one caller. |
| **V. Errors Never Pass Silently** | Every gate, fallback and lookup raises or increments a named counter surfaced in the result. | PASS — counters enumerated in [research.md](./research.md) §R9 and carried on the ingest/recall result contracts. | PASS — `IngestReport` and `RecallResult` in [data-model.md](./data-model.md) carry the counter set; no path returns a bare empty list. |
| **VI. Evidence Over Vibes** | Change justified by a metric named before the change. | PASS — US1 (baseline) blocks every other story by construction; SC-003 is the gate. | PASS — dead-weight check is a first-class panel output, not a review habit. |
| **VII. A Package Installed by Strangers** | Every third-party import declared with a lower bound; shipped non-`.py` declared; extras enforced. | PASS — this slice adds no third-party import. | PASS — pack data files (`graphknows/packs/data/**/*.json`) must be added to the build backend's package data and asserted by `tests/test_packaging.py`. |

**Principle II note (recorded, not waived).** FR-045 deletes the old core, and tests that
exercise deleted modules go with them. That is not weakening a check: a test removed *in
the same change that removes the code it exercises* asserts nothing about behaviour that
still exists, and `docs/agents/definition-of-done.md` item 6 already says deletion earns no
new tests. The rule this plan binds itself to, so the distinction stays checkable:

- A test may be deleted **only** in the change that deletes the module it imports, and the
  deletion commit must name that module.
- No test of behaviour that survives cutover may be deleted, skipped, retargeted, or have
  an assertion loosened. Behaviour that survives but moves gets its test moved, not rewritten.
- The panel baseline (US1) is the replacement acceptance for the deleted suite as a whole:
  the numbers recorded before cutover are what the new core must still hit.

No entry is needed in Complexity Tracking — no principle is violated.

## Project Structure

### Documentation (this feature)

```text
.claude/specs/004-neurosymbolic-memory-core/
├── plan.md                       # This file
├── research.md                   # Phase 0 output — every open question resolved
├── data-model.md                 # Phase 1 output — entities, fields, transitions
├── quickstart.md                 # Phase 1 output — runnable validation per story
├── contracts/
│   ├── python-api.md             # Memory facade + Parser/Extractor/DomainPack protocols
│   ├── storage-schema.md         # Golden DDL: types, edges, indexes, version stamp
│   ├── mcp-tools.md              # memory_ingest / memory_recall / memory_forget
│   ├── agent-hooks.md            # Hook stdin→stdout JSON contract
│   └── panel-report.md           # Baseline file and per-run report shape
├── checklists/
│   └── requirements.md           # (existing) spec quality checklist
└── tasks.md                      # Phase 2 output — created by /speckit-tasks, not here
```

### Source Code (repository root)

```text
graphknows/
├── exceptions.py                 # + PackConflictError, NoEvidence is a result not an error
├── models/                       # neutral data model (grows; message.py Role enum deleted)
│   ├── source.py                 # NEW  Source
│   ├── segment.py                # NEW  Segment, SegmentKind
│   ├── fact.py                   # NEW  Mention, Fact, Polarity, Modality, Validity
│   ├── symbols.py                # NEW  ConceptRef, PredicateRef, SymbolKind (incl. EPISODE)
│   ├── report.py                 # NEW  IngestReport, RecallResult, Counters
│   ├── hit.py  ingest.py  ingest_options.py  memory_class.py  scope.py   # kept
├── settings.py  llm.py  temporal.py                                      # kept
├── storage/
│   ├── embedder.py  namespace.py                                         # kept
│   └── arcadedb/
│       ├── _schema.py            # REPLACED  golden DDL; stamp mechanism kept verbatim
│       ├── _sql.py               # + the single forgotten-tombstone filter fragment
│       ├── client.py  _base.py                                           # kept
│       └── graph_store.py        # SPLIT     source/segment/fact/entity/symbol writers
├── ranking/                      # kept unchanged (rrf.py, frame_boost.py)
├── symbolic/
│   ├── index.py  match.py                                                # kept
│   ├── predicates.py             # NEW  predicate index (no free-string relations)
│   ├── ontology/                 # kept (loader + semantic-web stack)
│   ├── framenet/                 # REDUCED  becomes a concept+predicate emitter for a pack
│   └── wordnet.py                # deleted at cutover (T037); returns as the dialogue pack's lexicon
├── channels/
│   ├── base.py  registry.py      # REPLACED  registry takes packs, not `if enable_x`
│   └── ontology.py  _shared.py                                           # kept
├── ingestion/
│   ├── parsers/
│   │   ├── registry.py           # NEW  Parser protocol + MIME-keyed registry
│   │   ├── text.py  _sentences.py# kept, dialogue heuristics removed to the dialogue pack
│   │   ├── code.py               # NEW  stdlib `ast`, one segment per function/class
│   │   └── transcript.py         # NEW  agent transcript JSONL, tolerant + versioned
│   ├── extraction/
│   │   ├── protocol.py           # NEW  Extractor protocol; both paths behind it
│   │   ├── llm/                  # kept verbatim (decoder, anchor, gates, retry, schema)
│   │   ├── entities/             # REDUCED  hygiene becomes pack-supplied HygieneConfig
│   │   └── relations/            # REDUCED  predicate index replaces free strings
│   ├── consolidation/
│   │   ├── blocking.py           # NEW  blocking keys
│   │   ├── candidates.py         # NEW  vector-index neighbours, bounded + counted
│   │   └── entity_resolution.py  # REWRITTEN  mention layer, merge log, type histogram
│   ├── pipeline.py               # NEW  Source→Segment→Mention/Fact orchestration
│   └── stm/                      # DELETED at cutover (dialogue machinery)
├── retrieval/
│   ├── retriever.py              # REWRITTEN  symbol-keyed recall, budget, no-evidence
│   ├── profile.py                # NEW  RetrievalProfile supplied by a pack
│   └── models/context.py                                                 # kept
├── packs/                        # NEW LAYER — domain knowledge as data
│   ├── protocol.py               # DomainPack protocol
│   ├── loader.py                 # load, conflict detection, lifecycle
│   ├── code/                     # code pack (concepts, predicates, parser, extractor)
│   └── data/                     # shipped pack data (*.json) — declared to the backend
├── memory.py                     # REWRITTEN facade: Memory(packs=[...]), ingest, recall, forget
├── integrations/
│   ├── client/                   # kept, stdlib-only
│   ├── claude_code/              # NEW  hooks.py — stdlib-only, talks to the service
│   └── langgraph/                                                        # kept
├── server/mcp/                   # + memory_forget tool; topics tools deleted
├── cli/                          # topics.py DELETED
├── topics/                       # DELETED at cutover
├── bounds.py  linguistics.py  nlp.py                                     # kept (linguistics vocabularies become pack guidance, T065); worth.py deleted at cutover (T037)

evaluation/
├── scenarios.py                  # NEW  two-packs, schema-refusal, concurrent-ingest, merge-replay, resolve-scaling, transcript-drift, forget-roundtrip, hook-latency
├── data/identity/labelled.jsonl  # NEW  hand-labelled identity sample (>=200 clusters, not LoCoMo)
├── common/                       # kept (harness commons, reporting, judge, tracing)
├── locomo/  longmem/  beam/      # kept; turn-fed feeding mode added
├── repo/                         # NEW  repository-transcript panel row
├── audit/                        # kept; triplet audit generalised beyond dialogue
├── deadweight.py                 # NEW  written-vs-read type check
├── identity.py                   # NEW  pairwise precision/recall on the labelled sample
└── results/baseline/             # NEW  committed baseline, one file per row + metrics

tests/
├── ingestion/  retrieval/  storage/  models/  channels/  evaluation/     # kept, extended
├── packs/                        # NEW
└── integrations/claude_code/     # NEW
```

**Structure Decision**: single-package modular monolith, unchanged in shape — the slice adds
exactly one new layer, `graphknows.packs`, and one new leaf, `graphknows.integrations.claude_code`.

`.importlinter` changes, which are how FR-003 becomes machine-checked rather than aspirational:

```ini
[importlinter:contract:layers]
layers =
    graphknows.cli | graphknows.server | graphknows.integrations
    graphknows.memory
    graphknows.packs
    graphknows.ingestion | graphknows.retrieval
    graphknows.channels
    graphknows.symbolic
    graphknows.ranking
    graphknows.storage
    graphknows.models | graphknows.settings | graphknows.llm | graphknows.temporal
    graphknows.exceptions

[importlinter:contract:core-knows-no-domain]
name = The core must not import any domain pack
type = forbidden
source_modules =
    graphknows.ingestion
    graphknows.retrieval
    graphknows.channels
    graphknows.symbolic
    graphknows.storage
forbidden_modules =
    graphknows.packs
```

`graphknows.topics` leaves the layer list with the package. The existing
`client-stdlib-only` contract gains `graphknows.integrations.claude_code` as a second
source module, so the hook process cannot acquire an ML import by accident.

Packs sit **above** ingestion and retrieval rather than below, which is what lets a pack
own a parser and a channel without any core module importing it: the core defines the
protocols, `memory.py` loads packs and hands them down as values. The forbidden contract is
what makes the arrow one-way.

## Phasing

Ordered by the spec's own priorities. Each phase is independently demonstrable and gated on
the phase before it.

| Phase | Story | Gate to leave the phase |
|---|---|---|
| **A — Baseline** | US1 (P1) | A committed result file per panel row and per internal metric exists against the unmodified core (the harness and the store's read-tracking instrumentation may already exist), with a noise band from three repeats at a committed per-row limit. No new core code exists. (SC-003) |
| **B — Domain-neutral core** | US2 (P1) | Two packs load into one namespace, both recall facts with evidence, `lint-imports` proves the core imports no pack, and the grep test finds no dialogue term in core modules. (SC-001, SC-002, SC-005, SC-008, SC-013, SC-015) |
| **B′ — Cutover** | FR-045, FR-046 | Single change: old core modules deleted, their tests deleted with them, dropped planes no longer written or declared. Identity precision and dead weight hold and the two-pack, schema-refusal and concurrent-ingest scenarios pass; dialogue rows report not-run and keep the Phase A numbers as the reference for the dialogue-pack spec; the repository row joins the gate at Phase F against its own first recorded run (SC-004). Identity is exact-key only until Phase C (FR-048). |
| **C — Identity** | US3 (P2) | Pairwise precision/recall reported on the labelled sample, no full-table scan, every merge replayable and undoable from its log. (SC-006, SC-007) |
| **D — Repo as memory** | US4 (P2) | Repository source and one transcript ingest into one namespace; a known function and a known decision both recall with correct evidence ranges. (SC-014) |
| **E — Agent loop** | US5 (P3) | Hooks inject only on resolved symbols, inject nothing otherwise, ingest the session at Stop, and load no ML dependency. (SC-009, SC-010, SC-012) |
| **F — Repo panel row** | US6 (P3) | The new row runs end to end with the same metric shape and at least three question kinds. (SC-011) |

**Why B′ is its own phase.** Phase B deletes the dropped planes first and rewrites the
remaining modules in place; B′ is the single change that deletes what is left of the old
core and records that identity is exact-key only until Phase C (FR-048). Nothing routes to
two write paths at any point, and no old database is migrated — which is what FR-045
forbids; an in-place rewrite of a module is not a migration of its data.

## Complexity Tracking

No Constitution Check violation. Table intentionally empty.

Two deliberate ceilings are recorded here rather than hidden, each with its upgrade path,
and both carry a `ponytail:` comment at the call site:

| Simplification | Ceiling | Upgrade when |
|---|---|---|
| Code parsing via stdlib `ast` | Python only | A pack for a non-Python language is specified — then tree-sitter, behind the same `Parser` protocol, no core change. |
| Per-namespace ingest serialisation via an in-process `asyncio.Lock` | One memory-service process per deployment | A second writer process is deployed — then a lease row in ArcadeDB, same lock interface. |
