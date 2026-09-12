# Tasks: Universal Neurosymbolic Memory Core — First Vertical Slice

**Input**: Design documents from `.claude/specs/004-neurosymbolic-memory-core/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: This feature is measurement-gated (Constitution I, VI). Every task carries a
runnable check, so test tasks are folded into the task that owns the behaviour rather than
listed separately.

**Organization**: Tasks are grouped by user story. Phase order follows plan.md's phasing
table (A → B → B′ → C → D → E → F); US1 blocks everything by construction (SC-003).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1…US6); Cutover and Polish tasks carry `[Cutover]` / `[Polish]` instead
- Each task is followed by one `- Verify:` sub-bullet with a single runnable command that
  fails on the tree now and passes once the task is done. Commands run at `/app` in the
  workspace container.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: the harness surface every later phase reports through.

- [X] T001 Extend the dispatcher in `evaluation/__main__.py` with the `baseline`, `panel`, `deadweight`, `identity`, `scenario` and `download` subcommands routing to `evaluation/common/` entry points
  - Verify: `pytest -q tests/evaluation/test_dispatcher.py`
- [X] T002 [P] Define the per-row and internal-metric report shape (`row`, `feeding_mode`, `extractor`, `accuracy`, `evidence_recall`, `by_category`, `n`, `not_run`, and `median`/`band`/`runs`/`commit`) and the per-run internal-metrics object (`identity`, `fact_precision`, `dead_weight`) from contracts/panel-report.md in `evaluation/common/reporting.py`
  - Verify: `pytest -q tests/evaluation/test_report_shape.py`

---

## Phase 2: User Story 1 - Baseline before any new code (Priority: P1) 🎯 MVP

**Goal**: a committed number for every panel row and internal metric against the current
core, so every later change is judged against a record.

**Independent Test**: run the panel harness on the untouched checkout and get a committed
result file per row plus the internal metrics.

- [X] T011 [US1] Feed rows turn by turn in `evaluation/common/pipeline.py` and report `feeding_mode` per row, updating the LongMemEval and BEAM adapters in `evaluation/longmem/` and `evaluation/beam/`
  - Verify: `pytest -q tests/evaluation/test_feeding_mode.py`
- [X] T012 [P] [US1] Implement pairwise identity precision/recall over a labelled sample in `evaluation/identity.py`
  - Verify: `pytest -q tests/evaluation/test_identity_metric.py`
- [X] T013 [P] [US1] Implement the written-minus-read type diff in `evaluation/deadweight.py`: `deadweight` and `panel --compare` exit non-zero on a non-empty `unread_types`, while `baseline` records `passed: false` and continues (FR-041)
  - Verify: `pytest -q tests/evaluation/test_deadweight.py`
- [X] T058 [P] [US1] Generalise the triplet audit in `evaluation/audit/` beyond dialogue so fact precision is reported on every run (FR-040)
  - Verify: `pytest -q tests/evaluation/test_audit_generalised.py`
- [X] T014 [US1] Declare the touched types per store read method and accumulate them into a per-connection `read_types` set in `graphknows/storage/arcadedb/graph_store.py`
  - Verify: `pytest -q tests/storage/test_read_types.py`
- [X] T015 [US1] Implement `python -m evaluation baseline --rows … --repeats 3` writing `median`/`band`/`runs`/`commit` per row and metric under `evaluation/results/baseline/`
  - Verify: `pytest -q tests/evaluation/test_baseline_cli.py`
- [X] T016 [P] [US1] Add the hand-labelled identity sample of at least 200 clusters, not from LoCoMo, at `evaluation/data/identity/labelled.jsonl`: the corpus is this repository's own Claude Code transcripts and source tree, a cluster is the set of mentions (names, paths, symbols) referring to one real referent, labelled by hand, with the procedure recorded beside the file
  - Verify: `pytest -q tests/evaluation/test_identity_sample.py`
- [X] T017 [US1] Record and commit the baseline files for every row and internal metric under `evaluation/results/baseline/` against the unmodified core
  - Verify: `pytest -q tests/evaluation/test_baseline_committed.py`

**Checkpoint**: SC-003 holds — no new core code may be written before this point.

---

## Phase 3: Foundational (Blocking Prerequisites)

**Purpose**: the neutral data model, the packs layer and the machine-checked architecture
rules that every user story below depends on.

**⚠️ CRITICAL**: this phase MUST NOT start until the Phase 2 baseline is committed (SC-003, FR-041). No user story after US1 begins until this phase is complete.

- [X] T003 [P] Create `Source` in `graphknows/models/source.py` with `id` derived from `uri` + `content_hash`, `mime`, `imported_at`, `namespace`, `meta`
  - Verify: `pytest -q tests/models/test_source.py`
- [X] T004 [P] Create `Segment` and the closed `SegmentKind` enum in `graphknows/models/segment.py` with `byte_range`, free-string `role`, `observed_at`, `observed_at_inferred`, `path`, `extractor_version`
  - Verify: `pytest -q tests/models/test_segment.py`
- [X] T005 [P] Create `Mention`, `Fact`, `Polarity`, `Modality`, `Validity`, `FactState` in `graphknows/models/fact.py`
  - Verify: `pytest -q tests/models/test_fact.py`
- [X] T006 [P] Create `ConceptRef`, `PredicateRef` and `SymbolKind` (concept, predicate, episode) in `graphknows/models/symbols.py`
  - Verify: `pytest -q tests/models/test_symbols.py`
- [X] T007 [P] Create `IngestReport`, `RecallResult`, `RecallBudget`, `FactWithEvidence` and the `Counters` set from research.md R9 in `graphknows/models/report.py`
  - Verify: `pytest -q tests/models/test_report.py`
- [X] T008 [P] Add `PackConflictError` to `graphknows/exceptions.py`
  - Verify: `python -c "from graphknows.exceptions import PackConflictError"`
- [X] T009 Create the packs layer: `DomainPack` protocol in `graphknows/packs/protocol.py` and the all-or-nothing loader with concept-URI/predicate-id conflict detection in `graphknows/packs/loader.py`
  - Verify: `pytest -q tests/packs/test_loader.py`
- [X] T010 Add the `packs` layer, the `core-knows-no-domain` forbidden contract and `graphknows.integrations.claude_code` as a `client-stdlib-only` source module to `.importlinter`
  - Verify: `pytest -q tests/test_import_contracts.py`

**Checkpoint**: neutral model and pack layer exist; user story phases can begin.

---

## Phase 4: User Story 2 - A core that does not know the domain (Priority: P1)

**Goal**: two unrelated packs in one namespace, recall returning facts with evidence, and no
domain branch anywhere in the core.

**Independent Test**: load two packs into one namespace, ingest one source per pack, recall a
symbol from each, and confirm both return facts with evidence.

- [X] T034 [US2] Delete `graphknows/topics/`, `graphknows/cli/topics.py` and the `topics_*` MCP tools, with their tests
  - Verify: `pytest -q tests/test_dropped_plane_topics.py`
- [X] T018 [US2] Replace the DDL wholesale with the golden schema in `graphknows/storage/arcadedb/_schema.py`, keeping the `SCHEMA_STAMP` mechanism verbatim
  - Verify: `pytest -q tests/storage/test_golden_schema.py`
- [X] T019 [US2] Define the single `state <> 'forgotten'` fragment in `graphknows/storage/arcadedb/_sql.py` and assert every read query text composes it
  - Verify: `pytest -q tests/storage/test_tombstone_filter.py`
- [X] T020 [US2] Split `graphknows/storage/arcadedb/graph_store.py` into source, segment, fact, entity and symbol writers, rejecting at write any fact without an `ASSERTED_IN` edge (FR-008)
  - Verify: `pytest -q tests/storage/test_writers.py`
- [X] T021 [P] [US2] Add the `Parser` protocol and the MIME-keyed registry in `graphknows/ingestion/parsers/registry.py`, and strip the dialogue heuristics from `graphknows/ingestion/parsers/text.py`
  - Verify: `pytest -q tests/ingestion/parsers/test_registry.py`
- [X] T022 [P] [US2] Add the `Extractor` protocol in `graphknows/ingestion/extraction/protocol.py` and put `LocalExtractor` and `LLMExtractor` behind it
  - Verify: `pytest -q tests/extraction/test_protocol.py`
- [X] T065 [US2] Wire `pack.entity_labels()` (replacing, never unioning), `pack.prompt_addendum()`, `pack.hygiene()` and `pack.thresholds()` into `LocalExtractor` and `LLMExtractor` under `graphknows/ingestion/extraction/`, deleting the core's conversational label inventory and hygiene defaults and moving `linguistics.py` vocabularies into pack guidance (FR-022)
  - Verify: `pytest -q tests/extraction/test_pack_guidance.py`
- [X] T023 [P] [US2] Add the predicate index in `graphknows/symbolic/predicates.py`, replacing free-string relations in `graphknows/ingestion/extraction/relations/`
  - Verify: `pytest -q tests/ontology/test_predicate_index.py`
- [X] T024 [US2] Implement the Source → Segment → Mention/Fact orchestration in `graphknows/ingestion/pipeline.py`, including bottom-up concept labelling from the loaded packs: `SEGMENT-[EVOKES]->CONCEPT` and `ENTITY-[INSTANCE_OF]->CONCEPT` (FR-002)
  - Verify: `pytest -q tests/ingestion/test_pipeline.py`
- [X] T025 [US2] Deduplicate on `Source.content_hash` before any model runs and report `sources_deduplicated` in `graphknows/ingestion/pipeline.py`
  - Verify: `pytest -q tests/ingestion/test_dedup.py`
- [X] T026 [US2] Implement the functional-predicate currency rule (supersede newer/older, contradict on equal times) in `graphknows/ingestion/pipeline.py`
  - Verify: `pytest -q tests/ingestion/test_supersede.py`
- [X] T027 [US2] Rewrite `graphknows/retrieval/retriever.py` as symbol-keyed recall returning facts with evidence under an explicit budget, with `no_evidence` and `truncated_by`, and top-down activation reading `EVOKES` and `INSTANCE_OF` for candidate generation (FR-002)
  - Verify: `pytest -q tests/retrieval/test_symbol_recall.py`
- [X] T028 [P] [US2] Add the `RetrievalProfile` value object with core defaults in `graphknows/retrieval/profile.py`; pack supply arrives with the dialogue pack (FR-022)
  - Verify: `pytest -q tests/retrieval/test_profile.py`
- [X] T029 [US2] Rewrite `graphknows/channels/registry.py` as the fixed core collectors with no `enable_x` flags; pack channels arrive with the dialogue pack (FR-022)
  - Verify: `pytest -q tests/channels/test_registry_packs.py`
- [X] T030 [US2] Rewrite the facade in `graphknows/memory.py` as `Memory(packs=[...])` with `ingest`/`recall`/`forget` and a per-namespace `asyncio.Lock` reporting `ingest_queue_wait_ms`, and the `graphknows.cli` `ingest`/`recall` commands on that facade
  - Verify: `pytest -q tests/memory/test_facade_packs.py`
- [X] T032 [P] [US2] Add the grep test asserting no benchmark, dialogue, speaker or turn-regex term appears in core modules, at `tests/test_core_domain_neutral.py`
  - Verify: `pytest -q tests/test_core_domain_neutral.py`
- [X] T043 [P] [US2] Implement the stdlib-`ast` code parser emitting one segment per function and class with the qualified symbol as `path`, in `graphknows/ingestion/parsers/code.py`
  - Verify: `pytest -q tests/ingestion/parsers/test_code.py`
- [X] T044 [US2] Ship the code pack — concepts, predicates (calls, imports, defines, tests), `CodeExtractor` with no model, a veto rule (a function and a module symbol with the same normalised name never merge, FR-019), and its data — under `graphknows/packs/code/` and `graphknows/packs/data/`
  - Verify: `pytest -q tests/packs/test_code_pack.py`
- [X] T045 [P] [US2] Implement the tolerant, versioned agent-transcript parser in `graphknows/ingestion/parsers/transcript.py`, counting unknown record and block types and retaining timestamp, `cwd`, `gitBranch` and `isSidechain`
  - Verify: `pytest -q tests/ingestion/parsers/test_transcript.py`
- [X] T046 [US2] Link tool results as citable `Source`s rather than segments in `graphknows/ingestion/parsers/transcript.py`, and assert recall survives a citation to a tool result that was never ingested
  - Verify: `pytest -q tests/ingestion/parsers/test_transcript_tool_results.py`
- [X] T064 [US2] Ship the agent pack under `graphknows/packs/agent/` and `graphknows/packs/data/`: concepts session, prompt, decision, file, tool; predicates asked_in, touched, changed, decided_in and the functional decision_status; a deterministic `AgentExtractor` emitting facts from tool_use/tool_result blocks; extraction guidance labelling `decision` from user and assistant text; an open decision is one with no superseding `decision_status` fact (FR-004, FR-032)
  - Verify: `pytest -q tests/packs/test_agent_pack.py`
- [X] T033 [US2] Implement the `two-packs`, `schema-refusal` and `concurrent-ingest` scenarios in `evaluation/scenarios.py`
  - Verify: `pytest -q tests/evaluation/test_scenarios.py`

**Checkpoint**: SC-001, SC-002, SC-008, SC-013, SC-015 demonstrable on the new core.

---

## Phase 5: Cutover (FR-045, FR-046)

**Purpose**: one change — what is left of the old core goes and the interim identity state
is recorded. The dropped planes were already deleted at the head of Phase 4 so the tree never
writes a type the schema no longer declares. Tests are deleted only in the change that deletes
the module they import.

<!-- T035/T036 moved here from Phase 4 on 2026-09-09: deleting the STM package before the new
pipeline (T024) and facade (T030) exist breaks memory.py at import time, and dropping the
frame types before T027/T029 leaves their readers dangling. They are cutover deletions. -->
- [X] T035 [Cutover] Delete `graphknows/ingestion/stm/` and the closed `Role` enum in `graphknows/models/message.py`; a session is a `Source`, a turn is a `Segment`
  - Verify: `pytest -q tests/test_dropped_plane_stm.py`
- [X] T036 [Cutover] Stop declaring and writing the frame-element mirror, frame-specific types, graph-ranking columns and free-string relations, reducing `graphknows/symbolic/framenet/` to a concept+predicate emitter
  - Verify: `pytest -q tests/test_dropped_plane_frames.py`
- [X] T037 [Cutover] Delete the remaining old core modules in one change — `graphknows/symbolic/wordnet.py`, `graphknows/worth.py`, `graphknows/ingestion/extraction/entities/hygiene.py`'s dialogue-only rules, the old `entity_resolution.py` layers replaced at T063, and every test that imports them; `bounds.py`, `nlp.py` and `linguistics.py` stay. `graphknows/memory.py` was rewritten in place at T030 and nothing routes to two write paths (FR-045)
  - Verify: `pytest -q tests/memory/test_cutover.py`
- [X] T063 [Cutover] Implement exact-key-only identity on the cut-over core — no `MERGED_INTO` and no `SAME_AS` edge is written — and the cutover-report line that states it (FR-048)
  - Verify: `pytest -q tests/ingestion/consolidation/test_exact_key_interim.py`
<!-- T063's interim guard (tests/ingestion/consolidation/test_exact_key_interim.py, the
EXACT_KEY_ONLY banner) was deleted on 2026-09-10 when T040 landed: FR-048 scopes exact-key-only
identity to the window *between cutover and the identity story*, and that window closed. -->
<!-- T066-T068 added 2026-09-10: the cutover run (T037) deleted modules but left the old core's
read/write paths inside graph_store.py, memory.py and retriever.py; tests/storage/test_read_types.py
(T014) is the signal — 24 read methods still declare CHUNK/TURN/FRAME types the golden schema no
longer has. -->
- [X] T066 [Cutover] Make `graphknows/storage/arcadedb/graph_store.py` speak only the golden schema: delete every read and write method over the dropped types (CHUNK, TURN, SESSION, FILE, TEMPORAL, FRAME, FRAME_INSTANCE, FE, SEMTYPE, TOPIC, SYNSET, SENSE, REL) and the graph-analytics methods (pagerank, communities, graph embeddings, analytical view), add the golden reads the retriever needs (segment ANN, full-text and by-id, neighbours by NEXT, facts by entity and predicate, entity resolution by name key), and delete the tests that imported the removed methods (FR-045, FR-046)
  - Verify: `pytest -q tests/storage/test_read_types.py`
- [X] T067 [Cutover] Rewrite the collectors in `graphknows/retrieval/retriever.py` and `graphknows/channels/base.py` over SEGMENT and FACT through the store's golden reads; no chunk-plane read remains (FR-002, FR-045)
  - Verify: `pytest -q tests/retrieval/test_no_chunk_reads.py tests/retrieval/test_symbol_recall.py`
- [X] T068 [Cutover] Route `graphknows/memory.py` through `graphknows/ingestion/pipeline.py` only: drop the TURN buffer, the flush-time analytics (pagerank, communities, graph embeddings) and session consolidation; `flush` becomes identity resolution plus the dead-weight report; update `graphknows/storage/namespace.py`, `graphknows/ingestion/extraction/relations/verifier.py` and `graphknows/symbolic/ontology/rdf/store.py` callers (FR-045, FR-046)
  - Verify: `pytest -q tests/memory/test_no_old_core_paths.py tests/memory/test_facade_packs.py`
- [X] T069 [Cutover] Make ingest and recall work on a live ArcadeDB: each pack reads its own domain with its own deterministic extractor, so facts, `EVOKES`, `INSTANCE_OF`, `NEXT` and the segment embeddings reach the database and a recall returns facts with evidence (FR-002, FR-008, FR-030)
  - Verify: `pytest -q -m integration tests/storage/test_live_ingest.py`

**Checkpoint**: SC-004 and SC-005 — identity precision and dead weight hold and the three
scenarios pass; dialogue rows report not-run and the repository row joins the gate in Phase 9
(FR-042); identity is exact-key only until Phase 6 (FR-048).

---

## Phase 6: User Story 3 - Identity that does not guess (Priority: P2)

**Goal**: bounded candidate generation, precision-first merging, replayable merge log.

**Independent Test**: ingest a corpus with known duplicates and known distinct entities, check
pairwise precision and recall against the labels, and replay every merge from its log.

- [X] T038 [P] [US3] Implement blocking keys (`name_norm[:4]` + dominant type) in `graphknows/ingestion/consolidation/blocking.py`
  - Verify: `pytest -q tests/ingestion/consolidation/test_blocking.py`
- [X] T039 [P] [US3] Implement bounded vector-index candidate generation with a `candidates_truncated` counter in `graphknows/ingestion/consolidation/candidates.py`
  - Verify: `pytest -q tests/ingestion/consolidation/test_candidates.py`
- [X] T040 [US3] Rewrite `graphknows/ingestion/consolidation/entity_resolution.py` with the mention layer, veto → strong → weak ladder, type histogram and `MERGED_INTO` log written before any structural change
  - Verify: `pytest -q tests/ingestion/consolidation/test_entity_resolution_layers.py`
- [X] T041 [US3] Implement merge replay and undo from the `MERGED_INTO` log alone, and the `merge-replay` scenario in `evaluation/scenarios.py`
  - Verify: `pytest -q tests/ingestion/consolidation/test_merge_replay.py`
- [X] T061 [US3] Resolve `SAME_AS` candidate edges at recall time in `graphknows/retrieval/retriever.py`: a recalled entity's candidates are unified into one committed entity and the commitment is reported (FR-018)
  - Verify: `pytest -q tests/retrieval/test_candidate_commit.py`
<!-- T070 added 2026-09-10: measured on the live database — SymbolRecall._resolve matches CONCEPT
labels only and _terms() splits on word boundaries, so a query naming an entity ("load_packs",
"graphknows/memory.py") resolves to nothing and the symbol-keyed hooks (FR-032, FR-033) would
inject nothing. -->
- [X] T070 [US3] Resolve query terms to entities as well as concepts in `graphknows/retrieval/retriever.py`: keep identifier-shaped terms whole (snake_case, dotted paths, slashes), match them against `ENTITY.name_norm` through the same normalisation ingest uses, and activate the facts those entities are subject of (FR-032, FR-033)
  - Verify: `pytest -q tests/retrieval/test_entity_resolution_recall.py`
- [X] T042 [US3] Implement the `resolve-scaling` scenario in `evaluation/scenarios.py`, failing when per-batch wall clock at ten times the namespace size is not under twice the base (SC-006)
  - Verify: `pytest -q tests/evaluation/test_resolve_scaling.py`

**Checkpoint**: SC-006 and SC-007 hold.

---

## Phase 7: User Story 4 - The repository and its transcripts become memory (Priority: P2)

**Goal**: code and agent transcripts ingest as data, with no core change.

**Independent Test**: ingest the repository and one transcript into one namespace and recall a
known function and a known decision with correct evidence ranges.

- [X] T047 [P] [US4] Declare `graphknows/packs/data/**/*.json` to the build backend in `pyproject.toml` and assert it in `tests/test_packaging.py`
  - Verify: `pytest -q tests/test_packaging.py -k pack_data`
- [X] T048 [US4] Implement the `transcript-drift` scenario in `evaluation/scenarios.py`, proving a drifted record is counted and never fatal
  - Verify: `pytest -q tests/evaluation/test_transcript_drift.py`

**Checkpoint**: SC-014 holds; the phase diff touches packs and parsers only.

---

## Phase 8: User Story 5 - Recall arrives in the agent's loop (Priority: P3)

**Goal**: symbol-keyed injection into a live session, stdlib-only, and the session written
back without being asked.

**Independent Test**: run a session with the integration installed; each event injects only
symbol-resolved facts, an unrelated prompt injects nothing, and the session is ingested at the
end.

- [X] T049 [US5] Implement the `context`, `recall`, `preview`, `observe`, `remember` and `catchup` verbs in `graphknows/integrations/claude_code/hooks.py`, talking only to the service through `graphknows/integrations/client/`
  - Verify: `pytest -q tests/integrations/claude_code/test_hooks.py`
- [X] T050 [US5] Implement per-transcript checkpointing by record `uuid` in `graphknows/integrations/claude_code/hooks.py`, never advancing past a malformed trailing record
  - Verify: `pytest -q tests/integrations/claude_code/test_checkpoint.py`
- [X] T051 [P] [US5] Add the subprocess test asserting `torch`, `transformers`, `spacy`, `sentence_transformers` and `gliner` are absent from `sys.modules` after importing the hook module, at `tests/integrations/claude_code/test_no_ml_imports.py`
  - Verify: `pytest -q tests/integrations/claude_code/test_no_ml_imports.py`
- [X] T031 [US5] Implement forget as a tombstone in the store, cascading from a forgotten `Segment` to the facts it evidences and counting `facts_excluded_tombstoned`, and expose the `memory_forget` tool in `graphknows/server/mcp/`
  - Verify: `pytest -q tests/mcp_server/test_forget_tool.py`
- [X] T052 [US5] Implement the `forget-roundtrip` scenario in `evaluation/scenarios.py`, proving a forgotten record vanishes from recall while its tombstone and merge log stay readable
  - Verify: `pytest -q tests/evaluation/test_forget_roundtrip.py`
- [X] T053 [P] [US5] Ship the record-versus-retrieve guidance skill and the hook settings block with `graphknows/integrations/claude_code/`
  - Verify: `pytest -q tests/integrations/claude_code/test_guidance_shipped.py`
- [X] T062 [US5] Implement the `hook-latency` scenario in `evaluation/scenarios.py`, measuring recall-verb p95 against the 1 s budget and the 5 s ceiling over 200 samples against a warm service (SC-010)
  - Verify: `pytest -q tests/evaluation/test_hook_latency.py`

**Checkpoint**: SC-009, SC-010 and SC-012 hold.

---

## Phase 9: User Story 6 - A yardstick that includes this repository's own work (Priority: P3)

**Goal**: a panel row built from this repository's transcripts and git history, scored the
same way as every other row.

**Independent Test**: run the new row end to end and get a report with the same metric shape.

- [X] T054 [US6] Implement the repository-transcript row adapter in `evaluation/repo/` running through `evaluation/common/pipeline.py`
  - Verify: `pytest -q tests/evaluation/test_repo_row.py`
- [X] T055 [US6] Build the gold set in `evaluation/repo/` covering decision location, change rationale and test coverage of a symbol
  - Verify: `pytest -q tests/evaluation/test_repo_row_kinds.py`
- [X] T056 [US6] Implement `python -m evaluation panel --compare` as the merge gate — no row below `median − band`, identity precision not dropped, dead weight clean
  - Verify: `pytest -q tests/evaluation/test_panel_compare.py`
- [X] T057 [US6] Report the local and LLM extraction paths as separate columns in `evaluation/common/reporting.py`
  - Verify: `pytest -q tests/evaluation/test_extractor_columns.py`

**Checkpoint**: SC-011 holds; all six stories are independently demonstrable.

---

## Phase 10: Polish & Cross-Cutting Concerns

- [X] T059 [P] [Polish] Document the packs layer, the hook integration and the merge gate in `docs/` and `README.md`
  - Verify: `pytest -q tests/test_docs_packs.py`
- [X] T060 [Polish] Assert every command in `quickstart.md` exists and is dispatchable, skipping `<...>` operator placeholders
  - Verify: `pytest -q tests/test_quickstart_commands.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **US1 (Phase 2)**: depends on Setup. Blocks all later phases by construction — no new core
  code may exist when the baseline is recorded (SC-003). The harness and the store's
  read-tracking instrumentation (T014) are not core code (FR-041).
- **Foundational (Phase 3)**: depends on the committed baseline. Blocks every later user story.
- **US2 (Phase 4)**: depends on Phase 2 and Phase 3.
- **Cutover (Phase 5)**: depends on Phase 4; its gate is identity precision, dead weight and the three scenarios (FR-042).
- **US3 (Phase 6)**, **US4 (Phase 7)**: depend on Cutover; independent of each other.
- **US5 (Phase 8)**: depends on Cutover (parsers and packs land in Phase 4).
- **US6 (Phase 9)**: depends on US4 and the panel harness from US1.
- **Polish (Phase 10)**: last.

### Within Each User Story

Models → storage → parsers/extractors → pipeline → retrieval → facade → scenarios.

### Parallel Opportunities

- T002 alongside T001.
- T003–T008 all in parallel (separate model files), only after T017.
- T012, T013, T016 in parallel within US1.
- T021, T022, T023, T028, T032 in parallel within US2.
- T038 and T039 in parallel within US3.
- T043, T045 in parallel within Phase 4 before T033; T047 alone in US4.
- US3 and US4 can run in parallel after Cutover.

---

## Parallel Example: Foundational

```bash
Task: "Create Source in graphknows/models/source.py"
Task: "Create Segment and SegmentKind in graphknows/models/segment.py"
Task: "Create Mention/Fact in graphknows/models/fact.py"
Task: "Create ConceptRef/PredicateRef/SymbolKind in graphknows/models/symbols.py"
Task: "Create IngestReport/RecallResult/Counters in graphknows/models/report.py"
Task: "Add PackConflictError to graphknows/exceptions.py"
```

---

## Implementation Strategy

### MVP First

1. Phase 1 Setup.
2. Phase 2 (US1) — record and commit the baseline. **Stop and validate**: SC-003.
3. Phase 3 Foundational.
4. Phase 4 (US2) — the domain-neutral core. This is the feature.
5. Phase 5 Cutover — one change; repository row, identity precision and dead weight hold.

### Incremental Delivery

Each of US3, US4, US5, US6 adds one demonstrable capability on top of the cut-over core and is
gated on the whole panel, never on a single row.

---

## Notes

- `[P]` tasks touch different files and have no dependency on an incomplete task.
- Each `- Verify:` command must fail before its task and pass after; a Verify that already
  passes means the task is not a task.
- A test is deleted only in the change that deletes the module it imports (plan.md, Principle
  II note).
- The merge gate is `make gate` plus `python -m evaluation panel --compare
  evaluation/results/baseline` plus `python -m evaluation deadweight`. A green `make gate`
  alone is not acceptance.

---

## Phase 11: Follow-ups found by running the thing (added 2026-09-10)

**Purpose**: three gaps the live runs exposed after every task was ticked. Each was reported
rather than quietly absorbed, and each has a runnable check.

- [X] T071 [Polish] Add the `memory_recall` MCP tool specified by `contracts/mcp-tools.md` — a thin adapter over `Memory.recall` returning `RecallResult` as JSON (`facts` with `source_uri`/`byte_range`/`text` evidence, `no_evidence`, `budget`, `truncated_by`, `counters`) — and switch `_recall_about` in `graphknows/integrations/claude_code/hooks.py` from `memory_query`'s fact sheet to it (FR-009, FR-015, SC-002)
  - Verify: `pytest -q tests/mcp_server/test_recall_tool.py`
- [X] T072 [Polish] Build the repository row on real agent transcripts (FR-044): read them at runtime from the user's own Claude Code project directory rather than committing session content, keep one committed fixture carrying the real sixteen-field record shape (`uuid`, `parentUuid`, `sessionId`, `cwd`, `gitBranch`, `isSidechain`) so the loader is exercised on the shape it meets in the wild and CI stays deterministic, derive each gold answer from that transcript and `git log`, and correct `evaluation/data/repo/README.md`, which currently claims the fixtures keep bookkeeping records they do not have
  - Verify: `pytest -q tests/evaluation/test_repo_row_is_real.py`
