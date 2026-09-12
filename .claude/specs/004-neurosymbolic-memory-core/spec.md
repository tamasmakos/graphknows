# Feature Specification: Universal Neurosymbolic Memory Core — First Vertical Slice

**Feature Branch**: `004-neurosymbolic-memory-core`

**Created**: 2026-09-08

**Status**: Draft

**Input**: User description: "Feature: universal neurosymbolic memory core, first vertical slice (code and agent work)." Evidence: `docs/generalisation-audit.md` (60 issues with `file:line` anchors). Twelve binding owner decisions, two marked provisional, reproduced in Assumptions.

## Context

Today's memory is shaped by one benchmark. The core knows it is reading dialogue (a
`Speaker: text` regex in parsing, extraction, hygiene, deixis and keyword search), four
stored planes are written and never read, one of three symbolic channels runs by default,
identity resolution scans the whole entity table in Python and merges by irreversible
delete, and no input other than plain prose has a parser.

This slice makes the memory **domain-neutral by construction** and proves it on the domain
that matters first: an agent working in a repository. The agent's own transcripts and the
repository's code become memory; recall comes back into the agent's loop as facts with
evidence, keyed by symbols the prompt actually mentions. Dialogue becomes just another
domain pack, specified separately and out of scope here — but the core must accept it
later without a core change.

## Clarifications

### Session 2026-09-08

- Q: Should each segment keep exactly one observation time, or should every fact carry its own time anchor? → A: Keep one observation time per segment; fact-level dates stay optional validity metadata. Adopt a per-fact anchor only if a temporal or knowledge-update panel row fails against this model (FR-013) (owner confirmed 2026-09-08)
- Q: Should episode symbols be materialised as stored vertices in this slice? → A: No. Episode stays in the symbol vocabulary, but nothing is stored until a traversal measurement shows recall that segment observation times and fact validity cannot answer; an unread episode plane would fail the dead-weight check (FR-014) (owner confirmed 2026-09-08)
- Q: Do both the local and the language-model extraction paths stay first-class in this slice, or does the local pipeline move to a later one? → A: Both stay first-class and both are gated on the panel — the code pack needs the deterministic local path anyway (FR-030), so deferring it would remove a shipped capability, not save work (FR-043) (owner confirmed 2026-09-08)
- Q: What happens when two sessions ingest into the same namespace at the same time? → A: One writer at a time per namespace — the memory service serialises ingest for a namespace and queues the rest; readers are unaffected (FR-047) (owner confirmed 2026-09-08)
- Q: Does forget physically delete a fact or entity, or mark it excluded from recall? → A: Tombstone — the record is marked forgotten and excluded from every retrieval path, so merge logs and provenance stay replayable; physical purge is out of scope (FR-037) (owner confirmed 2026-09-08)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Baseline before any new code (Priority: P1)

A maintainer needs to know what the current system scores before it is replaced, on every
row of the yardstick panel and on the internal quality metrics, so that every later change
is judged against a recorded number instead of a memory of one.

**Why this priority**: Constitution VI (Evidence Over Vibes). A rebuild with no
pre-measurement can never demonstrate it did not regress; the panel is also the merge gate
for every later story here.

**Independent Test**: Run the panel harness against the current (old) core on an untouched
checkout and get a committed results file per row plus the internal metrics, with no new
core code existing yet.

**Acceptance Scenarios**:

1. **Given** the current core at the baseline commit, **When** the panel is run, **Then** a
   committed result exists for each panel row (LoCoMo, LongMemEval, BEAM) recording
   accuracy, evidence recall and per-category breakdown.
2. **Given** a panel row whose data has turns, **When** it is run, **Then** it is fed turn
   by turn, not as one blob per session, and the report states which feeding mode was used.
3. **Given** a hand-labelled identity sample, **When** the metrics run, **Then** pairwise
   precision and recall for entity resolution are reported as numbers.
4. **Given** a run of the panel, **When** any stored vertex or edge type was written during
   ingest and read by no retrieval path, **Then** the dead-weight check reports it and
   fails the run.

---

### User Story 2 - A core that does not know the domain (Priority: P1)

A maintainer loads packs for two unrelated domains into one namespace, ingests sources of
different shapes, and asks "what do you know about X". The answer is a budgeted slice of
facts, each carrying the evidence segment it came from. No part of the core needed to know
which domain the input belonged to.

**Why this priority**: This is the feature. Every other story is either the measurement
around it or a client of it.

**Independent Test**: Load two packs into one namespace, ingest one source per pack, recall
a symbol from each, and confirm both return facts with evidence — with no domain name,
kind check or benchmark heuristic anywhere in the core's own code path.

**Acceptance Scenarios**:

1. **Given** two domain packs loaded into a single namespace, **When** sources for both are
   ingested and recalled, **Then** both return correct results and the core contains no
   branch on domain, corpus or input kind.
2. **Given** a source in any supported shape, **When** it is ingested, **Then** it is stored
   as one source record plus segments carrying kind, structure path, byte offsets and a
   time anchor, and re-ingesting the identical content adds nothing.
3. **Given** an ingested segment, **When** extraction runs, **Then** every asserted fact
   records subject, predicate, object, polarity, modality, confidence, validity, extractor
   identity and version, and links to the evidence segment with byte offsets.
4. **Given** a recall for a symbol, **When** results are returned, **Then** they are facts
   with evidence, never bare segments, and the response is bounded by an explicit budget.
5. **Given** a query whose symbol resolves to nothing above the floor, **When** recall runs,
   **Then** an explicit "no evidence" result is returned rather than an empty list that
   reads as a lookup failure.
6. **Given** a database written by a different schema version, **When** it is opened,
   **Then** the open is refused with a clear message; no migration and no compatibility
   path exists.
7. **Given** a functional predicate receiving a newer value, **When** it is asserted,
   **Then** the older fact is retained, marked not current, and linked as superseded.

---

### User Story 3 - Identity that does not guess (Priority: P2)

Two mentions of the same name in different files must not silently become one thing, and
one thing seen under two names must still be findable. Merges the system is sure about
happen and are replayable; merges it is not sure about are recorded as candidates and
decided at recall time.

**Why this priority**: The audit's worst correctness risk (full-table scan, first-wins
typing, irreversible delete-merge). Recall quality in every other story rests on it.

**Independent Test**: Ingest a corpus with known duplicate and known distinct entities,
then check pairwise precision and recall against the labels, and check that every merge is
reconstructible from its log.

**Acceptance Scenarios**:

1. **Given** two mentions with identical normalised keys within a block and agreeing
   concepts, **When** ingest resolves them, **Then** they merge and a replayable merge-log
   edge records the layer, the evidence and the time.
2. **Given** weaker agreement (same concept via a pack, or embedding similarity above the
   floor plus margin), **When** ingest resolves them, **Then** a scored candidate
   same-as edge is written and no merge happens until recall commits to one.
3. **Given** two entities under disjoint sibling concepts, or a pack veto rule, **When**
   resolution runs, **Then** no merge and no candidate edge is created.
4. **Given** a mention of an entity, **When** it is stored, **Then** its surface form and
   span survive as a mention edge and are never collapsed into an alias string.
5. **Given** an entity seen under conflicting types, **When** its type is read, **Then** a
   histogram of observed types is available, not the first one seen.
6. **Given** a namespace with a large entity population, **When** a batch is resolved,
   **Then** only blocked candidates and vector-index neighbours are compared, and no
   full-table scan occurs.

---

### User Story 4 - The repository and its transcripts become memory (Priority: P2)

A maintainer points the system at this repository's source and at the agent transcripts for
it. Functions, classes and modules become recallable symbols with their call, import,
definition and test relations; the agent's prompts, replies and its own stated decisions
become segments with facts attached.

**Why this priority**: The first real proof that a non-dialogue domain is data, not code —
and the raw material for stories 5 and 6.

**Independent Test**: Ingest the repository and one transcript file into one namespace and
recall a known function name and a known decision, confirming both return facts with
evidence pointing at the right file range and the right transcript record.

**Acceptance Scenarios**:

1. **Given** a source file, **When** it is parsed by the code pack, **Then** each function
   and class becomes one segment whose path is its qualified symbol and whose byte offsets
   locate it in the file.
2. **Given** parsed code, **When** extraction runs, **Then** call, import, definition and
   test relations are asserted as facts with no model inference required.
3. **Given** an agent transcript file, **When** it is parsed, **Then** user prompts,
   assistant messages and the agent's own writes become segments; tool results do not
   become segments but are linked as citable sources.
4. **Given** a transcript already ingested, **When** it is ingested again or extended,
   **Then** previously seen records are skipped by their record identifier and only new
   records are added.
5. **Given** a transcript record, **When** it is stored, **Then** its timestamp, working
   directory, branch and side-chain flag are retained as metadata.
6. **Given** a transcript whose format has drifted (unknown block type, missing field),
   **When** it is parsed, **Then** the unknown part is skipped with a named counter rather
   than aborting the ingest or being silently dropped.
7. **Given** the code pack and the transcript source loaded together, **When** either is
   recalled, **Then** neither required a change in the core.

---

### User Story 5 - Recall arrives in the agent's loop, keyed by symbols (Priority: P3)

While the agent works, memory shows up exactly where it is relevant: what is already known
about this working directory at session start, about the entities named in a prompt, about
the file about to be edited, about the symbols just read. Nothing shows up when nothing
resolves. What the session produces is written back without the user asking.

**Why this priority**: The stated first goal — memory for an agent in a loop — but it is
only demonstrable once stories 2 and 4 exist.

**Independent Test**: Run a session in this repository with the integration installed and
confirm each event injects only symbol-resolved facts, that an unrelated prompt injects
nothing, and that the session's own records are ingested by the end of it.

**Acceptance Scenarios**:

1. **Given** a session starting in a known working directory, **When** it starts, **Then** a
   working-context block is injected containing open decisions and the current values of
   functional predicates for that directory.
2. **Given** a prompt naming an entity the memory knows, **When** it is submitted, **Then**
   facts attached to the resolved symbols are injected newest first, resolved through the
   same identity path used at ingest.
3. **Given** a prompt that resolves to no known symbol, **When** it is submitted, **Then**
   nothing is injected.
4. **Given** an edit or write about to touch a path, **When** the tool call is prepared,
   **Then** facts about that path are injected.
5. **Given** a read or search that just returned, **When** it completes, **Then** facts
   about the symbols observed are injected.
6. **Given** a session that stops, or is about to compact, **When** the event fires, **Then**
   transcript records new since the last checkpoint are ingested.
7. **Given** any of these events, **When** the integration runs, **Then** it talks to the
   running memory service over the lightweight client only and loads no machine-learning
   dependency.
8. **Given** a fact or entity the user wants gone, **When** forget is called for it,
   **Then** it stops appearing in recall.
9. **Given** an agent with the shipped guidance installed, **When** it decides whether to
   record or retrieve something, **Then** that guidance states when to do each.

---

### User Story 6 - A yardstick that includes this repository's own work (Priority: P3)

The panel gains a row built from this repository's real agent transcripts, with gold
answers taken from git history and the transcripts themselves: where a decision was made,
why a line changed, which test guards a function. Merging is gated on the whole panel, not
on any single row.

**Why this priority**: Without it, the agent-memory claim is untested; but the panel and
the core must exist first.

**Independent Test**: Run the new row end to end and get a scored report with the same
metrics shape as the other rows.

**Acceptance Scenarios**:

1. **Given** this repository's transcripts and git history, **When** the eval is built,
   **Then** it contains question–answer items with evidence keys of at least three kinds
   (decision location, change rationale, test coverage of a symbol).
2. **Given** the full panel, **When** a change is proposed, **Then** it merges only if no
   row regresses beyond recorded run noise and identity precision does not drop.
3. **Given** a panel run, **When** it reports, **Then** the local and the language-model
   extraction paths appear as separate columns.
4. **Given** the panel, **When** a single row moves, **Then** no row is treated as the
   headline result on its own.

---

### Edge Cases

- A source with no usable time anchor: the ingestion time is used and the segment is
  flagged as inferred, never silently dated.
- A transcript record referencing a tool result that was never ingested: the fact cites the
  source, and recall does not break on the missing segment.
- A file that parses into zero segments (empty, binary, unsupported syntax): counted and
  reported, not silently skipped.
- A name that blocks into a very large candidate set: comparison stays bounded and the
  truncation is reported.
- Two packs claiming the same concept identifier or predicate label: the conflict is
  detected at load, not resolved by last-writer-wins.
- A recall budget smaller than the facts attached to one symbol: the slice is truncated by
  a stated ordering, and the truncation is visible.
- A transcript still being written while it is ingested: partial or malformed trailing
  records do not corrupt the checkpoint.
- A fact whose evidence segment was later forgotten: the fact is tombstoned with it and
  neither is returned by any recall path.
- Concurrent sessions writing into the same namespace: ingest is serialised per namespace,
  so identity resolution never interleaves into a wrong merge; the waiting session queues
  rather than failing.

## Requirements *(mandatory)*

### Functional Requirements

**Core shape**

- **FR-001**: The core MUST consist of exactly two layers — a representation state
  (embeddings and the fused candidate set) and a symbolic index with exactly three symbol
  kinds: concept, predicate and episode — plus the traversal between them, the storage
  shape, and rank fusion.
- **FR-002**: The traversal MUST run both directions: bottom-up labelling of segments with
  symbols at ingest, and top-down activation of a symbol pulling back the facts and
  segments attached to it.
- **FR-003**: The core MUST NOT branch on domain, corpus, benchmark or input kind anywhere.
  Everything domain-specific MUST arrive as pack data: parsers, vocabularies, extraction
  guidance and retrieval profile.
- **FR-004**: The system MUST support loading packs for multiple unrelated domains into one
  namespace simultaneously, with no core change and no cross-pack interference.
- **FR-005**: The system MUST refuse to open a database written by a different schema
  version. No migration, compatibility shim, feature flag or old write path is provided.
- **FR-047**: Ingest into one namespace MUST be serialised to a single writer at a time;
  concurrent ingest requests queue rather than interleave, so identity resolution never sees
  a half-written batch. Recall MUST remain available while a write is in progress.

**Sources, segments, facts**

- **FR-006**: Ingestion MUST record every input as a source with its identity, location,
  media type, content hash and import time, and as segments carrying text, kind, structure
  path, byte offsets, optional role, time anchor, and the version of the extractor that
  produced their annotations.
- **FR-007**: Identical content MUST NOT be ingested twice; deduplication MUST happen before
  any model runs.
- **FR-008**: A fact MUST record subject, predicate, object, polarity, modality, confidence,
  validity interval, current-ness, extractor identity and version, and MUST link to the
  evidence segment with byte offsets.
- **FR-009**: Segments MUST be evidence only. Recall MUST NOT return a segment as a result
  in its own right.
- **FR-010**: "What do you know about X" MUST return a budgeted subgraph slice assembled
  from facts, with their evidence.
- **FR-011**: Predicates MUST be indexed symbols with a definition and a functional flag,
  never free strings.
- **FR-012**: A fact asserted on a functional predicate with a newer time anchor MUST
  supersede the older one: the old fact is retained, marked not current, and linked by a
  supersedes relation.
- **FR-013**: Each segment MUST carry exactly one observation time supplied by its parser,
  falling back to ingestion time with an explicit inferred flag. Dates found in text are
  optional fact-level validity metadata resolved against that observation time. This model
  is fixed for this slice; a per-fact anchor is adopted only if a temporal or
  knowledge-update panel row fails against it.
- **FR-014**: The episode symbol kind MUST be present in the core's symbol vocabulary, and
  episodes MUST NOT be materialised as stored vertices in this slice: recall answers time
  questions from segment observation times and fact validity. Materialisation waits for a
  traversal measurement showing recall those cannot answer; until then a stored episode
  plane would have no reader and MUST fail the dead-weight check (FR-024).
- **FR-015**: Retrieval MUST return an explicit no-evidence result when the candidate pool
  is empty or below the floor.

**Identity**

- **FR-016**: Every mention MUST be preserved with its surface form and span as an edge from
  its segment to the entity; surfaces MUST NOT be collapsed into alias strings.
- **FR-017**: Strong evidence (identical normalised key within a block and agreeing
  concepts) MUST merge at ingest and write a replayable merge-log edge recording layer,
  evidence and time.
- **FR-018**: Weaker evidence (shared concept via a pack, or embedding similarity above the
  floor plus margin) MUST produce a scored same-as candidate edge, with commitment deferred
  to recall. Recall MUST perform that commitment: a recalled entity carrying candidate edges
  is unified into one committed entity and the commitment is reported; a candidate plane
  that no recall path reads fails the dead-weight check (FR-024).
- **FR-019**: Disjoint sibling concepts, or a pack-supplied veto rule, MUST prevent both
  merge and candidate.
- **FR-020**: Candidate generation MUST use blocking keys and a vector index over entity
  definition embeddings; a full-table comparison MUST NOT occur.
- **FR-021**: Entity type MUST be stored as a histogram of observations, not fixed at first
  sight.

**Packs**

- **FR-022**: In this slice a pack MUST be able to supply: concepts (classes with
  definitions and is-a edges, iterated lazily), predicates (named relations with
  definitions, optional domain/range, functional flag), extraction guidance (a label set
  that replaces rather than unions, a prompt addendum, a hygiene preset, thresholds), an
  identity veto rule, and optionally a parser and a deterministic extractor. A lexicon of
  surface forms and senses, a channel that both populates and collects, a symbol expansion
  over the pack's own structure, and a pack-supplied retrieval profile are specified with
  the dialogue pack, where each gets its first caller; the protocol MUST NOT declare them
  before then.
- **FR-023**: Frames MUST be expressed as concepts with role predicates. No frame symbol
  kind exists in the core, and no frame-specific vertex or edge type is stored.
- **FR-024**: A pack that writes a plane no retrieval path reads MUST be detected by the
  dead-weight check and MUST NOT ship.
- **FR-025**: Conflicting concept or predicate identifiers across simultaneously loaded
  packs MUST be detected at load time and reported, not silently overwritten.

**Inputs shipped in this slice**

- **FR-026**: A parser MUST read agent transcript records, keeping user and assistant
  records that are not internal bookkeeping, walking their content blocks (text, thinking,
  tool use, tool result), deduplicating by record identifier, and retaining timestamp,
  working directory, branch and side-chain flag as metadata.
- **FR-027**: The transcript parser MUST be tolerant and versioned: unrecognised record or
  block shapes are counted and skipped, never fatal and never silent.
- **FR-028**: Tool results MUST NOT become segments; they MUST be linked as sources that a
  fact can cite.
- **FR-029**: A code pack MUST segment source files per function and class, set the
  structure path to the qualified symbol, and supply concepts for function, class, module
  and file and predicates for calls, imports, defines and tests.
- **FR-030**: Code facts MUST be produced deterministically, without a machine-learning
  model.

**Agent loop integration**

- **FR-031**: Recall into the agent loop MUST be symbol-keyed: injection happens only for
  symbols that resolve, through the same identity path used at ingest, and nothing is
  injected when nothing resolves. A per-prompt fuzzy budget MUST NOT be used.
- **FR-032**: Session start MUST inject a working-context block for the working directory
  containing open decisions and current values of functional predicates. An open decision
  is a decision fact with no superseding fact on the agent pack's functional
  `decision_status` predicate.
- **FR-033**: Prompt submission MUST inject facts attached to symbols resolved from the
  prompt, newest first.
- **FR-034**: An edit or write MUST inject facts about the target path before it runs; a
  read or search MUST inject facts about the symbols it observed after it runs.
- **FR-035**: Session stop and pre-compaction MUST ingest transcript records new since the
  last checkpoint.
- **FR-036**: The integration process MUST use the existing lightweight client against the
  running memory service and MUST NOT load machine-learning dependencies.
- **FR-037**: A forget operation MUST mark a named fact or entity as forgotten and exclude it
  from every recall path. The record itself is retained as a tombstone so merge logs and
  provenance stay replayable; physical purge is out of scope for this slice.
- **FR-038**: Shipped guidance MUST tell the agent when to record and when to retrieve.

**Measurement**

- **FR-039**: All panel rows MUST run through one harness and MUST be fed turn by turn where
  the data has turns.
- **FR-040**: Every run MUST report identity pairwise precision and recall on a labelled
  sample, fact precision via the existing triplet audit extended beyond dialogue, and a
  dead-weight check that fails the run when a written vertex or edge type has no reader.
- **FR-041**: A baseline for every panel row and internal metric MUST be recorded against
  the old core before any new core code is written. Unmodified means the core's behaviour
  is unchanged; the evaluation harness and read-tracking instrumentation of the store MAY
  be added before the baseline, since the dead-weight check needs them on the baseline
  itself. The baseline MUST record a per-row limit, and every gate run MUST use the same
  limit. A row introduced after the baseline is gated against its own first recorded run,
  named as such.
- **FR-042**: Merging MUST require no regression beyond recorded run noise on any panel row
  that runs on the new core, and no drop in identity precision. No single row is the
  headline. Until the dialogue pack exists, the dialogue rows (LoCoMo, LongMemEval, BEAM)
  report not-run on the new core and their old-core baseline stays as the reference for the
  dialogue-pack specification; this slice's cutover gate is identity precision, the dead-weight
  check, and the two-pack, schema-refusal and concurrent-ingest scenarios; the repository
  row joins the gate once it exists (FR-044), gated against its own first recorded run and
  named as such.
- **FR-043**: The panel MUST report the local extraction path and the language-model
  extraction path as separate columns, both behind one extractor contract, with all
  vocabulary supplied by packs rather than module constants. Both paths stay first-class in
  this slice and both are gated on the panel; neither is deferred to a later slice.
- **FR-044**: A new panel row MUST be built from this repository's own agent transcripts,
  with gold answers derived from git history and the transcripts, covering at least decision
  location, change rationale, and which test guards a given symbol.

**Cutover**

- **FR-045**: The cutover MUST be structural and complete: sound existing modules are
  carried over unchanged by copy; everything else in the old core is deleted. No old write
  path stays alive and no old database is migrated.
- **FR-046**: The dropped planes MUST no longer be written or declared: topics, graph
  ranking and community columns, the frame-element mirror, all frame-specific types,
  free-string relations, and session and turn as their own types (a session is a source, a
  turn is a segment).
- **FR-048**: Between cutover and the identity story, identity MUST be exact-key only: no
  merge and no candidate is produced. This is precision-first by construction, and the
  cutover report MUST state it.

### Key Entities

- **Source**: one ingested thing — a file, a session transcript, a URL. Carries identity,
  location, media type, content hash, import time, namespace.
- **Segment**: the unit of embedding and extraction and the only evidence. Carries text,
  kind (prose, turn, code, table, tool call, tool result, citation), structure path, byte
  range, optional role, observation time and whether it was inferred, embedding, extractor
  version. Belongs to a source; ordered by a next relation.
- **Entity**: a thing referred to. Carries name, normalised name, type histogram, state.
  Reached from segments by mention edges bearing span and confidence; related to other
  entities by merge-log and same-as-candidate edges.
- **Concept**: a symbol of the first index kind — a class, sense or frame. Carries label,
  definition, embedding and owning pack; organised by broader edges.
- **Predicate**: a symbol of the second index kind — a named relation with a definition, a
  canonical form and a functional flag, owned by a pack.
- **Episode**: a symbol of the third index kind — a time instance. Present in the vocabulary
  but not materialised as stored vertices in this slice (FR-014).
- **Fact**: the atom of recall. Subject entity, predicate, object entity, polarity,
  modality, confidence, validity, current-ness, extractor identity and version, and links to
  the evidence segments with spans. Related to other facts by contradicts and supersedes.
- **Domain pack**: the unit of domain knowledge as data — concepts, predicates, extraction
  guidance, an identity veto rule, optional parser, optional deterministic extractor.
  Lexicon, channel, symbol expansion and retrieval profile arrive with the dialogue pack
  (FR-022). Two packs ship in this slice: the code pack and the agent pack.
- **Decision**: a concept of the agent pack — a stated choice in a user or assistant
  segment, extracted under the pack's `decision` label and linked to its session, file and
  symbol entities. It carries the functional `decision_status` predicate; an open decision
  is one with no superseding `decision_status` fact.
- **Panel**: the yardstick — one harness over several evaluation rows plus the internal
  metrics, with a recorded baseline and a merge gate.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Packs for at least two unrelated domains load into one namespace at the same
  time and both recall correctly, with zero domain-conditional branches in the core (grep
  for benchmark, dialogue, speaker and turn-regex terms in core modules returns nothing).
- **SC-002**: 100% of recall results are facts carrying at least one evidence reference with
  a resolvable source and byte range.
- **SC-003**: A baseline number exists, committed, for every panel row and every internal
  metric before the first line of new core code is written.
- **SC-004**: No panel row that runs on the new core regresses beyond its recorded run
  noise at merge, identity pairwise precision is greater than or equal to the baseline, and
  the dead-weight check is clean. Rows that cannot run until the dialogue pack exists
  report not-run and are gated in that specification against the old-core baseline
  recorded here.
- **SC-005**: Every stored vertex and edge type written during a panel run is read by at
  least one retrieval path; the run fails otherwise.
- **SC-006**: Identity resolution compares only blocked and vector-neighbour candidates;
  resolution time per batch does not grow linearly with total entity count over at least a
  tenfold increase in namespace size: per-batch time at ten times the namespace size is
  under twice the base.
- **SC-007**: Every merge performed at ingest can be reconstructed and undone from its log
  alone.
- **SC-008**: Re-ingesting an unchanged source or an already-seen transcript record adds no
  new segments and no new facts.
- **SC-009**: Recall in the agent loop injects nothing for a prompt that resolves to no
  known symbol, in 100% of such cases.
- **SC-010**: The in-loop integration process starts and answers within the agent's hook
  budget without loading machine-learning dependencies: recall-verb p95 under 1 s and a
  hard ceiling of 5 s, measured by the hook-latency scenario.
- **SC-011**: The repository-transcript panel row runs end to end and reports the same
  metric shape as every other row, with at least three question kinds represented.
- **SC-012**: A forgotten fact or entity appears in no subsequent recall, while its tombstone
  and merge log remain readable for replay.
- **SC-013**: Opening a database written by an older schema fails immediately with a stated
  version mismatch, in 100% of attempts.
- **SC-014**: A malformed or drifted transcript record never aborts an ingest and is always
  reflected in a named counter.
- **SC-015**: Two sessions ingesting into one namespace at the same time produce the same
  entity and fact set as running them one after the other, and neither run fails.

## Assumptions

Owner decisions taken as binding for this spec (not to be re-litigated during planning):

- Memory for an agent in a loop comes first; corpora second; benchmarks are clients, not the
  shape of the system.
- The domain lives entirely in packs. Parsers, vocabularies, extraction guidance and
  retrieval profile are data; the two layers, the traversal, the storage shape and rank
  fusion are fixed.
- The atom of recall is a fact with evidence. A drop on the dialogue benchmark row is an
  accepted cost.
- Identity is precision-first with deferred commitment; the existing floor-plus-margin
  acceptance test and the graph store's vector index are the mechanisms.
- Frames are concepts with role predicates; the existing frame data source becomes a loader
  emitting concepts and predicates.
- In an agent session, prompts, assistant messages and the agent's own writes are ingested;
  tool results are not, but are citable sources. Sub-agent transcripts follow the same rule,
  marked as side chains.
- The graph store and its one-database-per-namespace model stay; only their use changes.
- Cutover is big-bang: no shims, no flags, no migration. Modules judged sound (the
  language-model decoder and its response gates, temporal reasoning, rank fusion and frame
  boost mathematics, the ontology loader and semantic-web stack, the memory service and its
  lightweight client, the evaluation harness commons) are carried over by copy; the rest of
  the old core is deleted at cutover.
- The dialogue pack, including the port of current dialogue behaviour, is the next
  specification and is out of scope here. The paper and PDF domain is later still. The core
  must accept both without a core change.
- All five clarification answers were confirmed by the owner on 2026-09-08 and are
  binding: the one-anchor-per-segment time model (FR-013), episodes not materialised in
  this slice (FR-014), both extractors first-class and panel-gated (FR-043), one writer per
  namespace (FR-047), and forget as a tombstone (FR-037). FR-013 and FR-043 remain the two
  the owner marked provisional: a failing temporal or knowledge-update row overturns the
  first, and the side-by-side extractor columns decide the second.
- The code parser uses the standard library's `ast` module for Python; tree-sitter arrives
  behind the same parser protocol with the first non-Python pack.
- Heavy dependencies remain in the core package; the in-loop integration process and the
  service client remain dependency-light.
- Development conventions (protocols over inheritance, dataclasses beyond two parameters,
  full type hints, no god functions, shortest working change) are inherited from the
  repository's own rules and are not restated as requirements here.

## Out of Scope

- The dialogue pack and the port of current dialogue-specific behaviour.
- Paper and PDF parsing, citation extraction, and a papers evaluation row.
- Any migration of existing databases or interoperability with the old schema.
- Query planning beyond symbol-keyed recall (multi-hop planning, aggregation, interval
  questions) unless a panel row demands it.
- Repackaging heavy dependencies into optional extras.
