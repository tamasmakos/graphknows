# Phase 1 — Data Model

**Feature**: Universal Neurosymbolic Memory Core — First Vertical Slice
**Date**: 2026-09-08

The model has two layers and nothing else, per FR-001: a **representation state**
(embeddings and the fused candidate set) and a **symbolic index** with exactly three symbol
kinds. Everything below is one of the two, or the traversal between them, or the storage
shape.

Vocabulary is Tensor Brain vocabulary where a thing maps onto a TB concept, per
`CLAUDE.md` and Constitution's Distribution section: *concept index*, *predicate index*,
*episodic index*, *bottom-up labelling*, *top-down activation*.

Field types are Python; the storage projection of each entity is in
[contracts/storage-schema.md](./contracts/storage-schema.md).

---

## 1. Input plane

### Source

One ingested thing: a file, a session transcript, a URL. Never returned by recall; only
cited.

| Field | Type | Rule |
|---|---|---|
| `id` | `str` | Stable. Derived from `uri` + `content_hash`, so re-ingesting the same bytes at the same location is the same source. |
| `uri` | `str` | Absolute and resolvable — a `file://` path with a line range must reach the evidence (SC-002). |
| `mime` | `str` | Selects the parser from the registry. No other field may select a parser (FR-003). |
| `content_hash` | `str` | SHA-256 of the raw bytes. Deduplication happens on this **before any model runs** (FR-007). |
| `imported_at` | `datetime` | UTC. Fallback for a segment with no parser-supplied time. |
| `namespace` | `str` | The ArcadeDB database. One writer at a time (FR-047). |
| `meta` | `dict[str, str]` | Parser-supplied. For a transcript: session id, `cwd`, `gitBranch`. |

**Validation**: `content_hash` unique per namespace. A second ingest of a matching hash
increments `sources_deduplicated` and writes nothing (SC-008).

### Segment

The unit of embedding and extraction, and **the only evidence** (FR-009). Never returned as
a recall result in its own right.

| Field | Type | Rule |
|---|---|---|
| `id` | `str` | Stable across re-ingest: derived from `source_id` + `byte_range`. |
| `source_id` | `str` | → `Source`. |
| `text` | `str` | The exact bytes in `byte_range`, decoded. |
| `kind` | `SegmentKind` | `prose \| turn \| code \| table \| tool_call \| tool_result \| citation`. Closed enum in the core; a pack cannot add a kind (adding one is a core change, deliberately). |
| `path` | `str` | Structure path. Heading path for prose, qualified symbol for code, record index for a turn. |
| `byte_range` | `tuple[int, int]` | Half-open, into the source's raw bytes. Never guessed — a parser that cannot locate a segment does not emit it. |
| `role` | `str \| None` | Free string, pack-supplied. **Not an enum** — the closed 4-value `Role` is deleted (audit I2). |
| `observed_at` | `datetime` | Exactly one per segment (FR-013). |
| `observed_at_inferred` | `bool` | True when `observed_at` fell back to `Source.imported_at`; increments `time_anchor_inferred`. Never silently dated (edge case 1). |
| `embedding` | `list[float] \| None` | The representation layer. Absent until embedded. |
| `extractor_version` | `str` | Version of whatever produced the segment's annotations. |
| `meta` | `dict[str, str]` | e.g. `isSidechain` for a transcript record. |

**Relationships**: `PART_OF` → `Source`; `NEXT` → `Segment` (reading order within a source).

**Validation**: `byte_range` non-empty and within the source length. A source that yields
zero segments increments `files_zero_segments` and is reported, not silently skipped
(edge case 3).

---

## 2. Symbolic index — three kinds, and only three (FR-001)

### Concept *(concept index)*

A class, sense or frame. Frames are concepts with role predicates; there is no frame symbol
kind and no frame-specific stored type (FR-023).

| Field | Type | Rule |
|---|---|---|
| `uri` | `str` | Globally unique. Conflict across simultaneously loaded packs raises at load (FR-025). |
| `label` | `str` | |
| `definition` | `str` | Required. It is what gets embedded — a concept with no definition has no index entry. |
| `embedding` | `list[float]` | Of the **definition**. The embedding *is* the connection weight, per the Tensor Brain rule this repo builds on. |
| `pack` | `str` | Owning pack name. |

**Relationships**: `BROADER` → `Concept` (is-a); `Entity` `INSTANCE_OF` → `Concept`;
`Segment` `EVOKES` → `Concept` (bottom-up labelling).

### Predicate *(predicate index)*

A named relation. Never a free string (FR-011) — the audit's Y7/X-class defects all trace to
free-string relations.

| Field | Type | Rule |
|---|---|---|
| `id` | `str` | Pack-scoped, globally unique; conflict raises at load. |
| `label` | `str` | |
| `definition` | `str` | Required, and embedded, same rule as `Concept`. |
| `canonical` | `str` | Surface form used in rendering. |
| `functional` | `bool` | True means at most one current object per subject; drives superseding (FR-012). |
| `domain` / `range` | `str \| None` | Concept URIs. Optional; when present, a fact violating them is rejected with a counter, not written. |
| `pack` | `str` | |

### Episode *(episodic index)*

A time instance. **Present in the symbol vocabulary; not materialised as a stored vertex in
this slice** (FR-014). `SymbolKind.EPISODE` exists in the enum and nothing writes an
`EPISODE` row. Time questions are answered from `Segment.observed_at` and `Fact.valid_*`.

The reason is mechanical, not stylistic: a stored episode plane with no traversal reading it
would fail the dead-weight check on its own first panel run (FR-024, SC-005).

---

## 3. Knowledge plane

### Entity

A thing referred to.

| Field | Type | Rule |
|---|---|---|
| `id` | `str` | |
| `name` | `str` | Canonical surface. Chosen deterministically: mention count, then length, then `created_at`. |
| `name_norm` | `str` | Normalised form used for blocking. |
| `type_histogram` | `dict[str, int]` | Observation counts per type. **Never** fixed at first sight (FR-021, audit E4). `dominant_type` is a derived property, not a stored column. |
| `block_key` | `str` | `name_norm[:4]` + `|` + dominant type. Indexed. Drives candidate generation (FR-020). |
| `embedding` | `list[float]` | Of the entity's *definition* — canonical name, dominant type, one context sentence. `LSM_VECTOR` indexed. |
| `state` | `EntityState` | `active \| forgotten`. Tombstone, never deleted (FR-037). |
| `created_at` | `datetime` | |

**Relationships**:

- `Segment` `MENTIONS {surface, span, confidence, extractor}` → `Entity`. The mention layer.
  A surface form is **never** collapsed into an alias string (FR-016, audit E3).
- `Entity` `MERGED_INTO {layer, evidence, at}` → `Entity`. The merge log — replayable and
  undoable from the edge alone (FR-017, SC-007). Written **before** any structural change.
- `Entity` `SAME_AS_CANDIDATE {score, layer, at}` → `Entity`. Weak evidence; no merge until
  recall commits (FR-018).
- `Entity` `INSTANCE_OF {score, pack}` → `Concept`.

### Fact

The atom of recall (FR-010). Every recall result is one of these, with its evidence.

| Field | Type | Rule |
|---|---|---|
| `id` | `str` | |
| `subject` | `str` | → `Entity`. |
| `predicate` | `str` | → `Predicate`. Never a string label. |
| `object` | `str` | → `Entity`. |
| `polarity` | `bool` | False means asserted-negative. "did NOT observe X" stores as negative, not as asserted (audit X10). |
| `modality` | `str \| None` | Pack-supplied vocabulary, not a core constant. |
| `confidence` | `float` | |
| `valid_from` / `valid_to` | `datetime \| None` | Optional validity metadata, resolved against the evidence segment's `observed_at` (FR-013). |
| `is_current` | `bool` | Maintained by the superseding rule below. |
| `extractor` | `str` | Identity. |
| `extractor_version` | `str` | |
| `state` | `FactState` | `active \| forgotten`. |

**Relationships**: `ASSERTED_IN {span}` → `Segment` (at least one, always — SC-002);
`USES` → `Predicate`; `SUPERSEDES` → `Fact`; `CONTRADICTS` → `Fact`;
`Entity` `SUBJECT_OF` → `Fact` `OBJECT_IS` → `Entity`; `CITES` → `Source` (for a tool
result, which is a source and never a segment — FR-028).

**Validation**: a fact with no `ASSERTED_IN` edge is rejected at write, not stored orphaned.

---

## 4. State transitions

### Fact currency (FR-012)

Triggered on write, only when `Predicate.functional` is true, against the current fact for
the same `(subject, predicate)`:

| Incoming `observed_at` vs current | Incoming | Existing | Edge |
|---|---|---|---|
| Newer | `is_current = true` | `is_current = false` | `incoming SUPERSEDES existing` |
| Older | `is_current = false` | unchanged | `existing SUPERSEDES incoming` |
| Equal | `is_current = true` | `is_current = true` | `CONTRADICTS`, both ways |

Nothing is deleted in any row. Equal times are recorded as a contradiction rather than
resolved by confidence — see research.md R12.

### Forget (FR-037, SC-012)

`active → forgotten`, one direction, no return path in this slice.

- A forgotten `Fact` or `Entity` keeps its row, its merge log and its provenance.
- Every read query composes the single tombstone predicate from
  `storage/arcadedb/_sql.py`; exclusion happens **in SQL**, before the recall budget, so a
  forgotten fact never consumes a budget slot.
- Forgetting a `Segment` tombstones every fact whose only evidence was that segment
  (edge case 8), incrementing `facts_excluded_tombstoned`.

### Identity resolution (FR-017 – FR-021)

Per flush, over entities touched by the flush plus their block neighbours only:

```
candidates := block(entity) ∪ vector_neighbours(entity)      # bounded; truncation counted
   ↓
veto?      disjoint sibling concepts, or a pack veto rule     → no merge, no candidate  (FR-019)
   ↓
strong?    identical name_norm in the block ∧ agreeing concepts
                                                             → MERGE + MERGED_INTO log  (FR-017)
   ↓
weak?      shared concept via a pack, or embedding similarity
           above floor + margin (symbolic/match.py:accept)    → SAME_AS_CANDIDATE        (FR-018)
   ↓
otherwise                                                     → nothing
```

A merge never deletes: `MERGED_INTO` is written first, and the surviving entity absorbs the
mention edges. Undo is replay of the log in reverse (SC-007).

---

## 5. Domain pack

The unit of domain knowledge **as data**. The core never imports one — enforced by the
`core-knows-no-domain` import contract in plan.md, not by convention.

| Member | Type | Rule |
|---|---|---|
| `name` | `str` | |
| `concepts()` | `Iterable[Concept]` | Lazy. Large ontologies are iterated, never materialised eagerly (audit Y4). |
| `predicates()` | `Iterable[Predicate]` | |
| `entity_labels()` | `tuple[str, ...]` | **Replaces** the label set; never unioned into a core inventory (FR-022, audit X3). |
| `prompt_addendum()` | `str` | |
| `hygiene()` | `HygieneConfig` | Value object. The core ships no conversational default (audit X2). |
| `thresholds()` | `Mapping[str, float]` | |
| `parser()` | `Parser \| None` | Optional. Registered by MIME. |
| `veto(a, b)` | `bool` | Identity veto rule (FR-019). |
| `extractor()` | `Extractor \| None` | Optional deterministic extractor (FR-030). |

Deferred to the dialogue-pack specification, where each gets its first caller (FR-022): `lexicon()`, `channel()` (must both `populate()` and `collect()`, FR-024), `expand()`, `retrieval_profile()`. The protocol does not declare them in this slice.

**Validation at load** (FR-025): concept URIs and predicate ids are collected into one dict
each, keyed globally. A collision raises `PackConflictError` naming both packs and the key,
and **no** pack is left loaded.

### RetrievalProfile

`neighbor_radius`, window sizes, pool factors, date gate, frame alpha, channel toggles. A
value supplied by a pack. The RRF spine in `graphknows/ranking/rrf.py` is fixed and is not
part of the profile — FR-001 makes rank fusion core.

---

## 6. Result objects

Both carry their counters. Neither returns a bare empty collection on a failed lookup
(Principle V, FR-015).

### IngestReport

`source_id`, `segments_written`, `facts_written`, `entities_touched`, plus the ingest
counter set from research.md R9 (`sources_deduplicated`, `files_zero_segments`,
`records_skipped_unknown_type`, `blocks_skipped_unknown_type`, `records_skipped_malformed`,
`records_skipped_duplicate`, `time_anchor_inferred`, `candidates_truncated`,
`merges_committed`, `candidate_edges_written`, `merges_vetoed`, `ingest_queue_wait_ms`).

### RecallResult

| Field | Type | Rule |
|---|---|---|
| `facts` | `list[FactWithEvidence]` | Possibly empty — but see `no_evidence`. |
| `no_evidence` | `bool` | True when the pool was empty or below the floor. This is what distinguishes "nothing is known" from a lookup failure (FR-015, SC-009). |
| `budget` | `RecallBudget` | The explicit budget applied (FR-010). |
| `truncated_by` | `str \| None` | The stated ordering used when the budget cut the slice (edge case 6). Visible, never silent. |
| `counters` | `Counters` | The recall counter set from research.md R9. |

`FactWithEvidence` is a `Fact` plus its evidence segments, each with `source_uri`,
`byte_range` and `text`. **Every** recall result carries at least one (SC-002); a fact that
cannot produce one is a write-time defect, not a read-time omission.

---

## 7. What this model deletes (FR-046)

Not written, and not declared in the DDL: `TOPIC` / `IN_TOPIC`; `ENTITY.pagerank`,
`ENTITY.community_id`, `ENTITY.graph_embedding`; the `FE` / `SEMTYPE` / `HAS_FE` /
`REQUIRES_FE` / `EXCLUDES_FE` mirror; every frame-specific vertex and edge type including
the `FRAME` versus `FRAME_INSTANCE` duplication; free-string relations; `SESSION` and `TURN`
as their own types — a session is a `Source`, a turn is a `Segment` of kind `turn`;
`REL.reified`; `REL.anchor`.

`EPISODE` is not declared either, per FR-014 — the symbol kind lives in the enum only.
