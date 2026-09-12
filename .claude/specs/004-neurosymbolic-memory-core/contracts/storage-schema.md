# Contract — Storage schema (golden)

**Store**: ArcadeDB, one database per namespace. DDL lives in
`graphknows/storage/arcadedb/_schema.py` as `_CORE_DDL`.

This replaces the current DDL wholesale. There is no migration and no compatibility path
(FR-005). Replacing the DDL changes `schema_version()`, which strands every existing
database — that is the intended, documented consequence of a big-bang cutover, not a
regression.

---

## Version stamp — carried over unchanged

Already implemented and reused verbatim:

- `schema_version(dims)` digests the DDL statement text plus the embedding dimension.
- `SCHEMA_STAMP` is a singleton, enforced by a `UNIQUE` index on `stamp_id` — a concurrent
  first connect is rejected by the index, not by a race-prone read-then-write.
- On open, a differing stamp raises `SchemaVersionMismatchError`. A **pre-existing,
  unstamped** database is treated as a mismatch, not assumed compatible.

No work is needed here beyond keeping its existing test green through the DDL replacement
(SC-013).

---

## Vertex types

```
SCHEMA_STAMP(stamp_id UNIQUE, version)

SOURCE(id UNIQUE, uri, mime, content_hash UNIQUE, imported_at, namespace, meta)
SEGMENT(id UNIQUE, source_id, text, kind, path, byte_start, byte_end,
        role, observed_at, observed_at_inferred, embedding, extractor_version, meta)
        INDEX embedding LSM_VECTOR

ENTITY(id UNIQUE, name, name_norm, type_histogram, block_key, embedding, state, created_at)
        INDEX block_key           -- blocking, FR-020
        INDEX embedding LSM_VECTOR -- ANN candidates, FR-020

CONCEPT(uri UNIQUE, label, definition, embedding, pack)
        INDEX embedding LSM_VECTOR
PREDICATE(id UNIQUE, label, definition, canonical, functional, domain, range, pack)

FACT(id UNIQUE, subject, predicate, object, polarity, modality, confidence,
     valid_from, valid_to, is_current, extractor, extractor_version, state)
```

**No `EPISODE` type.** The episode symbol kind exists in the core vocabulary and is not
materialised in this slice (FR-014) — a stored plane with no reader fails its own
dead-weight check.

---

## Edge types

```
SEGMENT -[PART_OF]->            SOURCE
SEGMENT -[NEXT]->               SEGMENT
SEGMENT -[MENTIONS {surface, span, confidence, extractor}]-> ENTITY
SEGMENT -[EVOKES {confidence, roles}]->                      CONCEPT

ENTITY  -[INSTANCE_OF {score, pack}]->                       CONCEPT
CONCEPT -[BROADER]->                                         CONCEPT

ENTITY  -[MERGED_INTO {layer, evidence, at}]->               ENTITY
ENTITY  -[SAME_AS_CANDIDATE {score, layer, at}]->            ENTITY

ENTITY  -[SUBJECT_OF]->  FACT  -[OBJECT_IS]->                ENTITY
FACT    -[USES]->                                            PREDICATE
FACT    -[ASSERTED_IN {span}]->                              SEGMENT
FACT    -[CITES]->                                           SOURCE
FACT    -[SUPERSEDES]->                                      FACT
FACT    -[CONTRADICTS]->                                     FACT
```

Every property that carries meaning is **declared** in the DDL. The audit's S6 defect —
`REL.evidence`, the one load-bearing provenance property, undeclared — must not recur: a
property written by any path and absent from `_CORE_DDL` fails the schema test.

---

## Not declared, not written (FR-046)

`TOPIC`, `IN_TOPIC`; `ENTITY.pagerank`, `ENTITY.community_id`, `ENTITY.graph_embedding`;
`FE`, `SEMTYPE`, `HAS_FE`, `REQUIRES_FE`, `EXCLUDES_FE`; `FRAME`, `FRAME_INSTANCE`,
`FRAME_EVOKED_IN`, `EVOKES`-as-frame, `EVOKED_BY`; `SESSION`, `TURN`; free-string `REL`;
`REL.reified`, `REL.anchor`; `CHUNK.speaker`, `CHUNK.kind` as a conversation spine; `FILE`
as a second parallel source type.

A session is a `SOURCE`. A turn is a `SEGMENT` of kind `turn`. A file is a `SOURCE`.

---

## The tombstone filter (FR-037, SC-012)

One SQL predicate fragment, defined once in `graphknows/storage/arcadedb/_sql.py`:

```sql
state <> 'forgotten'
```

Every read query composes it. Exclusion happens **in SQL, before the recall budget**, so a
forgotten fact cannot consume a budget slot and silently shrink an answer.

Enforcement is a test over the store's read-method registry asserting every query text
contains the fragment — not an audit by eye, which is how the eleventh read path forgets it.

---

## The dead-weight check (FR-024, FR-040, SC-005)

Two sets, diffed at the end of a panel run:

- **Written** — every vertex and edge type in `_CORE_DDL`, `SELECT count(*)` against the
  run's namespace; non-zero means written.
- **Read** — each store read method declares the types it touches as a `frozenset`; the
  store accumulates them into a per-connection `read_types` set as queries execute; the
  harness snapshots it.

`written - read` must be empty. Non-empty **fails the run** and names the offending types.

This is a runtime observation, not a static scan, deliberately: the audit's WordNet plane
survived because its reader was deleted while the write path stayed, and a grep for the type
name still hit the writer.

## Verify

```bash
pytest tests/storage -q
python -m evaluation deadweight --namespace <run-namespace>   # exits non-zero on a dead plane
```
