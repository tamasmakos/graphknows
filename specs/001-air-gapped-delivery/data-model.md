# Phase 1 Data Model: Air-Gapped Delivery

This feature adds exactly one persisted entity. Everything else here is an existing concept
recorded for context, or a build-time artifact with no representation in the graph.

## New persisted entity

### `SCHEMA_STAMP`

A singleton vertex in every namespace's database, recording which schema shape created it.

| Field | Type | Rules |
|---|---|---|
| `stamp_id` | STRING | Constant literal, identical in every database. Carries a UNIQUE index — this is what makes the singleton a singleton rather than a convention. |
| `version` | STRING | Short digest identifying the graph shape. Written once at creation, never updated in place. |

**Derivation of `version`**: a digest over the DDL statement text plus the embedding dimension —
`"\n".join(stmt for _lang, stmt in _CORE_DDL)` combined with `dims`. Note `_CORE_DDL` is
`list[tuple[str, str]]`, so the statement must be projected out of each pair (research.md R3).
`dims` participates because the DDL interpolates it into the `LSM_VECTOR` indexes: a graph built at
one embedding dimension cannot serve another, and a stamp blind to `dims` would let that mismatch
through.

**Why a derived value rather than a hand-maintained number**: a hand-bumped version depends on
someone remembering, and the cost of forgetting is a stale graph served silently — precisely what
Principle V forbids. A digest cannot be forgotten.

**Placement**: added to `_CORE_DDL` in `graphknows/storage/arcadedb/_schema.py`, following that
file's existing `type → properties → index` idiom, so it is created by the same idempotent DDL pass
as every other type.

#### State transitions

The stamp has three reachable states at the moment a namespace is opened. There is no update
transition — a stamp is written once and thereafter only read.

| Observed state | Meaning | Action |
|---|---|---|
| Absent, database freshly created | New namespace | Insert the current version, proceed |
| Absent, database pre-existing | Graph written before stamping existed | **Raise** — treat as a mismatch (FR-014) |
| Present, equals running version | Compatible | Proceed silently |
| Present, differs from running version | Incompatible | **Raise**, naming both versions and the remedy (FR-013) |

**Concurrency**: two processes opening the same brand-new namespace may both attempt the insert.
The UNIQUE index on `stamp_id` rejects the loser. That rejection MUST be caught the way `_apply_ddl`
already catches benign re-creation (a lowercased message containing `already exists` / `duplicate`),
then the stamp re-read and compared. Exactly one stamp survives and neither process fails (FR-016).
This behaviour was validated against a live ArcadeDB 26.5.1 during earlier prototyping.

**Deliberately not built**: no forward-migration machinery, no automatic drop-and-recreate, no
"compatible version range". Upgrades are re-ingest by decision (FR-022); refusing is the entire
behaviour.

## Existing concepts, for context

- **Namespace** — the hard multi-tenant isolation boundary; one physical ArcadeDB database each,
  resolved by `graphknows/storage/namespace.py`. That module stays pure name resolution and gains
  no version state; a second source of truth about schema identity is exactly the kind of
  duplication Principle IV rules out.
- **State** (`raw` → `consolidated`) — the consolidation lifecycle on one graph. Unrelated to the
  schema stamp, and easy to confuse with it by name. See `CONTEXT.md`.

## Build-time artifacts (no graph representation)

- **Release artifact** — one versioned wheel, attached to a private GitHub Release. Identified by
  the version literal in `graphknows/_version.py`, which the release workflow asserts against the
  git tag.
- **Deployment** — the client-built image plus the compose stack in `deploy/`. Built from the
  release artifact by exact version, with the base image pinned by digest.
- **Baked resources** — the models and corpora written into an image layer at build time by
  `scripts/bake_models.py`, enumerated by its own `_model_ids`. No new list of model names is
  introduced anywhere; the code that owns the choice stays the only place that names them.
