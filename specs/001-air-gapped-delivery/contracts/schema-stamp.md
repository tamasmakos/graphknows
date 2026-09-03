# Contract: Opening a Namespace

The behavioural contract at `GraphStore.ensure_schema(dims)` — the one seam every path crosses to
open a namespace's database. See [data-model.md](../data-model.md) for the `SCHEMA_STAMP` entity.

## Call sequence

```text
ensure_schema(dims)
  1. record whether the database already existed      # client.database_exists()
  2. _apply_ddl(_CORE_DDL, dims=dims)                 # unchanged, idempotent, creates the DB
  3. read the SCHEMA_STAMP singleton
  4. decide (table below)
  5. clear caches, log                                # unchanged
```

Step 1 must precede step 2, because `_apply_ddl` creates the database — after it runs, "did this
exist?" is unanswerable. This ordering is forced by `_base.py`, not a preference.

## Decision table

| `pre_existing` | Stamp | Outcome |
|---|---|---|
| false | absent | Insert current version. Proceed. |
| true | absent | **Raise** `SchemaVersionMismatchError`. Graph predates stamping; not assumed compatible. |
| either | equal | Proceed. No log line — a matching version is not news. |
| either | differs | **Raise** `SchemaVersionMismatchError`. |
| false | absent, then insert rejected as duplicate | A concurrent connect stamped it first. Re-read, then apply the rows above. Never fail for this reason. |

## `SchemaVersionMismatchError`

The `Error` suffix is required, not stylistic: ruff's `N818` is enabled via the `"N"` select group and every existing exception in `graphknows/exceptions.py` carries it. An earlier draft of this contract mandated `SchemaVersionMismatch`, which introduced a lint error and was correctly refused at validation.

New exception in `graphknows/exceptions.py`, subclassing the existing `StoreError` so callers
already handling store failures keep working.

The message MUST name four things (FR-013), because the operator recovering from this has only the
message:

1. the namespace / database
2. the version recorded in the graph
3. the version the running code expects
4. the remedy — that the namespace must be rebuilt from its original sources, and how

The remedy must reference something that actually exists. If no reset entry point is reachable by
the client, the ticket adds one or the message names the real gesture; a message pointing at a flag
that does not exist is worse than no message.

## Guarantees

- A mismatched graph is **never read from and never written to**. The refusal happens before any
  query the caller asked for.
- Refusal is total — no partial or read-only mode. There is no "compatible range".
- No migration is attempted, and nothing is dropped automatically. Destroying a client's graph to
  resolve a version disagreement is not a decision this code gets to make.
- Concurrent first-open of a fresh namespace leaves exactly one stamp and fails neither process.

## Out of scope, and worth stating

`graphknows/symbolic/ontology/rdf/store.py` has its own `ensure_schema(client, db)`, and
`graphknows/storage/__init__.py` constructs stores directly. The single-seam claim was verified for
`Memory` and the STM service only. Whether those other paths need the guard is **unverified** — the
implementer must check rather than assume, and say what they found.

## Acceptance

New file; does not exist today, so it fails collection on the tip:

```bash
pytest tests/storage/test_schema_stamp.py -q
```

Covering: fresh namespace stamps and proceeds; matching stamp proceeds; differing stamp raises with
all four message elements; pre-existing unstamped database raises; concurrent insert leaves one
stamp and no failure.
