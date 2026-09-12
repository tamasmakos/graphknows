# Contract — Python API

**Surface**: what `pip install graphknows` exposes. This repository is a package installed
by strangers (Constitution VII), so everything below is a compatibility promise.

Types referenced here are defined in [../data-model.md](../data-model.md).

---

## The facade

```python
class Memory:
    def __init__(
        self,
        *,
        namespace: str = "default",
        packs: Sequence[DomainPack] = (),
        mode: MemoryMode | str = "llm_free",
        settings: GraphKnowsSettings | None = None,
    ) -> None: ...

    async def ingest(self, source: Source | str | Path) -> IngestReport: ...
    async def recall(self, query: str, *, budget: RecallBudget | None = None) -> RecallResult: ...
    async def forget(self, *, fact_id: str | None = None, entity_id: str | None = None) -> IngestReport: ...
```

**Rules the constructor enforces**

- `packs` is the *only* way domain knowledge enters. There is no `enable_x` flag ladder and
  no channel toggle in settings that selects a domain behaviour (FR-003, audit Y3).
- Loading is all-or-nothing. A concept URI or predicate id claimed by two packs raises
  `PackConflictError` naming both packs and the key; no pack is left loaded (FR-025).
- Packs for unrelated domains coexist in one namespace with no cross-pack interference
  (FR-004). A pack's parser is registered by MIME; its concepts and predicates are namespaced
  by URI, not by load order.
- Opening a namespace whose `SCHEMA_STAMP` differs from the running schema raises
  `SchemaVersionMismatchError` immediately. No migration, no shim, no flag (FR-005).

**Rules the methods enforce**

- `ingest` deduplicates on `Source.content_hash` **before any model runs** (FR-007) and
  serialises per namespace: concurrent calls queue on a per-namespace lock and report their
  wait as `ingest_queue_wait_ms` (FR-047). `recall` takes no lock.
- `recall` returns facts with evidence, never segments (FR-009, SC-002), bounded by an
  explicit budget (FR-010), with `no_evidence=True` rather than an empty list when the pool
  is empty or below the floor (FR-015).
- `forget` tombstones; it never deletes. The merge log and provenance stay replayable
  (FR-037).
- Every method returns its counters. No path logs a skip and returns success (Principle V).

---

## Protocols

Three, all `typing.Protocol`, all duck-typed against what already exists.

### Parser

```python
class Parser(Protocol):
    mimes: frozenset[str]
    version: str
    def parse(self, source: Source, data: bytes) -> Iterable[Segment]: ...
```

- Registered MIME-keyed; a pack supplies one, or `Memory(parsers=[...])` overrides.
- Byte offsets are exact, never searched for. A parser that cannot locate a segment does not
  emit it (audit I4).
- Every segment gets exactly one `observed_at`; falling back to `Source.imported_at` sets
  `observed_at_inferred` and increments `time_anchor_inferred` (FR-013).
- Unrecognised input shapes are counted and skipped, never fatal and never silent (FR-027).
- Yielding zero segments for a non-empty source increments `files_zero_segments`.

**Shipped in this slice**: `text` (existing, dialogue heuristics removed), `code` (stdlib
`ast`; one segment per function and class, `path` = qualified symbol — FR-029),
`transcript` (agent transcript JSONL; tolerant and versioned — FR-026, FR-027).

### Extractor

```python
class Extractor(Protocol):
    name: str
    version: str
    def extract(self, segment: Segment, pack: DomainPack) -> tuple[list[Mention], list[Fact]]: ...
```

Both paths sit behind this one contract and both stay first-class in this slice (FR-043):

| Implementation | Path | Vocabulary |
|---|---|---|
| `LocalExtractor` | GLiNER2 entities + relation filter/verifier | entirely from `pack.entity_labels()`, which **replaces** rather than unions (FR-022) |
| `LLMExtractor` | the existing `LLMDecoder` + `ResponseGates` | `pack.prompt_addendum()` |
| `CodeExtractor` | stdlib `ast` walk, no model at all (FR-030) | the code pack's predicates |

The panel reports the local and the LLM path as separate columns (FR-043). Neither is
deferred.

**Every fact an extractor emits** records subject, predicate (a `Predicate` id, never a
string), object, polarity, modality, confidence, validity, extractor identity and version,
and at least one evidence span into its segment (FR-008). A fact with no evidence span is
rejected at write.

### DomainPack

Full member list in [../data-model.md](../data-model.md) §5. The contract points that
matter to a pack author:

- `concepts()` and `predicates()` are **iterables**, consumed lazily. A pack must not
  materialise a large ontology into a dict (audit Y4).
- `entity_labels()` replaces the extraction label set. There is no core inventory to union
  with (FR-022).
- `lexicon()`, `channel()`, `expand()` and `retrieval_profile()` are deferred to the
  dialogue-pack specification and are not declared in this slice (FR-022). When they arrive:
  a channel that only `populate()`s is incomplete by construction (FR-024), and `expand()`
  feeds candidate generation, not only the re-ranker (FR-002).
- Frames are expressed as concepts with role predicates. There is no frame symbol kind and
  no frame-specific stored type (FR-023).

**Shipped in this slice**: the code pack. The dialogue pack is the next specification and
must load without any core change.

---

## Errors

| Raised | When |
|---|---|
| `SchemaVersionMismatchError` | Existing. Opening a namespace stamped with a different schema version (FR-005). |
| `PackConflictError` | New. Two loaded packs claim the same concept URI or predicate id (FR-025). |
| `MissingExtraError` | Existing. An optional extra's call site reached without the extra installed. Every such extra is declared and enforced by `tests/test_packaging.py` (Constitution VII). |

**Not an error**: "nothing is known about X". That is `RecallResult(no_evidence=True)`, so a
caller can tell it apart from a failed lookup (FR-015).

---

## Compatibility notes

- `models/message.py`'s closed four-value `Role` enum is **removed**. `Segment.role` is a
  free string supplied by a parser (audit I2). Callers passing `[{"role": ..., "content": ...}]`
  reach a parser like any other input.
- `graphknows.topics` and its CLI and MCP tools are removed (FR-046).
- `graphknows.ingestion.stm` is removed; a session is a `Source`, a turn is a `Segment`.
- No new third-party import is added by this slice. Any that appears later must be declared
  with a lower bound (Constitution VII).

## Verify

```bash
make gate                       # ruff, mypy, lint-imports, bandit, unit suite, coverage floor
lint-imports                    # proves core imports no pack, and hooks import no ML
pytest tests/packs tests/models -q
```
