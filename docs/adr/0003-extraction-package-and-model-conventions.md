# ADR 0003 — One extraction package; model conventions

Status: Accepted (2026-06-22)

## Context

Knowledge extraction was split across `graphknows.graphgen.extraction` (NER/relation
engine) and `graphknows.ingestion.ontology` (RDF metadata), and the word "ontology"
named two different things: the GLiNER label space vs. RDF/OWL ontology metadata. Data
models were scattered with duplicate names and several dead models.

## Decision

1. Consolidate the extraction engine into one package `graphknows.extraction` with
   `entities/`, `relations/`, and `ontology/` sub-packages plus `chunking`/`diagnostics`/
   `utils`. Rename the GLiNER label-space module `ontology.py` → `entities/label_space.py`
   to end the name clash. `graphknows.ingestion` keeps `domain/` and `parsers/` (ingestion
   vocabulary, not the extraction engine).
2. Canonical data contracts are Pydantic models in `graphknows.memory.models`, one per
   concept. Internal hot-path/config state uses `@dataclass` and stays with its producer.
   Dead models (`RecallQuery`, `RecallResult`, the normalized `ExtractionResult`,
   `MemoryRecord`, `RetrievalHit`) were deleted; recall output stays `list[dict]` on
   `ReturnContext` (typing it was judged churn, not value).

## Consequences

- `tach.toml` updated; extraction boundary violations removed.
- `relation_filter.py` (previously untracked) now lives in `extraction/ontology/`.
- Four pre-existing cross-module leaks remain (config→stores, query→config.adapters,
  stores→models) — deferred, not introduced by this change.
