# ADR 0002 — A single MemoryMode drives extraction and retrieval

Status: Accepted (2026-06-22)

## Context

Two independent settings governed behaviour: `relation_extraction_mode`
(`gliner2`/`llm_assisted`/`off`) and `retrieval_mode` (`deterministic`/`agentic`). They
had to be kept consistent by hand, and "llm-free vs llm-assisted" — the distinction users
actually reason about — was not a first-class concept.

## Decision

Introduce `MemoryMode {llm_free, llm_assisted}` (`graphknows.config`), selected by one
env var `MODE`. The legacy `relation_extraction_mode` and `retrieval_mode` are now
**derived properties** of `MemoryMode`, so existing internal consumers (flush, promotion
policy, corpus, store builders) keep working unchanged while there is a single knob.

Back-compat: if `MODE` is unset but legacy `RELATION_EXTRACTION_MODE=llm_assisted` or
`RETRIEVAL_MODE=agentic` is present (e.g. from the evaluation harness), `MemoryMode` is
derived from them. `spacy_relations` is a separate LLM-free toggle.

The `off` relation mode was dropped (entity extraction is always on in `llm_free`).

## Consequences

- One coherent switch to develop and compare both modes.
- Relation strategy seam stays the existing `IRelationExtractor` port (DSPy adapter); a
  second parallel `RelationExtractor` Protocol was considered and rejected as unused
  duplication.
