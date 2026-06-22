# GraphKnows — Domain & Architecture Context

This file is the shared vocabulary for the codebase. Use these terms exactly.

## Composition

- **Runtime** (`graphknows.runtime.GraphKnowsRuntime`) — the single, transport-neutral
  composition root. Every entry point (MCP stdio server, web Memory Mirror) builds and
  drives one Runtime, so all transports see identical behaviour. There is no other
  composition root.

## Memory model

- **STM** (short-term memory) — the per-session evidence graph + vectors on ArcadeDB,
  written during ingestion (`graphknows.memory.stm`).
- **LTM** (long-term memory) — the consolidated knowledge graph produced by promoting
  (flushing) an STM session (`graphknows.memory.flush`, `graphknows.memory.ltm`).
- **Promotion / flush** — the STM→LTM pipeline (entity resolution, MERGE, optional GDS
  + topic summarisation).
- **Recall** — retrieval over STM/LTM via the retriever; output is
  `ReturnContext.fused_results` (RRF-fused hits).

## Mode

- **MemoryMode** (`graphknows.config.MemoryMode`) — the single switch governing both
  extraction and retrieval. One env var: `MODE`.
  - `llm_free`: GLiNER entities + spaCy relations + deterministic retriever. No network
    LLM calls on the ingest/recall hot paths.
  - `llm_assisted`: additionally runs DSPy relation extraction, topic summarisation, and
    the agentic retriever.
  The legacy `relation_extraction_mode` / `retrieval_mode` are now **derived properties**
  of `MemoryMode`, not independent settings. `spacy_relations` is an independent toggle
  (LLM-free) for development.

## Extraction

- **Extraction engine** (`graphknows.extraction`) — the one package for turning text into
  graph structure:
  - `entities/` — GLiNER NER (`gliner`), surface-form `canonicalization`, and the GLiNER
    **label space** (`label_space`; formerly the confusingly-named `ontology.py`).
  - `relations/` — relation strategies: `spacy` (LLM-free dep patterns) and `dspy`
    (LLM-assisted, RDF-constrained). The relation **seam** is the `IRelationExtractor`
    port in `config.adapters.ports`; the DSPy adapter implements it.
  - `ontology/` — RDF/OWL **ontology metadata**: loading, routing, registry, and the
    post-extraction `relation_filter`. (Distinct from the entity label space — that name
    clash is why `label_space` was renamed.)
- **Ingestion domain** (`graphknows.ingestion`) — `domain/` (graph schema, taxonomy,
  normalization) and `parsers/`. This is ingestion vocabulary, not the extraction engine.

## Data models

- Canonical data contracts live in `graphknows.memory.models` as **Pydantic** models
  (one model per concept). Internal hot-path/config state may use `@dataclass` and stays
  with its producer (e.g. the raw `ExtractionResult` dataclass lives with the extractor).
  Recall output deliberately stays an untyped `list[dict]` on `ReturnContext` — it is the
  load-bearing cross-transport shape and typing it was judged churn, not value.
