# Ontology Injection Investigation Report

## Scope
- Workspace: `/root/graphknows`
- Objective investigated: why `~1,841 / 5,859` Neo4j nodes have no `ontology_class`
- Method: code-path trace + live Neo4j verification

## Executive Finding
This is **not an entity-typing dropout in the current extraction path**. The 1,841 nodes missing `ontology_class` are almost entirely non-entity structural/topic nodes (chunks, segments, docs, topic/subtopic) plus a smaller legacy mixed-schema slice.

`ontology_class` is currently assigned for `ENTITY_CONCEPT` nodes only. In the current DB, `ENTITY_CONCEPT` coverage is `4,018 / 4,018` (100%).

## Neo4j Verification

### Requested query result (missing `ontology_class`)
- `CHUNK`: 938
- `SUBTOPIC`: 378
- `SEGMENT`: 311
- `Entity` (legacy label): 129
- `TOPIC`: 58
- `DOC`: 13
- `Chunk` (legacy label): 8
- `Topic` (legacy label): 6
- Total missing: `1,841`

### Coverage by node_type
- `ENTITY_CONCEPT`: total 4,018, with `ontology_class` 4,018, missing 0
- `CHUNK`: total 938, with 0, missing 938
- `SUBTOPIC`: total 378, with 0, missing 378
- `SEGMENT`: total 311, with 0, missing 311
- `TOPIC`: total 58, with 0, missing 58
- `DOC`: total 13, with 0, missing 13
- `node_type = NULL`: total 143, with 0, missing 143

### `ontology_type` check
- Neo4j query confirms `ontology_type` does not exist on any node (`0` matches).

## Root Cause(s) and Volumes

### Root Cause A (primary): Non-entity nodes are never assigned `ontology_class`
**Volume:** `1,698 / 1,841` missing nodes (92.2%)

Breakdown:
- `CHUNK`: 938
- `SEGMENT`: 311
- `DOC`: 13
- `TOPIC`: 58
- `SUBTOPIC`: 378

These nodes are intentionally created as lexical/topic structure nodes and are not passed through entity typing logic.

### Root Cause B (secondary): Legacy/mixed-schema nodes from older ingestion shape
**Volume:** `143 / 1,841` missing nodes (7.8%)

Breakdown:
- `Entity`: 129
- `Chunk`: 8
- `Topic`: 6

Observed samples show older property shapes (`name/type/label`, missing `node_type`, often missing `id`), indicating prior/alternate ingestion artifacts rather than current `ENTITY_CONCEPT` pipeline output.

## Exact Code Paths (with line references)

### Where `ontology_class` is set correctly today
1. `services/graphgen/pipeline/entity_relation/extraction.py:684`
   - `ontology_label = llm_type or gliner_label or "UNKNOWN"`
2. `services/graphgen/pipeline/entity_relation/extraction.py:688`
   - New entity node creation sets `ontology_class=ontology_label`
3. `services/graphgen/pipeline/entity_relation/extraction.py:706`
   - Existing nodes are updated with precedence `llm_type` -> GLiNER label -> default `UNKNOWN`
4. `services/graphgen/pipeline/entity_relation/extraction.py:712`
   - Missing `ontology_class` backfilled to `UNKNOWN`

### Where nodes are created without `ontology_class` (expected for non-entity nodes)
1. Lexical graph builder:
   - `services/graphgen/pipeline/lexical_graph_building/builder.py:48` (SEGMENT)
   - `services/graphgen/pipeline/lexical_graph_building/builder.py:72` (CHUNK)
   - `services/graphgen/pipeline/lexical_graph_building/builder.py:183` (DOC)
2. Topic hierarchy builder:
   - `services/graphgen/pipeline/community/subcommunities.py:26` (TOPIC)
   - `services/graphgen/pipeline/community/subcommunities.py:58` (SUBTOPIC)
   - `services/graphgen/pipeline/community/subcommunities.py:103` (fallback TOPIC)
3. Iterative orchestration path also creates doc node without `ontology_class`:
   - `services/graphgen/pipeline/iterative_orchestrator.py:204`

### Why no null-valued property appears in Neo4j
`services/graphgen/utils/graphdb/neo4j_adapter.py:36-37` removes `None` properties before upload (`_clean_props_for_neo4j`), so absent `ontology_class` is omitted entirely rather than stored as null.

## Investigation Tasks Answered

### 1) Which Neo4j node types lack `ontology_class`?
- Confirmed above: almost all missing are non-entity structural/topic nodes, plus 143 legacy unlabeled-shape nodes.

### 2) Extraction pipeline trace
- Flow: lexical graph build -> NER (GLiNER/GLiNER2/Spacy) -> DSPy/LangChain relation extraction -> segment enrichment -> add entity nodes/edges -> semantic resolution -> upload.
- `ontology_class` is assigned in `add_triplets_to_graph_for_segment` for entity nodes.

### 3) Places where `ontology_class` is set on graph nodes
- The only concrete set operations are in `services/graphgen/pipeline/entity_relation/extraction.py` (`:688`, `:706-713`).

### 4) DSPy untyped entities behavior
- DSPy extractor can surface relation endpoints from triplets and creates `nodes_data` for unseen endpoints (`services/graphgen/pipeline/entity_relation/extractors.py:453-458`).
- If `source_type/target_type` absent, defaults to `"ENTITY"` (`.../extractors.py:444-445`).
- During graph write, those become `ontology_class = llm_type or gliner_label or "UNKNOWN"` (`.../extraction.py:684`), so they are typed (even if coarsely), not left missing.

### 5) `labels.py` mapping behavior
- `services/graphgen/utils/labels.py` resolves candidate NER labels from manual config and/or ontology labels, standardizes to uppercase, and de-duplicates.
- This controls NER candidate classes, not direct Neo4j property naming.

### 6) `iterative_loader.py` role
- `services/graphgen/pipeline/iterative_loader.py` only loads/samples line-based segments.
- It does **not** upload to Neo4j and does not set `ontology_class`.
- Upload happens in `IterativeOrchestrator._upload_final_graph` via uploader (`services/graphgen/pipeline/iterative_orchestrator.py:536-545`).

## What `ontology_type` was supposed to be vs. what exists
- Current codebase does not use `ontology_type` for node assignment in the traced pipeline.
- Current active property is `ontology_class` on entity nodes.
- Neo4j validation confirms `ontology_type` is absent (`0` nodes).
- Conclusion: `ontology_type` appears to be a stale/alternative naming expectation, not part of current persisted model.

## Suspected Causes from Brief vs. Observed Reality
1. DSPy infers new entities without type -> **Partially possible**, but current code assigns fallback type (`ENTITY`/`UNKNOWN`), so this does **not** explain missing property counts.
2. Entity resolution merge drops `ontology_class` -> **Not supported by current data**; `ENTITY_CONCEPT` missing count is zero.
3. Ontology mapping returns None -> **Could affect class quality**, but not observed as missing `ontology_class` in final entity nodes.

## Recommendations (retroactive classification of currently untyped 1,841 nodes)

### A) Separate structural vs entity semantics first
- Treat `CHUNK/SEGMENT/DOC/TOPIC/SUBTOPIC` as structural/topic classes and backfill deterministic `ontology_class` values from `node_type`:
  - `CHUNK -> STRUCTURAL_CHUNK`
  - `SEGMENT -> STRUCTURAL_SEGMENT`
  - `DOC -> STRUCTURAL_DOCUMENT`
  - `TOPIC -> TOPIC`
  - `SUBTOPIC -> SUBTOPIC`

### B) Handle the 143 legacy nodes with a heuristic stack
For nodes with missing `node_type` (labels `Entity/Chunk/Topic`):
1. Use label + key-set fingerprints (`keys(n)`) to infer origin family.
2. If `type` exists, map from `type` to normalized `ontology_class`.
3. If `name` patterns indicate people/orgs/locations/policy terms, classify by regex/rules.
4. If uncertainty remains, set `ontology_class='UNKNOWN_LEGACY'` and queue for LLM-assisted reclassification.

### C) Data signals to use for reclassification (as requested)
- `llm_type` (highest confidence when present)
- `node_type` (best deterministic signal for structural nodes)
- `name` patterns (capitalization, suffixes, date-like strings, known org/person tokens)
- `graph_type` (`entity_relation`, `lexical_graph`, `topic`) to split semantic vs structural pipelines

### D) Monitoring guardrail
- Add pipeline/DB assertions that report coverage separately for:
  - entity nodes expected to have semantic ontology class
  - structural/topic nodes where a structural class may be optional or deterministic

This prevents future false alarms caused by mixing node families in one missing-property metric.
