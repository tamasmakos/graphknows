# Brief: Reorganise tests/ to mirror graphknows/ package structure

## Goal
Restructure `tests/` so every file lives in a subdirectory that mirrors the `graphknows/`
package layout. Delete all `test_wave3_*` names — merge their content into the right domain
file. No empty shells, no `_extended` suffixes, no implementation-cycle names.

## Working directory
`/root/graphknows`

## Target structure
```
tests/
  conftest.py              (keep, no move)
  fixtures/                (keep, no move)

  memory/
    __init__.py
    test_adapter.py        (merge: test_memory_adapter.py + test_adapter_conformance.py + memory/test_adapter.py)
    test_adapter_types.py  (keep content: test_adapter_types.py)
    test_dialog.py         (keep content: test_dialog_writer.py)
    test_extractor.py      (keep content: test_extractor.py + test_wave3_extractor.py)
    test_flush.py          (keep content: test_flush.py)
    test_ingestor.py       (merge: test_memory_ingestor.py + test_ingestor_metadata.py + test_ingestor_rel_extractor.py)
    test_tool_summarizer.py (keep content: test_tool_summarizer.py)

  retrieval/
    __init__.py
    test_memgraph_store.py  (merge: test_memgraph_store.py + test_memgraph_store_extended.py)
    test_neo4j_store.py     (merge: test_neo4j_store.py + test_neo4j_store_extended.py + test_wave3_neo4j_advanced.py + test_wave3_neo4j_adapter.py + test_wave3_neo4j_adapter_uploader.py)
    test_pgvector_store.py  (keep content: test_pgvector_store.py)
    test_recall_stm.py      (merge: test_recall_stm.py + test_recall.py + test_recall_memory_class_priority.py + test_recall_module.py + test_wave3_mocked_stores.py [recall/embed parts])
    test_recall_ltm.py      (move ltm-specific classes from test_recall.py if any)
    test_retriever.py       (merge: test_retriever_advanced.py + test_retriever_rrf.py)
    test_context.py         (keep content: test_context.py)
    test_schema.py          (keep content: test_schema.py)

  ontology/
    __init__.py
    test_ontology.py        (merge: test_ontology.py + test_ontology_extractor.py + test_wave3_ontology_extended.py)
    test_rel_extractor.py   (move rel_extractor classes from ontology_extractor)

  graphgen/
    __init__.py
    test_graph_cleaning.py  (merge: test_graph_cleaning.py + test_resolution.py + test_wave3_resolution_advanced.py)
    test_community.py       (merge: test_wave3_community_detection.py + test_wave3_subcommunities.py)
    test_feature_engineering.py (merge: test_wave3_feature_engineering.py + test_wave3_feature_eng_extended.py)
    test_summarization.py   (merge: test_wave3_summarization_core.py + test_wave3_summarizer.py)
    test_builder.py         (merge: test_wave3_builder.py + test_wave3_extraction_helpers.py + test_wave3_misc_helpers.py [builder parts] + test_wave3_extractors_pure.py)
    test_orchestrator.py    (keep content: test_wave3_orchestrator.py)
    test_utils.py           (keep content: test_graphgen_utils.py + test_wave3_pure_functions.py)
    test_data_types.py      (move TestChunkExtractionTask, TestSegmentData from test_small_modules.py)

  ingestion/
    __init__.py
    test_parsers.py         (keep content: test_parsers.py + test_wave3_parsers_extractors.py)

  mcp/
    __init__.py
    test_stdio_server.py    (keep content: test_mcp_stdio_server.py)
    test_helpers.py         (keep content: test_wave3_mcp_helpers.py)

  cli/
    __init__.py
    test_memory.py          (merge: test_cli_memory.py + test_wave3_cli_memory.py)

  package/
    __init__.py
    test_config.py          (merge: test_graphknows_package.py + test_config_settings.py + TestNodeSchema from test_small_modules.py)
    test_container.py       (keep content: test_wave3_container.py)
```

## Instructions

### Step 1 — Create all target directories with __init__.py
```bash
for d in memory retrieval ontology graphgen ingestion mcp cli package; do
  mkdir -p tests/$d
  touch tests/$d/__init__.py
done
```

### Step 2 — For each target file
For each target file listed above:
1. Read all source files that contribute content to it (listed in parentheses above)
2. Write the merged file to the target path
3. Keep ALL test classes and ALL test functions — no content may be dropped
4. Add one blank line between test classes
5. Deduplicate: if the same class name appears in multiple source files, merge the test methods into one class. If identical method names exist, keep the one with more assertions.
6. Update imports if needed (paths don't change since graphknows.* imports are all absolute)

### Step 3 — Verify
```bash
cd /root/graphknows && uv run pytest tests/ -q -m 'not integration' --tb=short 2>&1 | tail -20
```
The suite must pass with >= 1048 tests. If any import breaks, fix it — do not delete the test.

### Step 4 — Delete old files
Only after the suite passes:
```bash
git rm tests/test_wave3_*.py \
       tests/test_adapter_conformance.py \
       tests/test_adapter_types.py \
       tests/test_cli_memory.py \
       tests/test_config_settings.py \
       tests/test_context.py \
       tests/test_dialog_writer.py \
       tests/test_extractor.py \
       tests/test_flush.py \
       tests/test_graph_cleaning.py \
       tests/test_graphgen_utils.py \
       tests/test_graphknows_package.py \
       tests/test_ingestor_metadata.py \
       tests/test_ingestor_rel_extractor.py \
       tests/test_mcp_stdio_server.py \
       tests/test_memgraph_store.py \
       tests/test_memgraph_store_extended.py \
       tests/test_memory_adapter.py \
       tests/test_memory_ingestor.py \
       tests/test_neo4j_store.py \
       tests/test_neo4j_store_extended.py \
       tests/test_ontology.py \
       tests/test_ontology_extractor.py \
       tests/test_parsers.py \
       tests/test_pgvector_store.py \
       tests/test_recall.py \
       tests/test_recall_memory_class_priority.py \
       tests/test_recall_module.py \
       tests/test_recall_stm.py \
       tests/test_resolution.py \
       tests/test_retriever_advanced.py \
       tests/test_retriever_rrf.py \
       tests/test_schema.py \
       tests/test_small_modules.py \
       tests/test_tool_summarizer.py \
       tests/memory/test_adapter.py
# Also remove now-empty tests/memory/ if it only had test_adapter.py in it before
```

### Step 5 — Run ruff and final suite
```bash
uv run ruff check --config ruff.toml graphknows/ tests/
uv run ruff format --check --config ruff.toml graphknows/ tests/
uv run pytest tests/ -q -m 'not integration' --tb=short 2>&1 | tail -10
```

### Step 6 — Do NOT commit

## Acceptance criteria
- `find tests/ -name 'test_wave3_*'` returns empty
- `find tests/ -name 'test_*.py' -maxdepth 1` returns empty (no flat test files at root)
- Suite passes with >= 1048 tests, 0 failures
- ruff clean on both graphknows/ and tests/

## What NOT to do
- Do NOT drop any test class or test method
- Do NOT rename any graphknows.* import paths — those are stable
- Do NOT modify any production code under graphknows/
- Do NOT add noqa comments
- Do NOT commit
