# Discovery Brief: Relation Extraction → Memory Pipeline

## Working directory
/root/graphknows

## Goal
Deep discovery of the relation extraction + memory pipeline. Do NOT modify any production code. Do NOT commit. Write findings to `/root/graphknows/research/discovery_relations_memory.md`.

## What to investigate (in order)

### 1. Trace the full graphgen relation extraction path
Read these files fully, understand them:
- `graphknows/graphgen/pipeline/entity_relation/extractors.py` (DSPyExtractor)
- `graphknows/graphgen/pipeline/entity_relation/extraction.py` (GLiNER2 + LLM extraction)
- `graphknows/graphgen/pipeline/entity_relation/dspy_module.py` (the DSPy module)
- `graphknows/graphgen/pipeline/entity_relation/ontology.py` (ontology injection)
- `graphknows/graphgen/config/taxonomy.py` and `taxonomy.eu-politics.yaml` (relation types defined)

Key questions:
- What relation types are defined in the EU-politics taxonomy?
- What exactly does DSPyExtractor extract? What is the Triplet schema?
- How does ontology get injected into the DSPy prompt?
- Does GLiNER2 actually extract relations here or just entities?
- Where does the graphgen pipeline write these relations to Neo4j? Which Cypher?

### 2. Trace the memory ingest path
Read these files:
- `graphknows/memory/ingestor.py` (MemoryIngestor)
- `graphknows/memory/extractor.py` (GLiNER2EntityExtractor)
- `graphknows/retrieval/stores/memgraph.py` (what Cypher is written)
- `graphknows/ontology/rel_extractor.py` (OntologyRelExtractor — DSPy based)
- `graphknows/memory/flush.py` (SessionFlusher — STM→LTM)

Key questions:
- Does `MemoryIngestor` call `OntologyRelExtractor` at all? If not, at what point does it stop?
- Does `GLiNER2EntityExtractor.extract()` return relations or only entities? (look at what `schema.relations()` sets up vs what `extract()` reads back from `raw`)
- What does the Memgraph store actually write — which node types, which relationship types?
- What does `SessionFlusher.flush()` actually write to Neo4j — does it write semantic entity-entity edges?

### 3. Check the MCP ingest path
Read:
- `graphknows/mcp/stdio_server.py` (the MCP server)
- `graphknows/mcp_ltm.py`

Key question: when `memory_ingest` MCP tool is called, does it go through `MemoryIngestor`? Does relation extraction happen at all?

### 4. Live test — run the ingest pipeline and inspect the graph
Set up the .env and run a live extraction test.

```bash
set -a && source /root/graphknows/.env && set +a
```

Write a small Python research script at `/root/graphknows/research/discovery_live_test.py` that:
1. Injects a sample political text (3-4 sentences with clear entities and relations, e.g. "Viktor Orbán opposes Ukraine's EU membership. The European Commission, led by Ursula von der Leyen, supports the accession process. Hungary has blocked several votes.")
2. Runs `GLiNER2EntityExtractor.extract()` on it and prints entities
3. Runs `OntologyRelExtractor.extract()` on the same text with the same entities and prints relations
4. Writes to Memgraph via `MemoryIngestor` 
5. Queries Memgraph to show: all nodes created, all edges created (including types)
6. Shows the full Cypher paths

Then run the script and capture the actual output.

### 5. Check what the GLiNER2 schema.relations() setup does
In `graphknows/memory/extractor.py`, `schema.relations(self._rel_descs)` is called in `_ensure_model_loaded`.
But `extract()` only reads `raw.get("entities")`.

Check: does GLiNER2 schema API also return relations under a different key? Look at the actual GLiNER2 schema API to see if there's a "relations" key in the output dict that we're silently ignoring.

Try: in the research script, also call `self._gliner.extract(text, schema, threshold, True, True, True)` and print the FULL raw dict (not just raw.get("entities")).

### 6. Check the graphgen Neo4j write path for relations
Find where graphgen writes relation triples to Neo4j. Check:
- `graphknows/retrieval/stores/neo4j.py` — look for `extract_and_write_relations`, any MERGE for relationship types

Show the actual Cypher that writes entity-entity edges — what property names, what relationship type name.

## Output
Write a structured markdown report to `/root/graphknows/research/discovery_relations_memory.md` with:
- Section per investigation area
- Exact file:line references
- Actual raw output from the live test
- A gap analysis: what exists, what's wired, what's missing
- No speculation — only what the code actually does

Do NOT modify any production file. Research only.
