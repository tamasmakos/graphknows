# REL-002: Fix SESSION→FILE→CHUNK linkage for MCP ingest path

## Problem
When text is ingested via the MCP tool (`memory_ingest` → `MemoryIngestor.ingest_text()`),
`read_session_subgraph()` returns `chunks=[]` and `chunk_entity_edges=[]` for those sessions.
This means DSPy `extract_and_write_relations()` at flush time gets no chunks and writes 0
relation edges.

## Investigation first — find the gap before fixing

1. Read `graphknows/memory/ingestor.py` — trace `ingest_text()` carefully:
   - Does it create a SESSION node in Memgraph?
   - Does it create a FILE node linked via IN_SESSION?
   - Does it create CHUNK nodes linked via HAS_CHUNK?
   - Does it create ENTITY nodes linked via HAS_ENTITY or MENTIONS?
   - Are session_id properties set on all nodes and edges?

2. Look at the Memgraph write methods called:
   - `write_session()` or equivalent
   - `write_file_node()` or equivalent  
   - `write_chunk()` or equivalent
   - `write_entity()` or equivalent

3. Read `read_session_subgraph()` in `graphknows/retrieval/stores/memgraph.py`:
   - The chunks query: `MATCH (sess:SESSION)-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)`
   - If FILE nodes are missing, or IN_SESSION/HAS_CHUNK edges are missing, chunks will be empty.
   - The ENTITY union branch: `MATCH (e:ENTITY) WHERE e.session_id = $sid` — this is the
     fallback that catches entities even without the FILE chain.

4. After identifying the gap, run a quick live check:
   ```python
   from graphknows.retrieval.stores.memgraph import MemgraphStore
   import asyncio
   mg = MemgraphStore('bolt://172.18.0.1:17687')
   with mg._driver.session() as s:
       print(s.run("MATCH (n) RETURN labels(n), count(n) GROUP BY labels(n)").data())
   ```
   (The graphknows STM may be empty now — if so, ingest a test session first using
   the CLI: `graphknows-memory ingest "test text with Memgraph and Neo4j" --session-id test-rel002`)

## Expected fix
The SESSION→FILE→CHUNK chain must be complete in Memgraph after every `ingest_text()` call.

If SESSION or FILE nodes are not being written, add those writes.
If the edges (IN_SESSION, HAS_CHUNK, HAS_ENTITY) are missing, add them.
If session_id is not propagated to all nodes, fix the propagation.

Follow the existing write patterns in `graphknows/retrieval/stores/memgraph.py` exactly.
Do not introduce new abstraction — just ensure the chain is complete.

## Acceptance criteria
- After `graphknows-memory ingest "Memgraph and Neo4j are graph databases" --session-id test-rel002`,
  calling `read_session_subgraph("test-rel002")` returns:
  - `chunks` list with ≥1 entry
  - `chunk_entity_edges` list with ≥1 entry
  - `entities` list with ≥1 entry
- All existing tests pass (202 baseline — run canonical test command below).
- At least 1 new test in `tests/test_memory_ingestor.py` or a new file `tests/test_ingestor_subgraph.py`
  verifying that `read_session_subgraph()` returns non-empty chunks after ingest.

## Canonical test command
```bash
cd /root/graphknows
uv run pytest tests/ -q -m "not integration" \
  --ignore=tests/test_ontology.py \
  --ignore=tests/test_ontology_extractor.py \
  --ignore=tests/test_langgraph_agent_search_eu.py \
  --ignore=tests/test_shared_config.py \
  --ignore=tests/test_hybridrag_api.py
```

## Files to touch
- `graphknows/memory/ingestor.py` — fix the STM write chain
- `graphknows/retrieval/stores/memgraph.py` — fix write methods if needed
- `tests/test_memory_ingestor.py` or new `tests/test_ingestor_subgraph.py`

## Files NOT to touch
Everything else. No refactors. No style fixes.

## Working dir: /root/graphknows
## Do NOT commit.
