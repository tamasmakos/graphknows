# MEM-003: entity_names always [] in MCP recall results

## Context
- Kanban: t_c5e7452d
- Working dir: /root/graphknows
- Venv: /root/graphknows/.venv
- Model: openrouter/deepseek/deepseek-v4-flash

## Problem
memory_recall MCP tool always returns `"entities": []` for every hit.
GLiNER2 extracts entities during ingest and stores them in Memgraph, but
entity names never make it into RecallResult.entity_names.

## Known facts (pre-investigated)

Entity enrichment for STM happens in `recall_stm._entity_boost()` (line ~109):
- Calls `mg_store.query_entity_chunks(session_id=..., entity_names=words, top_k=...)`
- `words` = [w for w in query.lower().split() if len(w) > 3]  ← keyword tokenization
- If chunk is found, sets `old_result.entity_names = ec.get("entity_names", [])`

The issue: `query_entity_chunks` searches Memgraph for CHUNK nodes co-mentioned with
ENTITY nodes whose name matches the query WORDS (not the entities found during ingest).
If the query words don't match entity names exactly, it returns nothing.

For LTM: `recall_ltm._neo4j_entity_boost()` line ~97 does the same via Neo4j.

## Investigation steps

1. First, read these files end to end:
   - graphknows/retrieval/recall_stm.py (especially _entity_boost)
   - graphknows/retrieval/stores/memgraph.py (query_entity_chunks)
   - graphknows/memory/ingestor.py (how entities are written during ingest)

2. Run a live trace:
   ```bash
   cd /root/graphknows && set -a && source .env && set +a && .venv/bin/python3 << 'EOF'
   from graphknows.retrieval.stores.memgraph import MemgraphStore
   import os
   mg = MemgraphStore(os.environ['MEMGRAPH_URI'])
   import asyncio
   # Check what's in Memgraph for hermes-20260524
   result = asyncio.run(mg.query_entity_chunks(
       session_id='hermes-20260524',
       entity_names=['mcp', 'hermes', 'opencode', 'graphknows'],
       top_k=5,
   ))
   print('entity_chunks result:', result)
   EOF
   ```

3. Check if ENTITY→CHUNK edges exist in Memgraph:
   ```bash
   cd /root/graphknows && set -a && source .env && set +a && .venv/bin/python3 -c "
   from neo4j import GraphDatabase
   import os
   d = GraphDatabase.driver(os.environ.get('MEMGRAPH_URI','bolt://172.18.0.1:17687'), auth=('',''))
   with d.session() as s:
       r = s.run('MATCH (e:ENTITY)-[]->(c:CHUNK) RETURN count(*) as n').single()
       print('ENTITY->CHUNK edges:', r['n'])
       r2 = s.run('MATCH (e:ENTITY) WHERE e.session_id=\"hermes-20260524\" RETURN e.name LIMIT 10').data()
       print('entities:', r2)
   d.close()
   "
   ```

## Your task based on investigation

Option A (likely correct): if ENTITY→CHUNK edges are missing from Memgraph ingest,
add them in ingestor.py where entities are written.

Option B: if edges exist but query_entity_chunks uses wrong property names, fix the query.

Option C: if keyword matching is too strict, change _entity_boost to use entity names
extracted from the KG rather than tokenized query words.

Whatever the root cause, fix it and add a test:
- tests/test_mcp_stdio_server.py or tests/test_memory_recall.py
- Test: ingest text with a known entity ("Graphknows MCP server"), recall with that entity
  name in the query, assert entity_names is non-empty in at least one hit.

## Acceptance criteria
- entity_names is populated for at least some recall results when entities match
- Test passes
- All existing tests still pass (run full suite: `pytest tests/ -x -q --ignore=tests/test_ontology.py --ignore=tests/test_ontology_extractor.py --ignore=tests/test_langgraph_agent_search_eu.py`)
- Do not commit
