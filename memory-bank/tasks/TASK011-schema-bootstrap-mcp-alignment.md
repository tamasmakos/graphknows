# TASK011 — Single Workflow + Schema Bootstrap + MCP Alignment

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** A — Cleanup & Unification (requires TASK010 complete)

---

## Original Request
After TASK010 deletes the legacy codepaths, this task ensures the surviving Neo4j codepath is fully bootstrapped: constraints and indexes exist at startup, the MCP server calls the same workflow as the REST API, and two open architectural questions are resolved (pgvector fate + MCP auth scope).

---

## Thought Process
The agent tools in `services/graphrag/src/agent/tools.py` use `CALL db.index.vector.queryNodes('chunk_embedding', ...)` and `CALL db.index.vector.queryNodes('entity_embedding', ...)`. These calls silently return zero results if the indexes don't exist — they don't throw. Similarly, without a `UNIQUE` constraint on `doc_id`, re-ingesting the same document creates duplicate nodes. Both of these are silent correctness bugs that will look like "the agent doesn't work."

The MCP server (`mcp/server.py`) currently calls `GraphWorkflow` (deleted by TASK010). After TASK010 it will be a broken import. TASK011 rewires it to `run_agent`.

### Open Questions to Resolve Before Starting
1. **pgvector**: Drop entirely (Neo4j vector indexes cover both CHUNK+ENTITY, one less container) / keep for CHUNK only / keep both. **Recommendation: drop.** Neo4j Community 5.11+ has unlimited vector indexes. pgvector adds a second source of truth and a second connection pool. If the team wants to reintroduce it later, the interface is trivial to add back.
2. **MCP server scope**: Keep it public-facing (means deciding on auth approach for it) / treat as internal dev tool only (no auth concern). **Recommendation: internal dev tool only** for MVP. Remove it from `docker-compose.yaml` `ports:` section; document it under "advanced usage."

---

## Implementation Plan

1. **Create `services/graphgen/src/kg/neo4j/schema_bootstrap.py`**:
   ```python
   async def bootstrap_schema(driver: AsyncDriver, database: str) -> None:
       """Idempotent: creates constraints + indexes if not present."""
       statements = [
           "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
           "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
           "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
           "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON c.embedding OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
           "CREATE VECTOR INDEX entity_embedding IF NOT EXISTS FOR (e:Entity) ON e.embedding OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
           "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
           "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
       ]
       async with driver.session(database=database) as session:
           for stmt in statements:
               await session.run(stmt)
   ```

2. **Wire `bootstrap_schema` into `services/graphgen/src/main.py` lifespan** — call before the `yield`.

3. **Add equivalent bootstrap call in `services/graphrag/src/main.py` lifespan** — graphrag also needs to confirm indexes exist (read-only confirmation via `SHOW INDEXES`; if missing, log warning rather than crash, since graphgen owns schema creation).

4. **Rewire `services/graphrag/src/mcp/server.py`**:
   - Remove `from src.workflow.graph_workflow import GraphWorkflow` import.
   - Replace `kg_chat` tool body with a call to `run_agent(query, _driver, database, messages)`.
   - Keep `kg_schema` and `kg_health` tools (they call Neo4j directly and are fine).

5. **Remove FalkorDB from `services/graphrag/src/common/config/settings.py`** — delete `falkordb_host`, `falkordb_port`, `graphdb_type` fields.

6. **pgvector decision** (resolve before coding):
   - If dropping: remove `pgvector` service from `docker-compose.yaml`; remove `psycopg2-binary` + `pgvector` from both services' `pyproject.toml`; remove pgvector env vars from `.env.example`.
   - If keeping for CHUNK: keep service but document clearly that CHUNK embeddings go to pgvector and ENTITY embeddings go to Neo4j.

7. **Update `.env.example`** to remove FalkorDB entries; update Neo4j + LLM section to reflect final state.

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 11.1 | Resolve: drop pgvector? | Not Started | 2026-04-21 | Decision required first |
| 11.2 | Resolve: MCP public vs internal? | Not Started | 2026-04-21 | Decision required first |
| 11.3 | Create kg/neo4j/schema_bootstrap.py | Not Started | 2026-04-21 | Idempotent, 7 statements |
| 11.4 | Wire bootstrap_schema into graphgen lifespan | Not Started | 2026-04-21 | |
| 11.5 | Add index confirmation in graphrag lifespan | Not Started | 2026-04-21 | Log warn if missing |
| 11.6 | Rewire MCP kg_chat to run_agent | Not Started | 2026-04-21 | Needs TASK010 done |
| 11.7 | Remove FalkorDB fields from graphrag settings | Not Started | 2026-04-21 | |
| 11.8 | Execute pgvector decision (drop or keep) | Not Started | 2026-04-21 | Depends on 11.1 |
| 11.9 | Update .env.example (remove FalkorDB vars) | Not Started | 2026-04-21 | |

## Verification
- `docker compose up neo4j graphgen` → after graphgen starts, run `SHOW INDEXES` in Neo4j — see 2 vector indexes + 2 fulltext indexes + 3 unique constraints.
- `POST /graphrag/chat` and MCP `kg_chat` tool both use the same agent; responses are equivalent.
- `grep -r "falkordb" .env.example` returns zero.

## Progress Log
### 2026-04-21
- Task created. Two open questions must be answered before starting subtasks 11.3+.
