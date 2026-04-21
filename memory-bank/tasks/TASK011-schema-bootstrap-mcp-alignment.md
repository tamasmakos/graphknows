# TASK011 — Single Workflow + Schema Bootstrap + MCP Alignment

**Status:** Completed  
**Added:** 2026-04-21  
**Updated:** 2026-04-22  
**Phase:** A — Cleanup & Unification (required TASK010 complete)

---

## Summary
Bootstrapped Neo4j schema at graphgen startup, added index verification in graphrag lifespan, rewired MCP server to use `run_agent`, dropped pgvector/postgres from both services and docker-compose, updated `.env.example`.

## Completed Subtasks
- ✅ 11.1: Resolved pgvector fate → DROPPED (Neo4j Community 5.11+ covers all vector indexes)
- ✅ 11.2: Resolved MCP scope → internal dev tool only
- ✅ 11.3: Created `services/graphgen/src/kg/neo4j/schema_bootstrap.py`
- ✅ 11.4: Wired `bootstrap_schema` into graphgen lifespan
- ✅ 11.5: Added index verification in graphrag lifespan (SHOW INDEXES + warn if missing)
- ✅ 11.6: Rewired MCP `kg_chat` and `kg_schema` to use `run_agent` + `create_driver()`
- ✅ 11.7: Removed postgres fields from graphrag settings
- ✅ 11.8: Dropped pgvector from docker-compose.yaml (moved behind `legacy-pgvector` profile); pyproject.toml and requirements.txt already cleaned in TASK010
- ✅ 11.9: Updated `.env.example` — removed POSTGRES_* vars, added NEO4J_AUTH

## Architecture Decisions
- pgvector: **DROPPED** — Neo4j Community 5.11+ unlimited vector indexes
- MCP server: **Internal dev tool only** — no auth required for MVP
- Schema ownership: graphgen bootstraps, graphrag only verifies
