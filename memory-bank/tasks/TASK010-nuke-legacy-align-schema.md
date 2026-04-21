# TASK010 — Nuke Legacy & Align Schema

**Status:** Completed  
**Added:** 2026-04-21  
**Updated:** 2026-04-22  
**Phase:** A — Cleanup & Unification

---

## Summary
Eliminated all legacy codepaths in graphgen and graphrag: removed FalkorDB, pgvector/postgres, DAY/SEGMENT/EPISODE schema, LifeLogParser, and dead workflow/infrastructure files. Rewired `/run` to use ParserRegistry. All syntax checks pass.

## Completed Subtasks
- ✅ 10.1–10.3: Deleted `kg/graph/parsers/`, `kg/utils/health.py`, `kg/graph/parsing.py`
- ✅ 10.4: Removed `falkordb` from graphgen requirements.txt
- ✅ 10.5: Stripped DAY/SEGMENT/EPISODE from extraction.py + fixed broken imports
- ✅ 10.6: Rewrote `/run` endpoint to use ParserRegistry
- ✅ 10.7: Added `pytesseract>=0.3.10` to graphgen pyproject.toml
- ✅ 10.8: Fixed embedding model default to `BAAI/bge-small-en-v1.5`
- ✅ 10.9–10.12: Deleted graphrag workflow/, graph_db.py, postgres_store.py, graph_retriever.py, context_builder.py
- ✅ 10.13: Removed FalkorDB from graphrag requirements.txt
- ✅ 10.14: Removed postgres fields from graphrag settings
- ✅ 10.15: Updated KG_README.md
- ✅ 10.16: Updated root README.md
- ✅ 10.17: Updated APP_README.md
