# Project Tasks & Roadmap

This file tracks the detailed implementation status of the "Noa" Agentic Memory Graph POC.

## Infrastructure & Core

- [x] **Dual-DB Setup**: Deploy FalkorDB (Graph) + Postgres (Vector) via Docker Compose.
- [x] **Hybrid Bridge**: Atomic writes via `src/kg/falkordb/uploader.py` and `src/kg/falkordb/postgres_store.py`.
- [ ] **Queued Worker**: Background worker for sequential processing (Code missing/refactored, need to verify).
- [x] **Health Checks**: Pre-flight connectivity checks for databases.
- [x] **Run Management**: Unique timestamped output directories for each pipeline run.

##  Ingestion Pipeline

- [x] **CSV Parser**: Generic tab-separated parser for LifeLog data (`src/kg/graph/parsers/generic.py`).
- [x] **Schema Mapping**: Map CSV columns to Graph Nodes (Person, Place, Context).
- [x] **Episode Segmentation**: Group segments by time gaps (>5min) into Episodes.
- [x] **Bulk Loading**: Optimized batch insertion for FalkorDB.
- [x] **Multilingual Support**: Detect and route [zh] vs [en] content to specific embedding models.

## Knowledge Graph Features

- [x] **Entity Resolution**: Merge similar entities (e.g., location fuzzy match).
- [x] **Community Detection**: Leiden algorithm for Topic/Subtopic clustering.
- [x] **Graph Pruning**: Remove disconnected or noisy nodes.
- [ ] **Image Context**: Extract `(Image)-[:DEPICTS]->(Context)` relationships.
- [ ] **Triplet Extraction**: Extract structured facts like `(Person)-[:SPEAKS_AT]->(Place)`.

## Retrieval & Agent

- [x] **Hybrid Search**: Combined Vector Search (Postgres) + Graph Traversal (FalkorDB).
- [x] **Latency Profiling**: Detailed timing logs for all retrieval stages.
- [x] **Context Builder**: Assemble "Time + Graph" context window for the LLM.
- [x] **Dockerization**: Containerized API service.
- [ ] **Golden Query Validation**: Verify accuracy against "What did I eat...?" type queries.

## 🚀 Production Readiness (New)

- [x] **Dependency Management**: Add `pytest` and other dev dependencies to `requirements.txt`.
- [x] **Security**: Externalize hardcoded secrets in `postgres_store.py`.
- [x] **Testing**: Ensure all tests satisfy `pytest` execution.
- [x] **Code Structure**: Reconcile file structure with documentation (missing `worker.py`).


Probably this agent should have a tool to rephrase the question and workflow to query the knowledge graph multiple time before answering the question.
We should improve the debugging frontend to include these thoughts.
