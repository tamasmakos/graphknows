# GraphGen: Knowledge Graph Generation Pipeline

The `graphgen` service is a high-performance ETL pipeline that transforms raw unstructured data into a structured Semantic Knowledge Graph.

## Key Architectural Features

- **Diamond Standard Isolation**: Strictly decoupled from the retrieval service.
- **Dependency Injection**: The `KnowledgePipeline` orchestrator accepts its infrastructure (Uploader, Embedder) as injected dependencies.
- **Strict Configuration**: Uses `pydantic-settings` to load configuration from environment variables, eliminating fragile YAML parsing.
- **Hybrid Storage**: Offloads heavy embeddings to **PostgreSQL (pgvector)** while maintaining graph topology in **FalkorDB**.

## Module Structure

```
services/graphgen/src/
├── main.py               # Composition Root (Wiring & Entrypoint)
└── kg/
    ├── pipeline/
    │   ├── core.py       # KnowledgePipeline class (Orchestrator)
    │   └── iterative.py  # Incremental update logic
    ├── config/
    │   ├── settings.py   # Pydantic Settings (Primary)
    │   ├── schema.py     # Pydantic Models for internal types
    │   └── compat.py     # Legacy config.yaml adapter
    ├── graph/            # Extraction, resolution, and pruning logic
    ├── falkordb/         # FalkorDB & Postgres storage drivers
    ├── community/        # Leiden community detection
    └── summarization/    # LLM-based community summaries
```

## Running the Pipeline

### 1. Configuration
Set the required environment variables in a `.env` file at the service root:

```env
FALKORDB_HOST=falkordb
FALKORDB_PORT=6379
OPENAI_API_KEY=sk-...
INPUT_DIR=/app/input
```

### 2. Execution
Run the pipeline as a module from the `src` directory:

```bash
python -m src.main
```

## Pipeline Stages

1. **Lexical Graph Construction**: Build document hierarchy (DAY/SEGMENT/CHUNK).
2. **Entity Extraction**: Extract triplets and metadata using LLMs.
3. **Pruning**: Remove low-value or redundant nodes.
4. **Merge & Upload**: Incrementally merge the local graph into the FalkorDB database.
5. **Entity Resolution**: Consolidate similar nodes based on semantic and text similarity.
6. **Community Detection**: Cluster the graph into topics using the Leiden algorithm.
7. **Summarization**: Generate hierarchical summaries for the detected topics.