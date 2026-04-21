# GraphGen: Knowledge Graph Generation Pipeline

The `graphgen` service is a high-performance ETL pipeline that transforms raw unstructured data into a structured Semantic Knowledge Graph.

## Key Architectural Features

- **Diamond Standard Isolation**: Strictly decoupled from the retrieval service.
- **Dependency Injection**: The `KnowledgePipeline` orchestrator accepts its infrastructure (Uploader, Embedder) as injected dependencies.
- **Strict Configuration**: Uses `pydantic-settings` to load configuration from environment variables, eliminating fragile YAML parsing.
- **Neo4j Only**: All graph topology and vector embeddings stored in Neo4j Community 5.11+.

## Module Structure

```
services/graphgen/src/
├── main.py               # Composition Root (Wiring & Entrypoint)
└── kg/
    ├── pipeline/
    │   └── core.py       # KnowledgePipeline class (Orchestrator)
    ├── config/
    │   └── settings.py   # Pydantic Settings
    ├── parser/           # Auto-discovered parsers (TXT/MD/PDF/DOCX/PPTX/XLSX/HTML/Image)
    ├── graph/            # Extraction, resolution, and pruning logic
    ├── neo4j/            # Neo4j driver, uploader, indexes, schema bootstrap
    ├── embeddings/       # Sentence-transformer embeddings (BAAI/bge-small-en-v1.5)
    ├── community/        # Leiden community detection
    └── summarization/    # LLM-based community summaries
```

## Running the Pipeline

### 1. Configuration
Set the required environment variables in a `.env` file:

```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme
GROQ_API_KEY=gsk_...
INPUT_DIR=/app/input
```

### 2. Execution
The service runs as a FastAPI app. Trigger the pipeline via:

```bash
curl -X POST http://localhost:8020/run
```

Or upload individual documents:

```bash
curl -X POST http://localhost:8020/documents -F "file=@myfile.pdf"
```

## Pipeline Stages

1. **Document Parsing**: Parse all files in `INPUT_DIR` using the auto-discovered parser registry (supports TXT, MD, PDF, DOCX, PPTX, XLSX, HTML, and images via pytesseract).
2. **Neo4j Upload**: Store `Document → [:CONTAINS] → Chunk` nodes immediately after parsing.
3. **Entity Extraction**: Extract entity/relation triplets from each Chunk using GLiNER + LLM.
4. **Semantic Enrichment**: Generate embeddings, resolve coreferences.
5. **Community Detection**: Cluster the graph using the Leiden algorithm.
6. **Summarization**: Generate hierarchical summaries for detected communities.
7. **Pruning**: Remove low-value or redundant nodes.

## Schema

```
(:Document {doc_id, title, source_path, created_at})
    -[:CONTAINS]->
(:Chunk {chunk_id, text, position, doc_id, embedding})
    -[:MENTIONS]->
(:Entity {entity_id, name, type, embedding})
    -[:RELATED_TO {relation}]->
(:Entity)
```