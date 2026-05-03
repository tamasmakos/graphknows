# GraphGen: Knowledge Graph Generation Pipeline

GraphGen transforms raw documents into a structured knowledge graph through a single-run `KnowledgePipeline`, with optional analytics and Neo4j upload. Experimental iteration-based analysis remains available for thesis workflows, but it is separate from the default execution path.

## Key Architectural Features

- **Dependency Injection**: The `KnowledgePipeline` orchestrator accepts uploader and extractor dependencies via its constructor.
- **Config-driven**: Loads `config.yaml` + `.env` via `pydantic-settings` for consistent overrides.
- **Modular Pipeline**: Each major step lives under `graphgen/pipeline/` for testable, isolated logic.
- **Standard Runtime**: `graphgen.main` runs the single-pass `KnowledgePipeline` orchestrator.
- **Experimental Utilities**: The repository also includes thesis-oriented iteration-based analysis tooling, but it is not the default runtime path.

## Module Structure

```
graphgen/
├── main.py                # Entrypoint
├── orchestrator.py        # KnowledgePipeline (core orchestrator)
├── analytics/             # Reports, metrics, visualizations
├── config/                # Settings, schema, loaders
├── pipeline/
│   ├── lexical_graph_building/  # Segment/chunk graph construction
│   ├── entity_relation/         # Entity/relationship extraction
│   ├── embeddings/              # KGE/RAG embeddings
│   ├── graph_cleaning/          # Pruning + resolution
│   ├── community/               # Leiden detection + subcommunities
│   ├── summarization/           # LLM summaries
│   ├── analysis/                # Topic separation statistics
│   └── visualization/           # Plots and reports
├── utils/
│   ├── graphdb/                 # Neo4j adapter
│   └── parsers/                 # Input parsers
└── tools/                   # CLI utilities
```

## Running the Pipeline

### 1. Configuration
Set environment variables in `.env` at the repository root (examples):

```env
NEO4J_HOST=neo4j
NEO4J_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
OPENAI_API_KEY=sk-...
INPUT_DIR=/app/input
OUTPUT_DIR=/app/output
```

Adjust pipeline behavior in `config.yaml` as needed (chunk sizes, thresholds, feature flags).

### 2. Execution

```bash
python3 -m graphgen.main
```

## Pipeline Stages

1. **Lexical Graph Construction**: Build document/segment/chunk hierarchy.
2. **Entity Extraction**: Extract entities and relations using configured backends.
3. **Semantic Enrichment**: Generate embeddings and resolve duplicates.
4. **Community Detection**: Cluster entities into topics/subtopics.
5. **Summarization**: Generate LLM summaries for communities.
6. **Topic Analysis** (optional): Statistical separation tests on embeddings.
7. **Pruning**: Cleanup and simplify low-value nodes/edges.
8. **Upload**: Persist to Neo4j if configured.
9. **Artifacts**: Save GraphML and analysis outputs.

## Operational Notes

- Logging defaults to INFO and stdout. Enable debug logs by setting `debug: true` in `config.yaml`.
- See `/app/README.md` for full logging and troubleshooting guidance.