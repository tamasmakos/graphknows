# Knowledge Graph Generation Pipeline

This module implements a comprehensive, **format-agnostic** pipeline for generating, analyzing, and storing semantic knowledge graphs from text documents.

## Key Architectural Updates

-   **Hybrid Storage (FalkorDB + Postgres)**: To handle large-scale vector data efficiently, the pipeline now offloads heavy embeddings for `CHUNK`, `TOPIC`, and `SUBTOPIC` nodes to **PostgreSQL (pgvector)**. FalkorDB maintains the graph structure and metadata, with `pg_embedding_id` pointers to the vector store.
-   **Pre-flight Health Checks**: Automatically verifies connectivity to FalkorDB and PostgreSQL before starting the ingestion process to prevent late-stage failures.
-   **Iteration Tracking**: Every pipeline run is encapsulated in a unique, timestamped directory (e.g., `output/run_20260108_142322`), ensuring experimental traceability.
-   **Native GDS Integration**: Uses FalkorDB's native Graph Data Science procedures (e.g., `algo.pageRank`) for high-performance centrality calculations.

## Pipeline Stages

The pipeline executes the following stages in order:

1.  **Lexical Graph Construction**:
    *   Processes input text files using configurable parsers.
    *   Extracts **Day/Date**, **Segment**, and **Chunk** nodes.
    *   Establishes structural relationships: `HAS_SEGMENT`, `HAS_CHUNK`.

2.  **Entity Extraction**:
    *   Uses LLMs (Groq/OpenAI) to extract **ENTITY_CONCEPT** nodes and semantic relationships from text chunks.
    *   Resolves coreferences and links entities to chunks via `HAS_ENTITY`.

3.  **Embedding Generation**:
    *   Generates vector embeddings for all node types.
    *   **Entities**: Embedded using name + type context.
    *   **Chunks/Topics**: Prepared for hybrid offloading to Postgres.

4.  **Semantic Entity Resolution**:
    *   Merges similar entities based on embedding similarity (threshold: 0.95).

5.  **Community Detection (Topic Hierarchy)**:
    *   Applies the **Leiden Algorithm** to detect hierarchical topic structures.
    *   Generates **TOPIC** and **SUBTOPIC** nodes with parent-child relationships.

6.  **Summarization**:
    *   Generates human-readable titles and summaries for every Topic and Subtopic using LLMs.

7.  **Database Upload & Hybrid Storage**:
    *   **Postgres Offloading**: Stores `CHUNK` and `TOPIC` embeddings in `pgvector`.
    *   **FalkorDB Indexing**: Uploads the graph structure, creates vector indexes for `ENTITY_CONCEPT`, and establishes pointers to Postgres.

## Graph Schema

### Node Types

| Node Type | Description | Key Attributes |
|-----------|-------------|----------------|
| `DAY` | Date container | `name` (ISO date), `date`, `source_filename` |
| `SEGMENT` | Text segment | `content`, `sentiment`, `line_number` |
| `CHUNK` | Processing unit | `text`, `pg_embedding_id` (Postgres pointer) |
| `ENTITY_CONCEPT` | Extracted entity | `name`, `entity_type`, `pagerank`, `embedding` |
| `TOPIC` | Root community | `title`, `summary`, `pg_embedding_id` |
| `SUBTOPIC` | Child community | `title`, `summary`, `parent_topic` |

### Edge Types

| Edge Type | Source → Target | Description |
|-----------|----------------|-------------|
| `HAS_SEGMENT` | `DAY` → `SEGMENT` | Day contains segment |
| `HAS_CHUNK` | `SEGMENT` → `CHUNK` | Segment divided into chunk |
| `HAS_ENTITY` | `CHUNK` → `ENTITY_CONCEPT` | Chunk mentions entity |
| `IN_TOPIC` | `ENTITY_CONCEPT` → `TOPIC` | Entity belongs to topic |
| `PARENT_TOPIC` | `SUBTOPIC` → `TOPIC` | Hierarchy link |
| (Semantic) | `ENTITY` → `ENTITY` | Custom relations (e.g., `SUPPORTS`) |

## Usage

### Configuration (`config.yaml`)

Enable hybrid storage and health checks:

```yaml
postgres:
  enabled: true
  host: "localhost"
  user: "postgres"
  password: "password"
  database: "graphknows"

falkordb:
  host: "localhost"
  port: 6379
```

### Run Pipeline

```bash
# Automated health checks will run before start
python src/kg_main.py
```

## Output Structure

Each run generates a timestamped directory in `output/`:
- `run_metadata.json`: Full execution logs and parameters.
- `graphs/`: NetworkX and GraphML exports.
- `analytics/`: Statistics and CSV exports for external analysis.
- `visualizations/`: Interactive HTML graph maps.
