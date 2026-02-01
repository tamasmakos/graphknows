# Knowledge Graph Generation Service

This service is responsible for transforming raw unstructured data (Life Logs, Text Documents) into a structured, queryable Knowledge Graph. It employs a sophisticated pipeline that combines lexical structure, LLM-based extraction, semantic analysis, and hierarchical clustering.

## Pipeline Process

The generation process is orchestrated by the `KnowledgePipeline` class and executes the following stages sequentially:

### 1. Lexical Graph Construction
**Goal:** Establish the temporal and structural backbone of the graph.
- **Input:** Raw files (e.g., CSV Life Logs, Text files) from the `input` directory.
- **Parsing:** specialized parsers (like `LifeLogParser`) convert content into structured data.
- **Structure Created:**
  - `DAY`: Represents the date of the content.
  - `SEGMENT`: Represents the time of day (Morning, Afternoon, Evening).
  - `EPISODE`: Represents a specific event or time block (derived from log entries).
  - `CHUNK`: Text split into manageable pieces for processing.
  - `PLACE`: Locations extracted directly from metadata.
- **Hierarchy:** `DAY -> SEGMENT -> EPISODE -> CHUNK`

### 2. Entity & Relation Extraction
**Goal:** Extract semantic knowledge from the text chunks.
- **Entity Hints:** Uses **GLiNER** (or Spacy) to generate initial entity candidates to guide the LLM.
- **Extraction:** An LLM (via LangChain's `LLMGraphTransformer`) extracts entities and relationships (`source -> relation -> target`) from each chunk.
- **Enrichment:**
  - **Coreference Resolution:** String-based algorithms resolve extracted entities to canonical names (e.g., "The President" -> "President Smith") within an episode.
  - **Node Creation:** `ENTITY_CONCEPT` nodes are created and linked to `CHUNK` nodes via `HAS_ENTITY` edges.

### 3. Semantic Enrichment
**Goal:** Add vector representations and merge semantically identical entities.
- **Embeddings:** Uses **SentenceTransformers** (e.g., `all-MiniLM-L6-v2`) to generate vector embeddings for:
  - Entities (`ENTITY_CONCEPT`)
  - Topics/Subtopics
  - Chunks, Episodes, and Segments
- **Semantic Resolution:** Identifies nodes with high cosine similarity (threshold default: 0.95) and merges them into a single canonical node to reduce duplication.

### 4. Community Detection
**Goal:** Identify high-level topics and themes.
- **Algorithm:** Uses the **Leiden Algorithm** on the `entity_relation` subgraph.
- **Hierarchy:**
  - **Communities:** Clusters of tightly connected entities.
  - **Sub-communities:** Recursive clustering within larger communities.
- **Graph Artifacts:** Creates `TOPIC` and `SUBTOPIC` nodes.
  - Entities are linked to Subtopics/Topics (`IN_TOPIC`).
  - Subtopics are linked to Parent Topics (`PARENT_TOPIC`).

### 5. Summarization
**Goal:** Generate human-readable descriptions for the detected communities.
- **Process:** Aggregates text from chunks associated with each community.
- **LLM Processing:** Generates a concise **Title** and **Summary** for every `TOPIC` and `SUBTOPIC`.

### 6. Pruning
**Goal:** Clean up noise.
- Removes edges with low weights.
- Removes isolated nodes (except vital structural nodes).
- Removes small, disconnected components to ensure graph quality.

### 7. Hybrid Storage & Upload
**Goal:** Persist the graph for retrieval.
- **FalkorDB:** Stores the graph topology (nodes, edges, properties).
- **PostgreSQL (pgvector):** Stores heavy vector embeddings (Hybrid Storage) to optimize memory usage in FalkorDB.
- **Indexing:** Creates indices on IDs and Vector Indices for semantic search.

## Usage

The pipeline is triggered via the API or CLI:

```bash
# Run via module
python -m src.main
```

Configuration is handled via `pydantic-settings` and `.env` variables.

## Graph Schema Reference

### 1. Node Types & Attributes

The graph defines a strict schema with specific attributes for each node type.

| Node Type | Description | Key Attributes | Data Types |
| :--- | :--- | :--- | :--- |
| **`DAY`** | Root node for a calendar day. | `id`<br>`date`<br>`segment_count` | `string` (DAY_YYYY-MM-DD)<br>`date` (ISO)<br>`integer` |
| **`SEGMENT`** | Time-of-day block (Morning, Afternoon, etc.). | `id`<br>`time_of_day`<br>`date` | `string`<br>`enum`<br>`date` (ISO) |
| **`EPISODE`** | A discrete event or time block from logs. | `id`<br>`content`<br>`date`<br>`global_segment_order`<br>`image_description`<br>`embedding` | `string`<br>`text`<br>`datetime`<br>`integer`<br>`text`<br>`vector(384)` |
| **`CHUNK`** | Text split for processing. | `id`<br>`text`<br>`length`<br>`pg_embedding_id` | `string`<br>`text`<br>`integer`<br>`integer` (Ref to Postgres) |
| **`ENTITY_CONCEPT`** | Extracted semantic entity. | `id`<br>`name`<br>`ontology_class`<br>`llm_type`<br>`embedding` | `string`<br>`string`<br>`string` (e.g. Person)<br>`string`<br>`vector(384)` |
| **`TOPIC`** | High-level community/theme. | `id`<br>`title`<br>`summary`<br>`community_id`<br>`pg_embedding_id` | `string`<br>`string`<br>`text`<br>`integer`<br>`integer` (Ref to Postgres) |
| **`SUBTOPIC`** | Granular sub-theme. | `id`<br>`title`<br>`summary`<br>`subtopic_local_id` | `string`<br>`string`<br>`text`<br>`integer` |
| **`PLACE`** | Physical location from logs. | `id`<br>`name` | `string`<br>`string` |

### 2. Relationships

| Source | Relationship | Target | Description |
| :--- | :--- | :--- | :--- |
| `DAY` | `HAS_SEGMENT` | `SEGMENT` | Temporal hierarchy. |
| `SEGMENT` | `HAS_EPISODE` | `EPISODE` | Temporal hierarchy. |
| `EPISODE` | `HAS_CHUNK` | `CHUNK` | Content breakdown. |
| `EPISODE` | `HAPPENED_AT` | `PLACE` | Location linking. |
| `CHUNK` | `HAS_ENTITY` | `ENTITY_CONCEPT` | Lexical mention of an entity. |
| `ENTITY` | *Dynamic* | `ENTITY` | Extracted semantic relations (e.g., `FRIEND_OF`). |
| `ENTITY` | `IN_TOPIC` | `TOPIC` / `SUBTOPIC` | Community assignment. |
| `SUBTOPIC` | `PARENT_TOPIC` | `TOPIC` | Hierarchical topic structure. |

### 3. Constraints & Indexing

**FalkorDB Indexes:**
- **Standard Indexes** (for fast lookups): Created on `id` for all node types.
- **Vector Indexes** (for semantic search):
  - `ENTITY_CONCEPT` (on `embedding` property)
  - `EPISODE` (on `embedding` property)

**Hybrid Storage Constraints:**
- **PostgreSQL**: Stores embeddings for `CHUNK`, `TOPIC`, and `SUBTOPIC` to optimize graph memory.
- **Reference**: Nodes in FalkorDB store a `pg_embedding_id` pointer instead of the raw vector.

### 4. Operational Model

#### CRUD Operations
- **Creation**: Handled exclusively by the batch pipeline (`KnowledgePipeline`).
- **Read**: Supported via FalkorDB Cypher queries and Vector Similarity Search.
- **Update**:
  - **Incremental Merge**: The pipeline supports `merge` operations to update existing nodes/edges without duplication.
  - **Reprocessing**: Re-running the pipeline on modified input will update the graph structure.
- **Delete**: Supported via `clean_database=True` flag in the pipeline config (full wipe).

#### Access Control
- **Service Level**: No internal ACLs. The service assumes a trusted backend environment.
- **Network Level**: FalkorDB and Postgres should be secured within the private network (e.g., Docker network), inaccessible to the public internet.

#### Schema Versioning
- **Implicit**: Schema is defined in code (`services/graphgen/src/kg`).
- **Validation**:
  - **Pre-upload**: The `KnowledgeGraphUploader` strictly validates and sanitizes properties before Cypher execution.
  - **Type Checking**: Internal logic ensures consistent data types (e.g., converting lists to JSON strings where necessary).
