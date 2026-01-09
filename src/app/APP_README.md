# Graph RAG Agent

A specialized agentic retrieval system that navigates knowledge graphs to answer complex queries. This application implements a strict hierarchical retrieval strategy to provide Large Language Models with structured, high-signal context.

## Core Components

### 1. The Agent (`src/app/agent`)
The system consists of a reasoned agent capable of using tools to interact with the Knowledge Graph.

-   **Agent Core** (`agent/core.py`): Orchestrates the interaction between the user and the available tools. It maintains conversation history and determines when to fetch new information.
-   **Tools** (`agent/tools.py`):
    -   `retrieve_knowledge_graph`: The primary tool. It exposes the advanced retrieval logic to the LLM, allowing it to "look up" information dynamically.

### 2. Retrieval Service (`src/app/services/retrieval.py`)
The heart of the application. Unlike simple vector RAG, this service performs a deterministic, multi-stage graph traversal to construct a semantically rich subgraph.

**Key Features:**

-   **Hybrid Storage Support**: Seamlessly coordinates between **FalkorDB** (graph structure) and **PostgreSQL (pgvector)** (offloaded embeddings for Chunks and Topics).
-   **Latency Profiling**: Integrated `Profiler` class monitors every stage of the pipeline, outputting real-time performance metrics to the logs.
-   **Parallel Searching**: Executes multi-index (Entity, Topic, Subtopic) searches in parallel for minimal latency.

**Pipeline Stages:**

1.  **Seed Identification**:
    *   Extracts keywords from the query using the LLM.
    *   Performs parallel Vector Search (Topic, Subtopic, Entity) and Keyword Matching (searching `name`, `id`, and `title`).
    *   Reranks candidates using a weighted score: **Vector Similarity (70%) + PageRank Centrality (30%)**.
2.  **Semantic Expansion (Peer Enrichment)**:
    *   For identified seed entities, explicitly fetches **1st-degree semantic relationships** (e.g., `SUPPORTS`, `OPPOSES`, `PART_OF`) to other entities.
    *   *Result*: Captures the "web of influence" around key entities.
3.  **Content Retrieval**:
    *   Finds **Chunks** linked to the gathered entities (`[:HAS_ENTITY]`).
    *   If Hybrid Storage is enabled, fetches chunk content from FalkorDB using pointers retrieved from Postgres vector search.
4.  **Content Hierarchy Reconstruction**:
    *   Traverses the strict temporal hierarchy: **Day → Segment → Chunk**.
    *   *Result*: Reconstructs the timeline of events with full provenance.
5.  **Contextual Neighbor Expansion**:
    *   Expands from retrieved Chunks to find *other* entities mentioned in the same context.
    *   Applies strict limits to prevent context overcrowding.
6.  **Topic Hierarchy Closure**:
    *   Ensures every Entity is connected to its **Subtopic** and **Topic** (`Entity -> Subtopic -> Topic`).
7.  **Graph Filtering & Formatting**:
    *   Uses a tailored BFS to retain the relevant connected component.
    *   Converts the graph into a structured XML-like prompt for the LLM (`<timeline>`, `<relationships>`, `<entities>`).

### 3. Infrastructure (`src/app/infrastructure`)
-   **GraphDB** (`graph_db.py`): Unified driver supporting **FalkorDB** and **Postgres**. Handles hybrid "pointer-based" hydration automatically.
-   **LLM** (`llm.py`): Wraps LangChain/OpenAI interfaces for consistent model access.

### 4. Interfaces
-   **Web Agent** (`src/app/main.py`): FastAPI server exposing `POST /chat`.
-   **MCP Server** (`mcp/`): Implements the Model Context Protocol, allowing this graph agent to be used as a resource server (e.g., for Claude Desktop).
-   **Frontend** (`frontend/`): D3.js-based visualization tool for debugging and exploring the retrieved subgraphs.

## Performance & Monitoring

The backend prints detailed profiling information for every query:
```text
[Profiler] LLM Keyword Extraction took 0.842s
[Profiler] Parallel Seed Search took 0.150s
[Profiler] Semantic Expansion took 0.045s
[Profiler] Subgraph expansion took 0.321s
```

## Running the Application

### Prerequisites
-   Python 3.10+
-   **FalkorDB** instance (populated by the ingestion pipeline).
-   **PostgreSQL** with `pgvector` extension (optional, for hybrid storage).
-   `OPENAI_API_KEY` (or compatible LLM provider) environment variable.

### Start Server
```bash
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access
-   **Web UI**: [http://localhost:8000](http://localhost:8000)
-   **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
