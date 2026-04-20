# Knowledge Graph Project

This repository implements a decoupled Knowledge Graph system consisting of two primary services:
- **GraphGen** (Generation/ETL): Parses raw data, extracts entities, and builds the graph.
- **GraphRAG** (Retrieval/API): An agentic system for querying the graph.

## 🚀 Getting Started

Follow these steps to get the application up and and running.

### 1. Prerequisites
- **Docker** and **Docker Compose** installed.
- API Keys for **Groq** and **OpenAI**.

### 2. Configuration
The application relies on environment variables for API keys and model selection.

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and fill in your credentials:
   ```env
   GROQ_API_KEY=gsk_...
   OPENAI_API_KEY=sk-...
   ```

3. (Optional) Customize the LLM models for specific tasks:
   ```env
   # Extraction: Fast model for processing large text volumes
   EXTRACTION_MODEL=llama-3.1-8b-instant
   
   # Summarization: Stronger model for community insights
   SUMMARISATION_MODEL=llama-3.3-70b-versatile
   
   # Chat: Capable model for final answer synthesis
   CHAT_MODEL=llama-3.3-70b-versatile
   ```

### 3. Start the Services
Run the following command to build and start the entire stack:

```bash
docker-compose up --build -d
```

This will start:
- **FalkorDB** (Graph Database)
- **PostgreSQL/pgvector** (Vector Store)
- **GraphGen Service** (Port 8020)
- **GraphRAG Service** (Port 8010)

### 4. Run the Ingestion Pipeline
Once the services are running, you need to populate the graph with data.

1. **Add Data**: Place your text files (`.txt`, `.csv`) in the `input/` directory at the root of the project.
   
2. **Trigger Ingestion**: The GraphGen service exposes an API to start the pipeline. Run:
   ```bash
   curl -X POST http://localhost:8020/run \
     -H "Content-Type: application/json" \
     -d '{"clean_database": true}'
   ```
   *Note: Set `clean_database` to `false` for incremental updates.*

3. **Monitor Progress**: You can view the logs to watch the extraction process:
   ```bash
   docker-compose logs -f graphgen
   ```

### 5. Chat with your Data
Once ingestion is complete, use the GraphRAG service to explore the graph.

- **Web UI**: Open [http://localhost:8010](http://localhost:8010) in your browser.
- **API Documentation**: [http://localhost:8010/docs](http://localhost:8010/docs)

---


## Architecture details

- **`services/graphgen`**: The ETL pipeline. Responsible for parsing raw data, extracting entities, and building the graph in FalkorDB.
- **`services/graphrag`**: The Retrieval API. A FastAPI-based agentic system using LlamaIndex to explore the graph and answer user queries.

## Development

To run the services in development mode with hot-reloading, you can use the development compose file (or ensure your volumes are mapped correctly):

```bash
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up
```

