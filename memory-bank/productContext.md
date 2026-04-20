# Product Context

## Why This Project Exists
Most teams that want internal knowledge-graph RAG have to assemble a fragile bespoke stack: an ETL pipeline, a graph database, a vector store, a RAG service, and a UI — each written from scratch and tightly coupled. GraphKnows provides this as a ready-made, well-documented open-source template that works out of the box and is designed for customization.

## Problems It Solves
1. **Cold-start problem** — spinning up a KG-RAG system from zero takes weeks; this template takes 30 minutes.
2. **Monolithic pipelines** — most open examples bake assumptions (schema, document format, LLM provider) into every layer. GraphKnows isolates each concern.
3. **Opaque retrieval** — users can't see *why* an answer was generated. GraphKnows exposes the full reasoning trace, tool calls, and subgraph in the chat UI.
4. **Hard-to-extend schemas** — adding a new node type currently requires editing 5+ files. The new plugin system makes it a single file.

## How It Works
1. **Ingest:** Drop documents (PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, images) into the Document Manager. The graphgen ETL service parses → chunks → extracts entities → resolves → embeds → uploads to Neo4j + pgvector.
2. **Explore:** The Graph Explorer visualizes the knowledge graph with node filtering and subgraph expansion.
3. **Chat:** The Agent Chat interface sends queries to the graphrag service, which runs a multi-step agent loop (decompose → tool calls → synthesize), returning streamed answers with inline citations and a reasoning trace.
4. **Monitor:** The Analytics dashboard shows document processing status, entity distributions, and ingestion trends.

## User Experience Goals
- **Developer first:** the template should feel like a well-maintained OSS project, not a demo. Clear CONTRIBUTING.md. Clean extension points.
- **Transparency:** every chat answer is explainable — citations link to exact chunks; tool calls are visible in a collapsible timeline.
- **Fast feedback loops:** document upload shows live per-file progress and a real-time ingestion log stream.
- **Composable:** users should be able to swap the LLM provider, add a new entity type, or replace the embedding model by touching one config file.

## Target Users
- **Primary:** engineers at mid-size companies building internal knowledge bases (HR docs, engineering wikis, contract libraries).
- **Secondary:** researchers and indie developers exploring knowledge-graph RAG architectures.
- **Tertiary:** teams that need a starting point to build a specialized vertical (e.g., a civic data platform, a legal research tool).
