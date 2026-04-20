# System Patterns

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    apps/web  (Next.js 15)                        │
│  /documents   /chat   /graph   /analytics                       │
│  └── app/api/v1/**  (Route Handler BFF)                         │
└────────────┬────────────────────────────┬────────────────────────┘
             │ HTTP/SSE                   │ HTTP/SSE
             ▼                           ▼
┌────────────────────┐       ┌──────────────────────┐
│   graphgen :8020   │       │   graphrag :8010      │
│   (ETL service)    │       │   (Agent RAG service) │
│   FastAPI          │       │   FastAPI + SSE       │
└────────┬───────────┘       └──────────┬────────────┘
         │                              │
         │         ┌────────────────────┘
         │         │
         ▼         ▼
┌────────────────────────────────────────────────┐
│              Neo4j :7687  (graph + vectors)    │
│         pgvector :5432  (CHUNK embeddings)     │
└────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────┐
│  Langfuse :3000    │
│  (observability)   │
└────────────────────┘
```

## Key Design Patterns

### 1. Declarative Schema (Single Source of Truth)
All node labels, edge types, and properties are declared in `services/graphgen/src/kg/schema.py` as Pydantic `NodeSpec`/`EdgeSpec`/`PropertySpec` models. The Neo4j uploader, index generator, and extractor all consume this spec. No label is hard-coded anywhere else.

```
CORE_SCHEMA (GraphSchema)
├── nodes: DOCUMENT, CHUNK, ENTITY
└── edges: CONTAINS, MENTIONS, RELATED_TO
```

### 2. Plugin Auto-Discovery
Any file dropped into `services/graphgen/src/kg/plugins/` that subclasses `GraphPlugin` is automatically discovered at startup via `GraphPlugin.__subclasses__()` scanning. Plugins:
- Call `schema.register_node()` / `schema.register_edge()` in their `register()` method.
- Hook into pipeline events (`on_document_ingested`, `on_chunk_created`, `on_entities_extracted`, `on_pipeline_complete`).
- Example: `plugins/topics.py` (community detection + TOPIC/SUBTOPIC nodes) — shipped but optional.

### 3. Parser Registry (Auto-Discovery)
Each document format parser subclasses `BaseParser` and declares `supported_extensions`. The `ParserRegistry` scans the `parsers/` subpackage at startup; `registry.get_parser(filename)` returns the right parser. Adding support for `.log` files = add one file.

### 4. Agent Tool Interface
Each graphrag tool subclasses `AgentTool` with:
- A Pydantic `input_schema` class (drives LLM tool-selection prompts AND FastAPI docs).
- An `async run(ctx, **kwargs) -> ToolResult` method.
- A `ClassVar[str] name` and `description`.
Tools are registered automatically. The agent loop is tool-agnostic: it calls `select_tool()` → `execute_tool()` → `reflect()`.

### 5. SSE Streaming Contract
All streaming responses flow as POST → SSE using `sse-starlette` (FastAPI side) and `fetch` + `ReadableStream` (Next.js BFF/client side). The event taxonomy is:
```
event: reasoning    → decomposition / reflection steps
event: tool_call    → ToolCall JSON
event: tool_result  → summary of tool output
event: token        → partial answer text
event: citation     → Citation JSON (emitted as citations are finalized)
event: done         → full AgentResponse JSON
event: error        → error message
```

### 6. BFF via Next.js Route Handlers
`apps/web/app/api/v1/**` are thin proxies that:
- Forward requests to graphgen or graphrag with typed `fetch` calls using `packages/types`.
- Handle SSE passthrough via `ReadableStream` piping.
- Own conversation history (SQLite local store).
- Apply correlation IDs and request-level auth stubs.
No separate Python BFF service exists.

### 7. Idempotent Document Ingestion
1. Compute `sha256(raw_bytes)` → `content_hash`.
2. `MERGE (d:DOCUMENT {content_hash: $hash})` — creates or matches.
3. If already `complete`, skip (or re-ingest if `force=true`).
4. `chunk_id = f"{doc_id}:{position}"` — stable across re-runs.
5. Entity resolution happens globally after all chunks of a document are processed.

### 8. Structured Citations (Zero Ambiguity)
Every assistant answer has `[n]` markers in the text. Each marker indexes into `citations: list[Citation]` where `Citation.chunk_id` is the stable FK to a CHUNK node in Neo4j. No citation → no claim.

### 9. Langfuse Tracing Hierarchy
```
Trace (per request, id = correlation-id)
├── Span: decompose_query
├── Span: iteration_1
│   ├── Span: select_tool
│   ├── Span: tool.search_chunks
│   └── Span: reflect
├── Span: iteration_2
│   └── ...
└── Span: synthesize
```
Each span carries: input, output, model, token_usage, duration_ms, user_id.

## Component Relationships

### graphgen pipeline stages
```
POST /documents (upload)
  → DocumentService
      → ParserRegistry.get_parser(ext)
          → ConcreteParser.parse() → ParsedDocument
      → HeadingAwareChunker.chunk() → list[Chunk]
      → EntityExtractor (GLiNER hints + LLMGraphTransformer)
      → EntityResolver (string + semantic merge)
      → CommunityDetector (plugin: topics.py, optional)
      → NodePruner
      → Neo4jUploader (schema-driven Cypher batch)
      → PgvectorStore (CHUNK embeddings)
      → DOCUMENT.status = 'complete'
```

### graphrag agent loop
```
POST /v1/chat/stream
  → AgentWorkflow.run_stream()
      → decompose_query (LLM)  ──→ SSE: reasoning
      → loop (max 6 iterations):
          → select_tool (LLM)  ──→ SSE: tool_call
          → tool.run()         ──→ SSE: tool_result
          → reflect (LLM)      ──→ SSE: reasoning
          → sufficiency_check
      → synthesize (LLM)       ──→ SSE: token (streamed)
      → finalize citations      ──→ SSE: citation × N
                                ──→ SSE: done (AgentResponse)
```

## Monorepo Task Graph (Turborepo)
```
types#build  ←── web#dev
                 web#build
                 web#typecheck
                 web#lint
```
Python services are orchestrated via Docker Compose (not Turborepo); Turborepo manages the TS workspace only. `pnpm dev` = compose up infra + turbo dev for TS + uvicorn --reload for Python (via compose dev overlay).
