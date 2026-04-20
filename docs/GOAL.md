# GraphKnows → Template Monorepo: Architecture Planning Request

## Context & Codebase

You are planning a full refactor of the `graphknows` repository. Read
#file:services/graphgen #file:services/graphrag #file:frontend
#file:docker-compose.yaml #file:pyproject.toml before planning anything.

Current state:
- `services/graphgen/` — Python ETL pipeline: LangChain, GLiNER, NetworkX,
  Leidenalg, SentenceTransformers, FalkorDB + Postgres upload
- `services/graphrag/` — Python agentic retrieval: LlamaIndex, FastAPI,
  Langfuse tracing, Pydantic
- `frontend/` — React + Vite + Tailwind + react-force-graph-2d, FastAPI BFF
  on port 8001
- Infrastructure: FalkorDB (in-memory graph), Neo4j (graphdb) Postgres/pgvector (vectors), Docker Compose

## Goal

Transform this into a reusable open-source monorepo template that any company
can clone, configure with their own documents and LLM API keys, and have a
production-quality internal knowledge-graph RAG system running within 30
minutes. It must be genuinely modular: adding new node types, edge types, or
pipeline stages should require touching only isolated, well-documented
extension points — not rewriting core logic.

---

## Constraints & Non-Negotiables

1. Python backend stays Python. Keep FastAPI for both `graphgen` and
   `graphrag`. Migrate dependency management to `uv` workspaces if not
   already done.
2. Frontend stack: Next.js 15 (App Router) + TypeScript + Tailwind CSS +
   shadcn/ui + Kibo UI component registry. Replace the current Vite/React
   setup entirely.
3. Monorepo tooling: Turborepo at the root. `pnpm` workspaces for the
   TypeScript packages.
4. Graph schema simplification (critical): Strip the current temporal
   hierarchy (DAY → SEGMENT → EPISODE → CHUNK). The new canonical schema
   is strictly:

     (DOCUMENT)-[:CONTAINS]->(CHUNK)-[:MENTIONS]->(ENTITY)
     (ENTITY)<-[:RELATED_TO]->(ENTITY)

   Plan for easy extension: new node labels and relationship types should
   be addable via a `schema.py` config file without touching core pipeline
   code.
5. Docker best practices apply to all Dockerfiles: multi-stage builds,
   non-root users, pinned base image versions, HEALTHCHECK, .dockerignore.

---

## Phase 1 — Monorepo Restructure

Plan the exact new directory tree following this structure:

    graphknows/
    ├── apps/
    │   └── web/                  # Next.js 15 frontend + Kibo UI
    ├── services/
    │   ├── graphgen/             # Refactored ETL service
    │   └── graphrag/             # Refactored agentic RAG service
    ├── packages/
    │   ├── ui/                   # Shared shadcn/ui + Kibo UI components
    │   └── types/                # Auto-generated TypeScript types from OpenAPI specs
    ├── turbo.json
    ├── pnpm-workspace.yaml
    ├── docker-compose.yaml
    └── docker-compose.dev.yaml

For each directory, specify: what moves, what gets deleted, what gets
created new, and why. Pay attention to dependecy injection and clean and easy

---

## Phase 2 — Document Ingestion & Parsing Layer (graphgen refactor)

Design a `DocumentParser` abstraction in
`services/graphgen/src/kg/parser/`. Requirements:

- Input formats to support: PDF, DOCX, PPTX, XLSX, HTML, plain TXT,
  Markdown, and images (via OCR). Each format needs its own parser class
  implementing a common `BaseParser` interface.
- Output contract: every parser must produce a `ParsedDocument` Pydantic
  model containing: `doc_id`, `title`, `source_path`, `file_type`,
  `raw_markdown: str`, `metadata: dict`.
- Chunking strategy: chunk the `raw_markdown` by heading hierarchy
  (H1 → H2 → H3). Each chunk preserves its heading breadcrumb path for
  provenance. Use `markdown-it-py` or equivalent.
- DOCUMENT node: the top-level FalkorDB node. Specify its exact properties:
  `id`, `title`, `source`, `file_type`, `created_at`, `chunk_count`,
  `status` (pending/processing/complete/error).
- Incremental ingestion: the pipeline must support adding new documents
  without rebuilding the full graph. Plan the idempotency mechanism
  (content hash on DOCUMENT node).
- Recommend the best Python libraries for each format parser (e.g.,
  `pymupdf4llm` for PDF→Markdown, `python-docx`, `python-pptx`,
  `unstructured` as fallback).

---

## Phase 3 — Graph Schema & Extension System (graphgen refactor)

Design a `schema.py` configuration file that defines all node labels,
relationship types, and their mandatory/optional properties. This file is
the single source of truth for the graph shape.

Core schema to implement:

    DOCUMENT { id, title, source, file_type, created_at, chunk_count, status }
    CHUNK    { id, doc_id, content, heading_path, position, embedding_id }
    ENTITY   { id, name, label, description, embedding_id, degree_centrality }

    DOCUMENT -[:CONTAINS]-> CHUNK
    CHUNK -[:MENTIONS]-> ENTITY
    ENTITY -[:RELATED_TO { relation_type, confidence }]-> ENTITY

The extension system: plan a `plugins/` directory pattern where a developer
adds a new node type by creating a single Python file that declares node
properties, Cypher CREATE templates, and pipeline hooks. The core pipeline
should discover and register plugins automatically at startup.

---

## Phase 4 — Agentic GraphRAG Workflow (graphrag refactor)

Design a LlamaIndex `AgentWorkflow` (not a simple chain — a proper iterative
agent loop) in `services/graphrag/src/`. The agent must:

1. Decompose the user query into sub-questions.
2. Execute tools iteratively, with each tool call informing the next:
   - `search_chunks(query, top_k)` — hybrid vector + keyword search on CHUNKs
   - `get_entity_neighbors(entity_id, depth)` — Cypher subgraph expansion
   - `get_document_context(doc_id)` — full DOCUMENT metadata + its CHUNKs
   - `search_entities(name)` — fuzzy entity lookup
3. Accumulate a context object across tool calls, deduplicating retrieved
   chunks and entities.
4. Decide to stop when a sufficiency check passes (all sub-questions
   answered, or max iterations reached).
5. Return a structured response:
   `answer: str`, `citations: List[Citation]`,
   `retrieved_subgraph: GraphData`, `reasoning_steps: List[ReasoningStep]`,
   `tool_calls: List[ToolCall]`

Specify the exact `Citation` model: it must link back to `chunk_id`,
`doc_id`, `doc_title`, `heading_path`, `text_excerpt`.

Plan the Langfuse tracing integration: every tool call, reasoning step, and
token count should be a traced span.

---

## Phase 5 — Frontend (Next.js 15 + Kibo UI)

Design the app router structure and component breakdown. Four main views:

### 5a. Document Manager (/documents)
- Kibo UI Dropzone for multi-file upload with progress per file
- Upload triggers graphgen pipeline via BFF API
- Document list table: title, type, live-polled status badge, chunk_count,
  entity_count, actions (reprocess, delete)
- On-click document detail: chunks, extracted entities, ingestion log
  stream via WebSocket

### 5b. Agent Chat (/chat)
- Full-width chat interface using Kibo UI AI chat component
- Each assistant message renders:
  - Answer text with inline citation markers [1], [2]
  - Collapsible Citations panel: cards showing doc_title, heading_path,
    highlighted excerpt
  - Collapsible Reasoning trace: timeline of tool calls with
    inputs/outputs (Kibo UI Timeline component)
  - "Visualize Context" button: opens side panel with retrieved subgraph
- Streaming responses via SSE from graphrag FastAPI endpoint
- Left sidebar: conversation history list

### 5c. Graph Explorer (/graph)
- react-force-graph-2d (keep existing) with right-hand inspector panel
- Filter by node label (DOCUMENT / CHUNK / ENTITY), by document, by
  entity type
- Click node → inspector shows all properties + relationship counts
- "Expand neighbors" button triggers subgraph query and merges result
  into canvas
- Color-code nodes by label using design system token colors

### 5d. Analytics (/analytics)
- KPI cards: total documents, chunks, entities, avg entities per document
- Entity frequency bar chart (top 20 by degree centrality)
- Document status distribution donut chart
- Processing timeline (documents ingested over time)

Plan the BFF API layer: a thin FastAPI app or Next.js API routes that
aggregate graphgen (8020) and graphrag (8010) under a unified /api/v1/
prefix. Specify every endpoint needed to support the four views above.

---

## Phase 6 — Infrastructure & DevX

Plan updates to docker-compose.yaml and docker-compose.dev.yaml:
- Add the Next.js `web` container (port 3000)
- Hot-reload for Python services (--reload) and Next.js in dev compose
- All Dockerfiles: multi-stage builds, non-root users, pinned base image
  tags, HEALTHCHECK instructions
- Environment variable strategy: single root .env with naming convention
  (GRAPHGEN_*, GRAPHRAG_*, WEB_*, shared LLM_* keys)
- One-command dev start: `turbo dev` starts all services in parallel

Specify a `CONTRIBUTING.md` template covering the three main extension
points (adding a new parser, a new node type, a new agent tool) with
interface contract examples. This is critical for the template's usability.

---

## Deliverable Format

Structure your plan as follows for each phase:
1. Phase N — Title: 2-sentence summary of the goal
2. Current state: what exists today and what is wrong with it
3. Target state: what it looks like after the change
4. File-level change list: exact files to create, modify, or delete with
   a one-line reason each
5. Key decisions & rationale: 3-5 architectural choices and why
6. Extension points: where a future developer hooks in to customize

Do NOT write implementation code. Write only the architecture plan, file
structure, interface contracts (function signatures and Pydantic model
definitions are fine), and decision rationale.