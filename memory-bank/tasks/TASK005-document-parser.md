# TASK005 — Phase 2: DocumentParser Abstraction (graphgen)

**Status:** Pending  
**Added:** 2026-04-20  
**Updated:** 2026-04-20

## Original Request
Replace the CSV-only `LifeLogParser` with a pluggable `DocumentParser` system supporting 8 input formats, producing a normalized `ParsedDocument` with heading-aware chunking and content-hash idempotency. Add document management REST endpoints.

## Thought Process
The current parser produces `SegmentData` which is deeply life-log specific. The new system must be format-agnostic. The key design choices:
1. `ParsedDocument` is the normalized output of ANY parser — always contains `raw_markdown` and `content_hash`.
2. Chunking is separate from parsing — `HeadingAwareChunker` operates on the `raw_markdown`. This separation means chunking strategy can change without touching parsers.
3. Parser auto-discovery means no central registry to update.
4. `content_hash` drives idempotency — not filename or path.

Library choices:
- PDF: `pymupdf4llm` — best markdown output with heading detection
- DOCX: `python-docx` — direct XML parse; `mammoth` as optional fallback for better HTML
- PPTX: `python-pptx` — extract slide text + notes → markdown
- XLSX: `openpyxl` — convert sheets to markdown tables
- HTML: `trafilatura` (content extraction) + `markdownify` (HTML→MD)
- Markdown: identity pass-through (already markdown)
- TXT: single-section markdown wrap
- Image: `pytesseract` + Pillow → OCR → markdown

## Interface Contracts

```python
class ParsedDocument(BaseModel):
    doc_id: str              # sha256(bytes)[:16]
    title: str
    source_path: str
    file_type: Literal["pdf","docx","pptx","xlsx","html","md","txt","image"]
    raw_markdown: str
    content_hash: str        # full sha256
    metadata: dict[str, Any]

class Chunk(BaseModel):
    chunk_id: str            # f"{doc_id}:{position}"
    doc_id: str
    content: str
    heading_path: list[str]  # ["H1","H2","H3"]
    position: int
    token_count: int

class BaseParser(ABC):
    supported_extensions: ClassVar[tuple[str, ...]]
    @abstractmethod
    def parse(self, source: Path | bytes, *, filename: str) -> ParsedDocument: ...
```

## New REST Endpoints on graphgen
- `POST /documents` — multipart file upload, triggers pipeline async, returns `{doc_id, status}`
- `GET /documents` — list all documents with status
- `GET /documents/{id}` — detail: metadata + chunks + entities
- `POST /documents/{id}/reprocess` — re-ingest with `force=true`
- `DELETE /documents/{id}` — delete doc + its chunks/entities from Neo4j
- `WebSocket /documents/{id}/events` — live ingestion log stream

## Implementation Plan
- [ ] Create `services/graphgen/src/kg/parser/__init__.py`
- [ ] Create `services/graphgen/src/kg/parser/base.py` — BaseParser ABC, ParsedDocument, Chunk models
- [ ] Create `services/graphgen/src/kg/parser/registry.py` — ParserRegistry with auto-discovery
- [ ] Create `services/graphgen/src/kg/parser/chunker.py` — HeadingAwareChunker
- [ ] Create 8 concrete parsers under `parser/parsers/`
- [ ] Create `services/graphgen/src/kg/ingestion/document_service.py` — orchestration
- [ ] Add REST endpoints to `services/graphgen/src/main.py`
- [ ] Update `services/graphgen/pyproject.toml` with new deps
- [ ] Delete old parser files (life.py, old base.py, parsing.py)

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 5.1 | BaseParser + ParsedDocument + Chunk models | Not Started | 2026-04-20 | |
| 5.2 | ParserRegistry auto-discovery | Not Started | 2026-04-20 | |
| 5.3 | HeadingAwareChunker | Not Started | 2026-04-20 | markdown-it-py |
| 5.4 | PDF parser (pymupdf4llm) | Not Started | 2026-04-20 | |
| 5.5 | DOCX parser (python-docx) | Not Started | 2026-04-20 | |
| 5.6 | PPTX parser (python-pptx) | Not Started | 2026-04-20 | |
| 5.7 | XLSX parser (openpyxl) | Not Started | 2026-04-20 | |
| 5.8 | HTML parser (trafilatura + markdownify) | Not Started | 2026-04-20 | |
| 5.9 | Markdown + TXT parsers | Not Started | 2026-04-20 | trivial |
| 5.10 | Image/OCR parser (pytesseract) | Not Started | 2026-04-20 | |
| 5.11 | DocumentService (orchestration) | Not Started | 2026-04-20 | |
| 5.12 | New REST endpoints + WebSocket | Not Started | 2026-04-20 | |
| 5.13 | Update pyproject.toml deps | Not Started | 2026-04-20 | |
| 5.14 | Delete old parser code | Not Started | 2026-04-20 | |

## Progress Log
### 2026-04-20
- Task created during planning session.
