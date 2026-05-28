# LTM-9: Document Ingestion Adapter — PDF/URL → Chunk → Entity Dual-Write

## Context

GraphKnows LTM-8 added the schema:
```
(Document)-[HAS_CHUNK]->(Chunk)-[MENTIONS]->(Entity)
```

This task implements the pipeline that populates it. Goal: given a URL or local file path,
produce Document + Chunk nodes in both Memgraph (ephemeral, session-scoped) and Neo4j (LTM),
with MENTIONS edges to existing or new Entity nodes.

The ingestion must be **domain-agnostic** — entity types come from a passed-in ontology list,
not hardcoded labels. This is the same principle as the existing extraction pipeline.

## Architecture

```
DocumentIngestor
  .ingest(source: str, session_id: str, ontology: list[str]) -> IngestResult
      │
      ├─ fetch(source)           → raw text (URL via httpx, PDF via pymupdf / pdfminer)
      ├─ chunk(text)             → list[Chunk]  (500 tok, 50 tok overlap, tiktoken)
      ├─ extract_entities(chunk) → list[Entity] per chunk (GLiNER2 if available, else spaCy NER)
      ├─ write_memgraph(...)     → ephemeral session graph
      └─ write_neo4j(...)        → LTM persistent graph
```

## File to Create

### `services/graphknows/graphknows/document_ingestor.py`

```python
"""Document ingestion: URL/path → Chunk → Entity dual-write to Memgraph + Neo4j."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_ONTOLOGY = [
    "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT",
    "POLICY", "LAW", "INSTITUTION", "TECHNOLOGY", "PRODUCT",
]

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50


@dataclass
class ChunkData:
    id: str
    text: str
    char_offset: int
    token_count: int
    embedding: list[float] = field(default_factory=list)


@dataclass
class EntityMention:
    name: str
    entity_type: str
    chunk_id: str
    score: float = 1.0


@dataclass
class IngestResult:
    document_id: str
    title: str
    url: str
    chunk_count: int
    entity_count: int
    mention_count: int
    errors: list[str] = field(default_factory=list)


class DocumentIngestor:
    """Ingest documents from URL or local path into Memgraph + Neo4j."""

    def __init__(
        self,
        neo4j_driver: Any,          # sync neo4j.GraphDatabase.driver
        memgraph_driver: Any | None = None,  # sync mgclient or neo4j driver pointed at memgraph
        embedding_model: Any | None = None,  # SentenceTransformer or None
    ) -> None:
        self._neo4j = neo4j_driver
        self._mg = memgraph_driver
        self._embed = embedding_model

    def ingest(
        self,
        source: str,
        session_id: str,
        ontology: list[str] | None = None,
        title: str | None = None,
    ) -> IngestResult:
        ...
```

## Implementation Details

### 1. `_fetch(source: str) -> tuple[str, str]` → `(raw_text, resolved_url)`

- If source starts with `http://` or `https://`: use `httpx.get(source, follow_redirects=True,
  headers={"User-Agent": "Graphknows/1.0 (https://graphknows.io; research@graphknows.io)"},
  timeout=30)`. If content-type is `application/pdf`, write bytes to `/tmp/` and parse as PDF.
  Otherwise decode as UTF-8.
- If source ends with `.pdf` (local path): parse with `pymupdf` (fitz) if available, else
  `pdfminer.six`. Extract text page by page, join with `\n\n`.
- Otherwise: read as plain text file.
- Return `(text, source)` — source is the URL or absolute path.

### 2. `_chunk(text: str) -> list[ChunkData]`

- Use `tiktoken` encoding `cl100k_base` to count tokens.
- Sliding window: 500 tokens per chunk, 50 token overlap.
- Each chunk gets `id = sha256(text[:200])[:16]`.
- `char_offset` is the character index of the chunk start in the original text.
- Do NOT split mid-sentence where possible — after tokenizing, find the nearest sentence
  boundary (`.`, `!`, `?` followed by space or newline) to snap chunk boundaries.
- Minimum chunk size: 50 tokens. Drop smaller trailing chunks.

### 3. `_extract_entities(chunk: ChunkData, ontology: list[str]) -> list[EntityMention]`

Try GLiNER2 first, fall back gracefully:

```python
def _extract_entities(self, chunk: ChunkData, ontology: list[str]) -> list[EntityMention]:
    try:
        from gliner import GLiNER  # type: ignore
        model = self._get_gliner(ontology)
        preds = model.predict_entities(chunk.text, ontology, threshold=0.5)
        return [EntityMention(p["text"], p["label"], chunk.id, p["score"]) for p in preds]
    except ImportError:
        pass
    # spaCy fallback
    try:
        import spacy  # type: ignore
        nlp = self._get_spacy()
        doc = nlp(chunk.text)
        return [
            EntityMention(ent.text, ent.label_, chunk.id, 1.0)
            for ent in doc.ents
            if ent.label_ in ontology or not ontology
        ]
    except ImportError:
        log.warning("No NER backend available (GLiNER2 or spaCy). Returning empty entity list.")
        return []
```

Cache GLiNER model instance on `self._gliner_cache` keyed by `tuple(sorted(ontology))`.
Cache spaCy model on `self._spacy_cache`.

### 4. `_embed_chunks(chunks: list[ChunkData]) -> list[ChunkData]`

If `self._embed` is not None, call `self._embed.encode([c.text for c in chunks])` and attach
to each chunk's `.embedding` field as `list[float]`. Batch size 32. Skip if model is None.

### 5. `_write_neo4j(doc_id, title, url, session_id, chunks, mentions)`

Use sync Neo4j driver:

```cypher
-- Document node
MERGE (d:Document {id: $doc_id})
SET d.title = $title, d.url = $url, d.source_type = $source_type,
    d.ingested_at = datetime(), d.session_id = $session_id

-- Chunk nodes (per chunk)
MERGE (c:Chunk {id: $chunk_id})
SET c.text = $text, c.char_offset = $char_offset,
    c.token_count = $token_count, c.embedding = $embedding
MERGE (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c)

-- Entity nodes (per mention, MERGE on name)
MERGE (e:Entity {name: $name})
ON CREATE SET e.type = $entity_type
MERGE (c:Chunk {id: $chunk_id})-[:MENTIONS]->(e)
```

Batch all chunk + mention writes in a single transaction using `session.execute_write()`.

### 6. `_write_memgraph(doc_id, title, url, session_id, chunks, mentions)`

Same pattern but pointed at Memgraph. If `self._mg` is None, skip silently (Neo4j-only mode).
Memgraph uses the same Cypher dialect — same queries work.

Important: wrap each write in try/except and log failures as non-fatal. Memgraph is ephemeral;
if it's down, Neo4j write must still succeed.

## Document ID Strategy

`doc_id = sha256(url or abs_path).hexdigest()[:16]` — stable across re-ingests.
Re-ingesting the same URL updates properties but reuses all MERGE'd nodes.

## Test to Add

### `test_document_ingestor.py` (NEW file at repo root)

A standalone test — no GLiNER2, no GPU, no network:

```python
"""Test DocumentIngestor with synthetic text — no GLiNER2, no network."""
import os, sys
_repo = os.path.dirname(os.path.abspath(__file__))
for _p in [f"{_repo}/services", f"{_repo}/services/graphknows", f"{_repo}/services/graphknows/graphknows"]:
    if _p not in sys.path: sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv("/root/graphknows/.env")

from neo4j import GraphDatabase
from graphknows.document_ingestor import DocumentIngestor, ChunkData, EntityMention

# 1. Chunking — pure logic, no DB
TEXT = "The European Central Bank sets monetary policy. " * 30
ingestor = DocumentIngestor(neo4j_driver=None)
chunks = ingestor._chunk(TEXT)
assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
assert all(c.token_count <= 550 for c in chunks), "Chunk too large"
print(f"PASS  Chunking: {len(chunks)} chunks")

# 2. Entity extraction — spaCy fallback (GLiNER2 not required)
mentions = ingestor._extract_entities(chunks[0], ["ORG", "GPE", "PERSON"])
print(f"PASS  Entity extraction: {len(mentions)} mentions in first chunk")

# 3. Neo4j write — uses real DB
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ["NEO4J_PASSWORD"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
ingestor2 = DocumentIngestor(neo4j_driver=driver)

result = ingestor2.ingest(
    source="__synthetic__",
    session_id="test-ltm9",
    ontology=["ORG", "GPE", "PERSON"],
    title="Synthetic ECB Policy Test",
)
assert result.chunk_count >= 2, f"Expected >=2 chunks written, got {result.chunk_count}"

with driver.session() as s:
    doc_count = s.run("MATCH (d:Document {id: $did}) RETURN count(d) AS c", did=result.document_id).single()["c"]
    chunk_count = s.run("MATCH (:Document {id: $did})-[:HAS_CHUNK]->(c) RETURN count(c) AS c", did=result.document_id).single()["c"]
    assert doc_count == 1, "Document node missing"
    assert chunk_count == result.chunk_count, f"Chunk count mismatch: {chunk_count} vs {result.chunk_count}"
    # Cleanup
    s.run("MATCH (d:Document {id: $did}) DETACH DELETE d", did=result.document_id)
    s.run("MATCH (c:Chunk) WHERE c.id STARTS WITH 'synth-' DETACH DELETE c")

driver.close()
print(f"PASS  Neo4j write: {result.chunk_count} chunks, {result.mention_count} mentions")
print("PASS  Cleanup complete")
print("All LTM-9 tests passed.")
```

For the synthetic source `"__synthetic__"`, detect it in `_fetch()` and return `(TEXT, "synthetic://test")` where TEXT is the 30-repetition ECB sentence above.

## Acceptance Criteria

1. `services/graphknows/graphknows/document_ingestor.py` exists with `DocumentIngestor`, `ChunkData`, `EntityMention`, `IngestResult`
2. `_chunk()` produces 500-tok chunks with 50-tok overlap, sentence-boundary snapping
3. `_extract_entities()` tries GLiNER2, falls back to spaCy, then empty list — no crash if neither installed
4. `_write_neo4j()` creates Document, Chunk, Entity nodes and HAS_CHUNK + MENTIONS edges
5. `_write_memgraph()` reuses same Cypher, skips if driver is None
6. `test_document_ingestor.py` passes: chunking + entity extraction + Neo4j write + cleanup
7. Do NOT break existing `test_ltm_pipeline.py` (still 5/5 + schema + orphan prune)
8. Commit: `LTM-9: DocumentIngestor — URL/PDF/text chunking, NER, dual-write Neo4j+Memgraph`

## Constraints & Conventions

- Python 3.11, strict type hints, Google docstrings, 100-char lines, no bare except
- All Neo4j queries: `q=query` not `query=query` naming convention
- `httpx` is already installed in the venv (confirmed)
- `tiktoken` may need installing: `pip install tiktoken` if not present
- `pymupdf` (fitz) may need installing: `pip install pymupdf` if not present
- spaCy model: `en_core_web_sm` — install with `python -m spacy download en_core_web_sm` if needed
- GLiNER2 is optional — do NOT crash if missing, just log and skip
- Run tests: `cd /root/graphknows && /root/graphknows/services/agent/.venv/bin/python -B test_document_ingestor.py`
- PYTHONPATH: `/root/graphknows/services:/root/graphknows/services/graphknows:/root/graphknows/services/graphknows/graphknows`
- .env at `/root/graphknows/.env`
- Do NOT commit. Leave staged for review.

## Known Facts

- Neo4j: bolt://172.18.0.1:7687, credentials in .env as NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
- Memgraph: bolt://172.18.0.1:17687, no auth — use `neo4j.GraphDatabase.driver("bolt://172.18.0.1:17687", auth=("", ""))`
- Entity MERGE key: `name` only (not name+type) — this is the established convention
- Schema already applied: Document, Chunk constraints exist (LTM-8)
- Existing venv: `/root/graphknows/services/agent/.venv`
- Project root: `/root/graphknows`, branch: `main`, commit: `ee16b63`
