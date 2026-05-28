# STM-003: MemoryIngestor — text-first STM evidence writer

## Context
Wiki: /root/graphknows/llm-wiki/architecture/memory-layer-v2.md
Project root: /root/graphknows
Read .hermes/instructions/python-style.instructions.md first.
Read .hermes/instructions/tdd-test-driven-development.instructions.md first.
Model: github-copilot/claude-sonnet-4.6

Depends on: STM-001 (schema/pgvector write methods), STM-002 (parsers)

## Goal
MemoryIngestor: the single entry point for writing evidence (documents, code, files) into STM.
No HTTP fetch. No URL handling. Caller provides text (or ParsedFile). Pure pipeline.

## Create services/graphknows/graphknows/memory/__init__.py
Empty.

## Create services/graphknows/graphknows/memory/ingestor.py

```python
"""MemoryIngestor: write text/file content into short-term memory (Memgraph + pgvector STM)."""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from graphknows.ingestion.parsers.base import ParsedFile, TextSegment
from graphknows.ingestion.parsers.registry import get_parser
from graphknows.retrieval.stores.embedder import embed_one
from graphknows.retrieval.stores.memgraph import MemgraphStore
from graphknows.retrieval.stores.pgvector import PgVectorStore

log = logging.getLogger("graphknows.memory.ingestor")


@dataclass
class IngestResult:
    """Result of a single ingest call."""
    session_id: str
    file_id: str
    title: str
    chunk_count: int
    entity_count: int
    elapsed_s: float
    errors: list[str] = field(default_factory=list)


class MemoryIngestor:
    """Ingest parsed text into STM (Memgraph evidence graph + pgvector stm_chunks).

    Usage:
        ingestor = MemoryIngestor(mg_store, pg_store, extractor)
        result = ingestor.ingest_text("long document...", session_id="s1", title="My Doc")
        result = ingestor.ingest_file("/path/to/report.py", session_id="s1")

    Attributes:
        _mg: MemgraphStore for the session evidence graph.
        _pg: PgVectorStore for stm_chunks.
        _extractor: Entity extractor (InProcessExtractor or compatible).
    """

    def __init__(
        self,
        mg_store: MemgraphStore,
        pg_store: PgVectorStore,
        extractor: Any,  # InProcessExtractor — typed Any to avoid heavy import at module level
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._extractor = extractor

    def ingest_text(
        self,
        text: str,
        session_id: str,
        title: str = "",
        source_hint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest raw text into STM.

        Args:
            text: Raw source text. Parser selected as PlainTextParser.
            session_id: Session namespace.
            title: Optional human title.
            source_hint: Original path/URL for provenance.
            metadata: Extra key/value pairs stored on the FILE node.
        Returns:
            IngestResult with counts and timing.
        """
        parser = get_parser(source_hint or ".txt")
        parsed = parser.parse(text, source_hint=source_hint, title=title)
        return self._ingest_parsed(parsed, session_id, metadata or {})

    def ingest_file(
        self,
        path: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Read a file from disk and ingest into STM.

        Args:
            path: Absolute or relative path to file. Extension determines parser.
            session_id: Session namespace.
            metadata: Extra key/value pairs stored on the FILE node.
        Returns:
            IngestResult with counts and timing.
        """
        import os
        text = open(path, encoding="utf-8", errors="replace").read()
        parser = get_parser(path)
        parsed = parser.parse(text, source_hint=path, title=os.path.basename(path))
        return self._ingest_parsed(parsed, session_id, metadata or {})

    def ingest_parsed(
        self,
        parsed: ParsedFile,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest an already-parsed ParsedFile into STM."""
        return self._ingest_parsed(parsed, session_id, metadata or {})

    def _ingest_parsed(
        self, parsed: ParsedFile, session_id: str, metadata: dict[str, Any]
    ) -> IngestResult:
        t0 = time.monotonic()
        errors: list[str] = []
        entity_count = 0

        # 1. Ensure SESSION + FILE nodes in Memgraph
        try:
            self._mg.write_session_metadata(session_id, {"started_at": t0})
            self._mg.write_file_node(
                session_id=session_id,
                file_id=parsed.source_id,
                title=parsed.title,
                parser_name=parsed.parser_name,
                source_hint=metadata.get("source_hint", ""),
            )
        except Exception as exc:
            errors.append(f"graph_file_node: {exc}")

        # 2. Per-segment: extract entities, embed, write CHUNK + ENTITY nodes
        for seg in parsed.segments:
            chunk_id = hashlib.sha256(
                f"{parsed.source_id}::{seg.char_offset}".encode()
            ).hexdigest()[:16]

            entities: list[dict[str, Any]] = []
            try:
                entities = self._extractor.extract(seg.text)
            except Exception as exc:
                errors.append(f"extract chunk {chunk_id}: {exc}")

            embedding: list[float] = []
            try:
                embedding = embed_one(seg.text)
            except Exception as exc:
                errors.append(f"embed chunk {chunk_id}: {exc}")

            # Write to Memgraph
            try:
                self._mg.write_chunk_node(
                    session_id=session_id,
                    file_id=parsed.source_id,
                    chunk_id=chunk_id,
                    text=seg.text,
                    heading_path=seg.heading_path,
                    metadata=seg.metadata,
                )
                for ent in entities:
                    self._mg.write_entity_node(
                        session_id=session_id,
                        entity_name=ent["name"],
                        entity_type=ent.get("type", "ENTITY"),
                        confidence=ent.get("score", 1.0),
                    )
                    self._mg.write_chunk_entity_edge(
                        chunk_id=chunk_id,
                        entity_name=ent["name"],
                        session_id=session_id,
                    )
                    entity_count += 1
            except Exception as exc:
                errors.append(f"graph_chunk {chunk_id}: {exc}")

            # Write to pgvector STM
            try:
                import asyncio
                asyncio.get_event_loop().run_until_complete(
                    self._pg.write_stm_chunk({
                        "id": chunk_id,
                        "session_id": session_id,
                        "file_id": parsed.source_id,
                        "text": seg.text,
                        "heading_path": seg.heading_path,
                        "embedding": embedding,
                        "metadata": {**seg.metadata, **metadata},
                    })
                )
            except Exception as exc:
                errors.append(f"pg_chunk {chunk_id}: {exc}")

        elapsed = time.monotonic() - t0
        log.info(
            "ingest session=%s file=%s chunks=%d entities=%d elapsed=%.1fs errors=%d",
            session_id, parsed.source_id, len(parsed.segments), entity_count, elapsed, len(errors),
        )
        return IngestResult(
            session_id=session_id,
            file_id=parsed.source_id,
            title=parsed.title,
            chunk_count=len(parsed.segments),
            entity_count=entity_count,
            elapsed_s=elapsed,
            errors=errors,
        )
```

## Note on extractor interface
The extractor (InProcessExtractor from graphknows.graphgen.pipeline.entity_relation.extraction)
must expose `extract(text: str) -> list[dict]` where each dict has at minimum `name: str`.
Read the existing InProcessExtractor to confirm — adapt the call if the actual signature differs.
DO NOT change the InProcessExtractor itself — only adapt the call in _ingest_parsed.

## Write tests/test_memory_ingestor.py

Test cases (all unit — mock all external deps):
1. ingest_text() calls parser.parse() then _ingest_parsed (mock parser)
2. ingest_file() reads file from disk, selects correct parser, calls _ingest_parsed
3. _ingest_parsed: calls extractor.extract() for each segment
4. _ingest_parsed: calls embed_one() for each segment
5. _ingest_parsed: calls mg.write_chunk_node() for each segment
6. _ingest_parsed: calls mg.write_entity_node() for each entity
7. _ingest_parsed: calls pg.write_stm_chunk() for each segment
8. _ingest_parsed: extractor failure → appended to errors, continues processing other chunks
9. IngestResult has correct chunk_count, entity_count, session_id, file_id
10. ingest_text() with empty string → 0 chunks, 0 entities, no errors

Use unittest.mock.MagicMock for mg_store, pg_store, extractor.
Mock embed_one at graphknows.memory.ingestor.embed_one.

## Acceptance criteria
- All 10 tests pass
- ruff clean, mypy clean
- check_imports.py clean
- No HTTP calls, no live DB calls in production code path
