"""MemoryIngestor: write text/file content into short-term memory (Memgraph + pgvector STM)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from threading import Thread
from typing import Any

from graphknows.ingestion.parsers.base import ParsedFile
from graphknows.ingestion.parsers.registry import get_parser

# A single background thread with its own persistent event loop.
# All async store calls are dispatched into this loop — so asyncpg pools,
# which are bound to the loop they were created on, always see the same loop.
_BG_LOOP: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_BG_THREAD: Thread = Thread(
    target=_BG_LOOP.run_forever, daemon=True, name="graphknows-async-worker"
)
_BG_THREAD.start()


def _run(coro: Any) -> Any:
    """Submit *coro* to the persistent background event loop and block until done.

    Safe to call from any thread (including the main asyncio.run thread) because
    it targets a *different* dedicated loop — no nesting conflict.
    """
    future = asyncio.run_coroutine_threadsafe(coro, _BG_LOOP)
    return future.result()


log = logging.getLogger("graphknows.memory.ingestor")


@dataclass
class IngestResult:
    """Result of a single ingest call.

    Attributes:
        session_id: Session namespace.
        file_id: Deterministic source identifier.
        title: Human-readable document title.
        chunk_count: Number of chunks written.
        entity_count: Number of entities extracted.
        elapsed_s: Wall-clock time for the full operation.
        errors: Non-fatal errors collected during processing.
    """

    session_id: str
    file_id: str
    title: str
    chunk_count: int
    entity_count: int
    elapsed_s: float
    errors: list[str] = field(default_factory=list)


class MemoryIngestor:
    """Ingest parsed text into STM (Memgraph evidence graph + pgvector stm_chunks).

    Accepts raw text, a file path, or a pre-parsed ParsedFile. No HTTP fetch.

    Attributes:
        _mg: MemgraphStore for session evidence graph writes.
        _pg: PgVectorStore for stm_chunks writes.
        _extractor: Entity extractor exposing extract(text) -> list[dict].
        _embed: Callable that takes text and returns list[float] embedding.
    """

    def __init__(
        self,
        mg_store: Any,
        pg_store: Any,
        extractor: Any,
        embed_fn: Any | None = None,
        rel_extractor: Any | None = None,
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._extractor = extractor
        self._rel_extractor = rel_extractor
        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            from graphknows.retrieval.stores.embedder import embed_one

            self._embed_fn = embed_one

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
            text: Raw source text.
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

        text = open(path, encoding="utf-8", errors="replace").read()  # noqa: WPS515
        parser = get_parser(path)
        parsed = parser.parse(text, source_hint=path, title=os.path.basename(path))
        return self._ingest_parsed(parsed, session_id, metadata or {})

    def ingest_parsed(
        self,
        parsed: ParsedFile,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest an already-parsed ParsedFile into STM.

        Args:
            parsed: Pre-parsed file output from any BaseParser.
            session_id: Session namespace.
            metadata: Extra key/value pairs stored on the FILE node.

        Returns:
            IngestResult with counts and timing.
        """
        return self._ingest_parsed(parsed, session_id, metadata or {})

    def _ingest_parsed(
        self, parsed: ParsedFile, session_id: str, metadata: dict[str, Any]
    ) -> IngestResult:
        t0 = time.monotonic()
        entity_count = 0

        # 1. Ensure SESSION + FILE nodes in Memgraph
        _run(self._mg.write_session_metadata(session_id, {"started_at": t0}))
        _run(
            self._mg.write_file_node(
                session_id=session_id,
                file_id=parsed.source_id,
                title=parsed.title,
                parser_name=parsed.parser_name,
                source_hint=metadata.get("source_hint", ""),
            )
        )

        # 2. Per-segment: extract entities, embed, write CHUNK + ENTITY nodes
        for seg in parsed.segments:
            chunk_id = hashlib.sha256(
                f"{parsed.source_id}::{seg.char_offset}".encode()
            ).hexdigest()[:16]

            # Strip the '## Content\n\n' heading artifact inserted by PlainTextParser.
            # The prefix dilutes embedding signal — embed pure content, not the heading.
            import re as _re

            clean_text = _re.sub(r"^##\s*Content\s*\n+", "", seg.text, flags=_re.IGNORECASE).strip()

            entities: list[dict[str, Any]] = (
                self._extractor.extract(clean_text) if self._extractor is not None else []
            )

            embedding: list[float] = self._embed_fn(clean_text)

            # Write to Memgraph
            _run(
                self._mg.write_chunk_node(
                    session_id=session_id,
                    file_id=parsed.source_id,
                    chunk_id=chunk_id,
                    text=clean_text,
                    heading_path=seg.heading_path,
                    metadata=seg.metadata,
                )
            )
            for ent in entities:
                _run(
                    self._mg.write_entity_node(
                        session_id=session_id,
                        entity_name=ent["name"],
                        entity_type=ent.get("type", "ENTITY"),
                        confidence=ent.get("score", 1.0),
                    )
                )
                _run(
                    self._mg.write_chunk_entity_edge(
                        chunk_id=chunk_id,
                        entity_name=ent["name"],
                        session_id=session_id,
                    )
                )
                entity_count += 1

            # Wire relation extractor — write REL edges to Memgraph STM
            if self._rel_extractor is not None and len(entities) >= 2:
                try:
                    relations = self._rel_extractor.extract(
                        text=clean_text,
                        entities=entities,
                        ontology_schema="{}",
                    )
                    for rel in relations:
                        _run(
                            self._mg.write_relation_edge(
                                session_id=session_id,
                                subject_name=rel["subject"],
                                relation_name=rel["predicate"],
                                object_name=rel["object"],
                                confidence=rel.get("confidence", 0.5),
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "rel_extractor failed for session=%s: %s",
                        session_id,
                        exc,
                    )

            # Write to pgvector STM
            entity_names_list = [e["name"] for e in entities]
            _run(
                self._pg.write_stm_chunk(
                    {
                        "id": chunk_id,
                        "session_id": session_id,
                        "file_id": parsed.source_id,
                        "text": clean_text,
                        "heading_path": seg.heading_path,
                        "embedding": embedding,
                        "metadata": {
                            **seg.metadata,
                            **metadata,
                            "atom_type": seg.metadata.get("atom_type", "text"),
                            "memory_class": seg.metadata.get("memory_class", "raw_evidence"),
                            "entity_names": entity_names_list,
                        },
                    }
                )
            )

        elapsed = time.monotonic() - t0
        log.info(
            "ingest session=%s file=%s chunks=%d entities=%d elapsed=%.1fs",
            session_id,
            parsed.source_id,
            len(parsed.segments),
            entity_count,
            elapsed,
        )
        return IngestResult(
            session_id=session_id,
            file_id=parsed.source_id,
            title=parsed.title,
            chunk_count=len(parsed.segments),
            entity_count=entity_count,
            elapsed_s=elapsed,
            errors=[],
        )
