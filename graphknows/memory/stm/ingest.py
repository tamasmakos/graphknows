"""_STMIngestHandler: STM ingestion logic (moved from memory/ingestor.py)."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any

from graphknows.ingestion.parsers.base import ParsedFile
from graphknows.ingestion.parsers.registry import get_parser
from graphknows.memory.extractor import ExtractionResult as _ExtractionResult
from graphknows.memory.models.ingest import IngestResult
from graphknows.memory.stm._async_bridge import _run

log = logging.getLogger("graphknows.memory.stm.ingest")


class _STMIngestHandler:
    """Ingest parsed text into STM (Memgraph evidence graph + pgvector stm_chunks).

    # TODO(Wave 3): replace with IGraphDb adapter.

    Attributes:
        _mg: MemgraphStore for session evidence graph writes.
        _pg: PgVectorStore for stm_chunks writes.
        _extractor: Entity extractor exposing extract(text) -> ExtractionResult.
        _embed_fn: Callable that takes text and returns list[float] embedding.
        _rel_extractor: Optional relation extractor.
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
            path: Absolute or relative path to file.
            session_id: Session namespace.
            metadata: Extra key/value pairs stored on the FILE node.

        Returns:
            IngestResult with counts and timing.
        """
        import os

        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
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
            clean_text = re.sub(r"^##\s*Content\s*\n+", "", seg.text, flags=re.IGNORECASE).strip()

            # One GLiNER2 call returns both entities and relations — capture both.
            # Previously only .entities was captured; .relations was silently discarded.
            extraction = (
                self._extractor.extract(clean_text)
                if self._extractor is not None
                else _ExtractionResult()
            )
            entities: list[dict[str, Any]] = extraction.entities

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

            # Write GLiNER2 relations — bi-temporal edges with invalidation.
            # Minimum confidence threshold filters noisy low-score pairs.
            _rel_threshold = 0.5
            for rel in extraction.relations:
                if rel.get("score", 0.0) < _rel_threshold:
                    continue
                try:
                    _run(
                        self._mg.write_relation_edge(
                            session_id=session_id,
                            subject_name=rel["head"],
                            relation_name=rel["relation"],
                            object_name=rel["tail"],
                            confidence=rel.get("score", 0.5),
                        )
                    )
                except Exception as exc:
                    log.warning(
                        "write_relation_edge failed session=%s rel=%s: %s",
                        session_id,
                        rel.get("relation"),
                        exc,
                    )

            # DSPy ontology-guided relation extraction (Pass 3).
            # Runs after GLiNER2 to add richer, ontology-typed edges.
            # Uses domain-routed ontology (graphknows vs Schema.org) and
            # co-occurrence warm-start candidates for maximum recall.
            if self._rel_extractor is not None and len(entities) >= 2:
                try:
                    dspy_rels = self._rel_extractor.extract(
                        text=clean_text,
                        entities=entities,
                        ontology_schema=None,  # auto-routed by doc type
                        ontology_id=f"{session_id}:{chunk_id[:8]}",
                    )
                    for drel in dspy_rels:
                        try:
                            _run(
                                self._mg.write_relation_edge(
                                    session_id=session_id,
                                    subject_name=drel["subject"],
                                    relation_name=drel["predicate"],
                                    object_name=drel["object"],
                                    confidence=drel.get("confidence", 0.5),
                                )
                            )
                        except Exception as exc:
                            log.warning(
                                "DSPy write_relation_edge failed session=%s: %s",
                                session_id,
                                exc,
                            )
                    log.debug(
                        "DSPy extraction session=%s chunk=%s: %d triples",
                        session_id,
                        chunk_id[:8],
                        len(dspy_rels),
                    )
                except Exception as exc:
                    log.warning(
                        "DSPy rel_extractor failed session=%s chunk=%s: %s",
                        session_id,
                        chunk_id[:8],
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
