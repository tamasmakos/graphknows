"""_STMIngestHandler: STM ingestion logic (moved from memory/ingestor.py)."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from typing import Any

from graphknows.ingestion.parsers.base import ParsedFile
from graphknows.ingestion.parsers.registry import get_parser
from graphknows.memory.extractor import ExtractionResult
from graphknows.memory.models.ingest import IngestResult
from graphknows.memory.stm._async_bridge import _run
from graphknows.memory.stm._entity_writer import (
    graph_entities,
    write_entity_nodes_and_edges,
    write_relation_edges,
)

log = logging.getLogger("graphknows.memory.stm.ingest")

# Relation endpoints that are dates/times/bare numbers are noise — and writing a
# relation edge re-creates its endpoints as ENTITY nodes, which would otherwise
# smuggle the numeric/temporal entities (dropped from the graph) back in.
_NOISE_REL_LABELS = frozenset(
    {"TIME", "DATE", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}
)
_BARE_NUM_RE = re.compile(r"^[\W\d]+$")


def _noisy_endpoint(text: str, label: str) -> bool:
    """True if a relation endpoint is a date/time/number rather than a real entity."""
    if (label or "").upper() in _NOISE_REL_LABELS:
        return True
    t = (text or "").strip()
    return not t or bool(_BARE_NUM_RE.match(t))


class _STMIngestHandler:
    """Ingest parsed text into STM (ArcadeDB evidence graph + vector store).

    Attributes:
        _mg: ArcadeDBSTMStore for session evidence graph writes.
        _pg: ArcadeDBVectorStore for stm_chunks writes.
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
        spacy_relations: bool = True,
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._extractor = extractor
        self._rel_extractor = rel_extractor
        self._spacy_relations = spacy_relations
        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            from graphknows.memory.stores.embedder import embed_one

            self._embed_fn = embed_one

        # Ontology curation layer — applied AFTER extraction to drop relations
        # whose label is not part of the configured ontology.
        from graphknows.extraction.ontology.relation_filter import build_relation_filter

        self._relation_filter = build_relation_filter()

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

        self._ensure_session_and_file(session_id, parsed, metadata, t0)
        for idx, seg in enumerate(parsed.segments):
            # Preserve chunk order (was always None) so sequential/temporal
            # reasoning and ordered context assembly have a stable index.
            seg.metadata["chunk_index"] = idx
            chunk_id = self._make_chunk_id(session_id, parsed.source_id, seg)
            clean = self._clean_text(seg.text)
            extraction = self._extract(clean)
            embedding: list[float] = self._embed_fn(clean)
            temporal = self._resolve_temporal(extraction.entities, seg, metadata)
            entity_count += self._write_chunk_to_mg(
                session_id, parsed.source_id, chunk_id, clean, seg, extraction, temporal
            )
            self._write_chunk_to_pg(
                session_id,
                parsed.source_id,
                chunk_id,
                clean,
                seg,
                embedding,
                graph_entities(extraction.entities),
                {**metadata, "temporal": temporal} if temporal else metadata,
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

    # ------------------------------------------------------------------
    # Private step methods
    # ------------------------------------------------------------------

    def _ensure_session_and_file(
        self,
        session_id: str,
        parsed: ParsedFile,
        metadata: dict[str, Any],
        t0: float,
    ) -> None:
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

    def _resolve_temporal(
        self, entities: list[dict[str, Any]], seg: Any, metadata: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Resolve DATE/TIME spans to ISO dates, anchored on any supplied date.

        The anchor for relative expressions ("last Saturday") is taken from the
        segment metadata, then the file metadata; absent an anchor only absolute
        dates resolve.
        """
        from graphknows.memory.temporal import resolve_temporal

        seg_meta = getattr(seg, "metadata", {}) or {}
        anchor = (
            seg_meta.get("timestamp")
            or seg_meta.get("anchor_date")
            or metadata.get("anchor_date")
            or metadata.get("timestamp")
        )
        return resolve_temporal(entities, anchor)

    def _make_chunk_id(self, session_id: str, source_id: str, seg: Any) -> str:
        # Include the chunk text in the key: char_offset alone collides when a
        # section yields several chunks that resolve to the same offset (e.g.
        # sentence-packed dialogue), which would overwrite all but one chunk.
        key = f"{session_id}:{source_id}::{seg.char_offset}::{seg.text}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _clean_text(self, text: str) -> str:
        # Strip the '## Content\n\n' heading artifact inserted by PlainTextParser.
        return re.sub(r"^##\s*Content\s*\n+", "", text, flags=re.IGNORECASE).strip()

    def _extract(self, clean_text: str) -> ExtractionResult:
        # One GLiNER2 call returns both entities and relations — capture both.
        if self._extractor is not None:
            from typing import cast

            return cast(ExtractionResult, self._extractor.extract(clean_text))
        return ExtractionResult()

    def _write_chunk_to_mg(
        self,
        session_id: str,
        source_id: str,
        chunk_id: str,
        clean_text: str,
        seg: Any,
        extraction: ExtractionResult,
        temporal: list[dict[str, str]] | None = None,
    ) -> int:
        """Write CHUNK node, entity nodes/edges, and relation edges to ArcadeDB STM.

        Returns the entity count for this chunk.
        """
        chunk_metadata = {**seg.metadata, "temporal": temporal} if temporal else seg.metadata
        _run(
            self._mg.write_chunk_node(
                session_id=session_id,
                file_id=source_id,
                chunk_id=chunk_id,
                text=clean_text,
                heading_path=seg.heading_path,
                metadata=chunk_metadata,
            )
        )
        entity_count = write_entity_nodes_and_edges(
            mg=self._mg,
            session_id=session_id,
            entities=extraction.entities,
            edge_writer=lambda ent: _run(
                self._mg.write_chunk_entity_edge(
                    chunk_id=chunk_id,
                    entity_name=ent["name"],
                    session_id=session_id,
                )
            ),
            _run=_run,
        )
        # GLiNER2 relations — ontology-filtered, then written.
        gliner_relations = self._relation_filter.filter(extraction.relations, label_key="relation")
        write_relation_edges(self._mg, session_id, gliner_relations, _run)

        # SpaCy dep-pattern relations (intentionality strata G/H/I) — toggleable, soft-fail.
        if self._spacy_relations:
            self._write_spacy_dep_relations(session_id, clean_text, extraction.entities)

        # DSPy ontology-guided relation extraction (llm_assisted mode only).
        # extract_relations returns (triplets, nodes, diagnostics).
        # Each triplet is (subject, relation, object, props_dict).
        if self._rel_extractor is not None and len(extraction.entities) >= 2:
            try:
                raw = _run(self._rel_extractor.extract_relations(text=clean_text))
            except Exception as exc:
                # Never let a relation-extractor failure abort ingestion of the
                # whole document — the chunk is still embedded and graph-written.
                log.warning("DSPy relation extraction skipped for chunk %s: %s", chunk_id[:8], exc)
                raw = ([], [], {})
            dspy_triplets: list[tuple[Any, ...]] = raw[0] if isinstance(raw, tuple) else []
            for triplet in dspy_triplets:
                subj, rel_name, obj = triplet[0], triplet[1], triplet[2]
                props = triplet[3] if len(triplet) > 3 else {}
                confidence = props.get("confidence", 0.5) if isinstance(props, dict) else 0.5
                _run(
                    self._mg.write_relation_edge(
                        session_id=session_id,
                        subject_name=subj,
                        relation_name=rel_name,
                        object_name=obj,
                        confidence=confidence,
                    )
                )
            log.debug(
                "DSPy extraction session=%s chunk=%s: %d triples",
                session_id,
                chunk_id[:8],
                len(dspy_triplets),
            )
        return entity_count

    def _write_spacy_dep_relations(
        self,
        session_id: str,
        text: str,
        gliner_entities: list[dict[str, Any]],
    ) -> None:
        """Extract universal SVO relations and write them to ArcadeDB STM.

        Uses the domain-agnostic SVO extractor (subject–predicate–object over the
        dependency parse, with conversational participant resolution and a compact
        relation taxonomy) rather than the narrow speech-act intentionality
        patterns, which only fire on declarative prose. The extractor handles all
        precision filtering (subject must be a real entity, noun-phrase objects,
        questions/discourse pronouns dropped).

        Soft-fails: missing spaCy model or parse errors are logged at DEBUG and do
        not interrupt ingestion.
        """
        try:
            from graphknows.extraction.relations.svo import extract_svo_relations

            triples = extract_svo_relations(text, graph_entities(gliner_entities))
        except Exception as exc:
            log.debug("SVO relation extraction skipped: %s", exc)
            return

        written = 0
        for triple in triples:
            if triple.get("negated"):
                continue
            if _noisy_endpoint(triple["source"], triple.get("source_label", "")) or _noisy_endpoint(
                triple["target"], triple.get("target_label", "")
            ):
                continue
            try:
                _run(
                    self._mg.write_relation_edge(
                        session_id=session_id,
                        subject_name=triple["source"],
                        relation_name=triple["relation"],
                        object_name=triple["target"],
                        confidence=triple.get("confidence", 0.7),
                    )
                )
                written += 1
            except Exception as exc:
                log.debug("SVO relation write error (non-fatal): %s", exc)
        if written:
            log.debug("SVO relations: %d triples written for session=%s", written, session_id)

    def _write_chunk_to_pg(
        self,
        session_id: str,
        source_id: str,
        chunk_id: str,
        clean_text: str,
        seg: Any,
        embedding: list[float],
        entities: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        _run(
            self._pg.write_stm_chunk(
                {
                    "id": chunk_id,
                    "session_id": session_id,
                    "file_id": source_id,
                    "text": clean_text,
                    "heading_path": seg.heading_path,
                    "embedding": embedding,
                    "metadata": {
                        **seg.metadata,
                        **metadata,
                        "atom_type": seg.metadata.get("atom_type", "text"),
                        "memory_class": seg.metadata.get("memory_class", "raw_evidence"),
                        "entity_names": [e["name"] for e in entities],
                    },
                }
            )
        )
