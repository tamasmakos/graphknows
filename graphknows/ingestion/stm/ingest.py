"""_STMIngestHandler: ingest parsed text / turns into the golden GraphStore.

Writes CHUNK spine + ENTITY/MENTIONS + RELATION(evidence) + TEMPORAL against a
single per-namespace ``GraphStore`` (state defaults ``raw``). The text path
chunks a parsed file; the turn path (Phase-1 bridge) writes one CHUNK per
message with ``kind='turn'`` through the same write helper.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from typing import Any

from graphknows.ingestion.parsers.base import ParsedFile
from graphknows.ingestion.parsers.registry import get_parser
from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.models.ingest import IngestResult
from graphknows.ingestion.stm._entity_writer import graph_entities

log = logging.getLogger("graphknows.ingestion.stm.ingest")

# Frames matched per chunk by embedding cosine: top-k AND above a similarity floor
# (cosines run low under the remote embedder, ~0.3, so the floor is permissive).
_FRAME_TOP_K = 5
_FRAME_MIN_SIM = 0.20


class _STMIngestHandler:
    """Ingest parsed text into the golden GraphStore (state='raw').

    Attributes:
        _store: GraphStore for the single per-namespace database.
        _extractor: Entity extractor exposing extract(text) -> ExtractionResult.
        _embed_fn: Callable that takes text and returns list[float] embedding.
        _rel_extractor: Optional relation extractor (llm_assisted).
    """

    def __init__(
        self,
        store: Any,
        extractor: Any,
        embed_fn: Any | None = None,
        rel_extractor: Any | None = None,
        enable_frames: bool = False,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._rel_extractor = rel_extractor
        self._enable_frames = enable_frames
        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            from graphknows.storage.embedder import embed_one

            self._embed_fn = embed_one

    async def ingest_text(
        self,
        text: str,
        session_id: str,
        title: str = "",
        source_hint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest raw text into the GraphStore."""
        parser = get_parser(source_hint or ".txt")
        parsed = parser.parse(text, source_hint=source_hint, title=title)
        return await self._ingest_parsed(parsed, session_id, metadata or {})

    async def ingest_parsed(
        self,
        parsed: ParsedFile,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest an already-parsed ParsedFile into the GraphStore."""
        return await self._ingest_parsed(parsed, session_id, metadata or {})

    async def _ingest_parsed(
        self, parsed: ParsedFile, session_id: str, metadata: dict[str, Any]
    ) -> IngestResult:
        t0 = time.monotonic()

        await self._ensure_session_and_file(session_id, parsed, metadata, t0)

        # Pre-compute cleaned texts so they are identical for both embedding and
        # writing (no double-clean).
        cleaned_texts = [self._clean_text(seg.text) for seg in parsed.segments]

        # Batch all embeddings in one HTTP round-trip when using the default
        # embed_one function. If embed_fn has been overridden (e.g. in tests),
        # fall back to per-chunk calls to preserve the injection contract.
        use_batch = False
        batch_embeddings: list[list[float]] = []
        if cleaned_texts:
            from graphknows.storage.embedder import embed as embed_batch
            from graphknows.storage.embedder import embed_one

            if self._embed_fn is embed_one:
                use_batch = True
                import numpy as np  # embed_batch returns np.ndarray

                arr: np.ndarray = embed_batch(cleaned_texts)
                batch_embeddings = [arr[i].tolist() for i in range(len(cleaned_texts))]

        # Bounded chunk-level concurrency. Default 1 (sequential): the
        # ENTITY(name_norm) index is UNIQUE, so concurrent MERGEs of a shared
        # entity name across chunks can race on the unique key. Opt into
        # parallelism by setting INGEST_CHUNK_CONCURRENCY>1.
        cap = int(os.environ.get("INGEST_CHUNK_CONCURRENCY", "1"))

        # Batch NER for ALL chunks in one GLiNER2 forward pass when supported.
        batch_extractions: list[ExtractionResult] | None = None
        if (
            self._extractor is not None
            and hasattr(type(self._extractor), "extract_batch")
            and cleaned_texts
        ):
            batch_extractions = await asyncio.to_thread(
                self._extractor.extract_batch, cleaned_texts
            )

        for idx, seg in enumerate(parsed.segments):
            seg.metadata["chunk_index"] = idx

        async def _process_chunk(idx: int) -> int:
            seg = parsed.segments[idx]
            clean = cleaned_texts[idx]
            embedding: list[float] = batch_embeddings[idx] if use_batch else self._embed_fn(clean)
            if batch_extractions is not None:
                extraction = batch_extractions[idx]
            else:
                extraction = await asyncio.to_thread(self._extract, clean)
            temporal = self._resolve_temporal(extraction.entities, seg, metadata)
            chunk_id = self._make_chunk_id(session_id, parsed.source_id, seg)
            return await self._write_extracted(
                session_id=session_id,
                file_id=parsed.source_id,
                chunk_id=chunk_id,
                text=clean,
                seq=idx,
                kind="text",
                speaker="",
                embedding=embedding,
                extraction=extraction,
                temporal=temporal,
                memory_class=seg.metadata.get("memory_class")
                or metadata.get("memory_class")
                or "raw_evidence",
                atom_type=seg.metadata.get("atom_type") or "text",
            )

        async def _run_all() -> list[int]:
            sem = asyncio.Semaphore(cap)

            async def _with_sem(idx: int) -> int:
                async with sem:
                    return await _process_chunk(idx)

            return list(await asyncio.gather(*[_with_sem(i) for i in range(len(parsed.segments))]))

        counts: list[int] = await _run_all()
        entity_count = sum(counts)

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

    async def ingest_turn(
        self,
        session_id: str,
        file_id: str,
        seq: int,
        role: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> int:
        """Write one conversation turn as a CHUNK(kind='turn'). Returns entity count.

        Phase-1 bridge: gives the messages path the same full extraction (ENTITY
        + MENTIONS + evidenced relations + temporal) the text/chunk path has.
        ``infer=False`` writes the embedded turn without entity/relation extraction.
        """
        metadata = metadata or {}
        clean = self._clean_text(text) or text
        if not clean.strip():
            return 0
        embedding = self._embed_fn(clean)
        extraction = await asyncio.to_thread(self._extract, clean) if infer else ExtractionResult()
        anchor = metadata.get("timestamp") or metadata.get("anchor_date")
        temporal = self._resolve_temporal_entities(extraction.entities, anchor)
        chunk_id = hashlib.sha256(f"{session_id}:{file_id}:{seq}:{clean}".encode()).hexdigest()[:16]
        return await self._write_extracted(
            session_id=session_id,
            file_id=file_id,
            chunk_id=chunk_id,
            text=clean,
            seq=seq,
            kind="turn",
            speaker=role,
            embedding=embedding,
            extraction=extraction,
            temporal=temporal,
            memory_class=metadata.get("memory_class") or "raw_evidence",
            atom_type="turn",
        )

    # ------------------------------------------------------------------
    # Private step methods
    # ------------------------------------------------------------------

    async def _ensure_session_and_file(
        self,
        session_id: str,
        parsed: ParsedFile,
        metadata: dict[str, Any],
        t0: float,
    ) -> None:
        await self._store.ensure_session(session_id, started_at=str(t0))
        await self._store.ensure_file(
            file_id=parsed.source_id,
            session_id=session_id,
            name=parsed.title,
            media_type=parsed.parser_name,
        )

    async def _write_extracted(
        self,
        *,
        session_id: str,
        file_id: str,
        chunk_id: str,
        text: str,
        seq: int,
        kind: str,
        speaker: str,
        embedding: list[float],
        extraction: ExtractionResult,
        temporal: list[dict[str, str]],
        memory_class: str,
        atom_type: str,
    ) -> int:
        """Write one chunk's CHUNK + ENTITY/MENTIONS + RELATION + TEMPORAL. Returns entity count."""
        filtered = graph_entities(extraction.entities)
        await self._store.write_chunk(
            {
                "chunk_id": chunk_id,
                "session_id": session_id,
                "file_id": file_id,
                "text": text,
                "seq": seq,
                "kind": kind,
                "speaker": speaker,
                "memory_class": memory_class,
                "atom_type": atom_type,
                "entity_names": [e["name"] for e in filtered],
                "embedding": embedding,
            }
        )
        if filtered:
            await self._store.write_entities_bulk(session_id, chunk_id, filtered)

        # FRAME layer — match the chunk's embedding against FrameNet definitions
        # (top-k + min-sim floor, reusing the embedding already computed above; no
        # re-embed) so abstract/general questions can reach this chunk by the frame
        # it evokes. Each FRAME node carries the frame's summary + definition
        # embedding (copied from the catalog, not recomputed) so the query side can
        # vector-search the frames evoked in this namespace.
        if self._enable_frames:
            from graphknows.channels._frame_index import (
                frame_embedding,
                frame_summary,
                match_frames,
            )

            matches = match_frames(embedding, top_k=_FRAME_TOP_K, min_sim=_FRAME_MIN_SIM)
            if matches:
                await self._store.write_frames(
                    session_id,
                    [
                        {
                            "frame_type": name,
                            "chunk_id": chunk_id,
                            "lu": "",
                            "speaker": speaker or "",
                            "fes": [],
                            "fillers": [],
                            "confidence": sim,
                            "summary": frame_summary(name),
                            "embedding": frame_embedding(name),
                        }
                        for name, sim in matches
                    ],
                )

        # FRAME_INSTANCE layer — frame-guided semantic role labeling. The chunk's
        # top FrameNet frames (main-verb LUs, embedding-ranked) become a GLiNER2
        # core-FE schema; the role-fillability gate keeps 0-1 filled instances per
        # short text. Reuses the extractor's spaCy + GLiNER2 (no second load); a
        # second GLiNER2 schema pass, so only on fact-bearing chunks. Distinct from
        # the FRAME chunk-tag layer above.
        if self._enable_frames:
            instances = await asyncio.to_thread(
                self._frame_instances, text, chunk_id, speaker, embedding
            )
            if instances:
                await self._store.write_frame_instances(session_id, chunk_id, instances)

        # GLiNER2 relations — open-vocabulary labels, already precision-gated in
        # the extractor (dep-path/frame corroboration), evidenced to this chunk.
        # No ontology filter here: it would reject every open-vocab label.
        for rel in extraction.relations:
            await self._store.write_relation(
                rel["head"],
                rel["relation"],
                rel["tail"],
                fact=rel.get("evidence", "") or text,
                chunk_id=chunk_id,
                confidence=rel.get("score", 0.5),
                source=rel.get("source", "gliner"),
            )

        # DSPy ontology-guided relations (llm_assisted mode only).
        if self._rel_extractor is not None and len(extraction.entities) >= 2:
            raw = await self._rel_extractor.extract_relations(text=text)
            dspy_triplets: list[tuple[Any, ...]] = raw[0] if isinstance(raw, tuple) else []
            for triplet in dspy_triplets:
                subj, rel_name, obj = triplet[0], triplet[1], triplet[2]
                props = triplet[3] if len(triplet) > 3 else {}
                confidence = props.get("confidence", 0.5) if isinstance(props, dict) else 0.5
                evidence = props.get("evidence", "") if isinstance(props, dict) else ""
                await self._store.write_relation(
                    subj,
                    rel_name,
                    obj,
                    fact=evidence or text,
                    chunk_id=chunk_id,
                    confidence=confidence,
                    source="dspy",
                )

        iso_dates = [t["iso"] for t in temporal if t.get("iso")]
        if iso_dates:
            await self._store.write_temporal(chunk_id, iso_dates)
        return len(filtered)

    def _frame_instances(
        self, text: str, chunk_id: str, speaker: str, embedding: list[float] | None = None
    ) -> list[dict[str, Any]]:
        """Frame-guided role extraction for one chunk → write dicts (thread-run).

        Reuses the entity extractor's already-loaded spaCy + GLiNER2 (no second
        load) and the chunk's already-computed ``embedding`` for frame ranking (no
        re-embed). Speaker/speakers come from the chunk's turn-lines (same logic the
        SVO pass uses), falling back to the passed ``speaker``. Returns
        ``[{frame, match_sim, role_confidence, roles}]`` (0-1 items per short text,
        per the role-fillability gate; the FRAME_INSTANCE id is content-hashed at
        write time). Empty if the extractor doesn't expose reusable models.
        """
        extractor = self._extractor
        if extractor is None or not hasattr(extractor, "nlp") or not hasattr(extractor, "gliner"):
            return []
        from graphknows.ingestion.extraction.entities.extractor import _line_speakers
        from graphknows.ingestion.extraction.relations.frame_srl import (
            extract_frame_instances,
            lu_candidate_frames,
        )

        line_speakers = _line_speakers(text)
        speakers = sorted({s for _, s in line_speakers if s})
        primary = speaker or (speakers[0] if speakers else "")
        if primary and primary not in speakers:
            speakers = sorted({*speakers, primary})

        nlp = extractor.nlp
        gliner = extractor.gliner
        instances = extract_frame_instances(
            text, primary, speakers, gliner,
            lambda t: lu_candidate_frames(t, nlp, embedding=embedding),
        )
        return [
            {
                "frame": inst.frame,
                "match_sim": inst.match_sim,
                "role_confidence": inst.role_confidence,
                "roles": inst.roles,
            }
            for inst in instances
        ]

    def _resolve_temporal(
        self, entities: list[dict[str, Any]], seg: Any, metadata: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Resolve DATE/TIME spans to ISO dates, anchored on any supplied date."""
        seg_meta = getattr(seg, "metadata", {}) or {}
        anchor = (
            seg_meta.get("timestamp")
            or seg_meta.get("anchor_date")
            or metadata.get("anchor_date")
            or metadata.get("timestamp")
        )
        return self._resolve_temporal_entities(entities, anchor)

    @staticmethod
    def _resolve_temporal_entities(
        entities: list[dict[str, Any]], anchor: Any
    ) -> list[dict[str, str]]:
        from graphknows.channels.temporal import resolve_temporal

        return resolve_temporal(entities, anchor)

    def _make_chunk_id(self, session_id: str, source_id: str, seg: Any) -> str:
        # Include the chunk text in the key: char_offset alone collides when a
        # section yields several chunks that resolve to the same offset.
        key = f"{session_id}:{source_id}::{seg.char_offset}::{seg.text}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _clean_text(self, text: str) -> str:
        # Strip the '## Content\n\n' heading artifact inserted by PlainTextParser.
        return re.sub(r"^##\s*Content\s*\n+", "", text, flags=re.IGNORECASE).strip()

    def _extract(self, clean_text: str) -> ExtractionResult:
        if self._extractor is not None:
            from typing import cast

            return cast(ExtractionResult, self._extractor.extract(clean_text))
        return ExtractionResult()
