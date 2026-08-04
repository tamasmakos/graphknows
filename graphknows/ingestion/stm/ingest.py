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
import time
from collections import deque
from itertools import zip_longest
from typing import Any

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.parsers.text import (
    ParsedFile,
    PlainTextParser,
    search_surface,
    strip_content_heading,
)
from graphknows.ingestion.stm._entity_writer import graph_entities, is_graph_entity_name
from graphknows.models.ingest import IngestResult

log = logging.getLogger("graphknows.ingestion.stm.ingest")

# Frames matched per chunk by embedding cosine: top-k AND above a similarity floor
# (cosines run low under the remote embedder, ~0.3, so the floor is permissive).
_FRAME_TOP_K = 5
_FRAME_MIN_SIM = 0.20

# Ontology classes tagged onto a chunk as MATCHES_CLASS edges. Deliberately
# narrower than the label-injection top-k: injected labels are cheap guesses the
# extractor can ignore, whereas a persisted tag is signal retrieval will trust.
_ONT_TAG_TOP_K = 3

# Floor for mapping an entity TYPE onto an ontology class by embedding. Measured
# against personal.ttl with bge-small, correct mappings (PERSON->Person 0.66,
# LOCATION->Place 0.54, DATE->TimePoint 0.45) sit above it while wrong ones
# (OCCUPATION->Possession 0.43, ANIMAL->Emotion 0.38) fall below. Leaving an
# entity unmapped is the safe outcome, so the floor errs high.
_ONT_CLASS_MIN_SIM = 0.45

# Relation properties offered to the extractor per chunk, selected by DOMAIN and
# RANGE (see _relation_labels). Not a free parameter: at 5 offered the extractor
# emitted `Gina spouse Jon` and `Gina children Jon` about two friends, because a
# closed vocabulary with no way to decline FORCES a pick; at ~14 the same turns
# produced colleague/owns/follows. GLiNER and GLiREL are both trained at <=25
# labels per instance and our relex model card demos 2-5, so the ceiling is real
# too. 15 sits in the measured band.
_RELATION_LABEL_CAP = 15

# Turns to accumulate before the lexical index is frozen. Document frequency is
# what separates a discriminating altLabel ("dance") from a ubiquitous one
# ("about"), and a handful of turns is not enough to tell them apart; 40 is,
# and waiting longer only delays every chunk before it.
_LEXICAL_WARMUP = 40

# The one class the range gate enforces against (see _range_admits). Asserting a
# non-person-ranged property over a person is the whole observed defect; every
# other range mismatch is the ontology being narrower than the data.
_PERSON_CLASS = "person"

# The gate only fires when the object is CONFIDENTLY a person. Censused over
# every entity/relation-endpoint type in conv-30 (20 distinct, namespace
# lg30v2), exactly three map onto Person:
#
#     PERSON        1.000   correct   -> must block
#     noun.person   0.840   correct   -> must block
#     PRODUCT       0.580   WRONG     -> must admit
#
# so the floor goes at the midpoint of that gap. An earlier 0.9 was set from the
# label-hit score alone and sat on the wrong side of it: relation endpoints are
# typed by WordNet supersense when NER does not label them (see _arg_type), a
# supersense never label-matches a class, and `noun.person` therefore scored
# 0.840 and let `Jon owns students` through — the single edge the gate exists
# for. Retune only against a fresh census; the two neighbours are 0.26 apart.
_RANGE_GATE_MIN_CLASS_SIM = 0.7


def _entities_from(entities: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    """Entities lying in the current turn, with spans rebased onto it.

    Spans before ``offset`` belong to the window's earlier turns, which already
    minted them against their own chunks. An entity carrying no offsets at all
    (``start < 0``) is kept: it cannot be attributed to an earlier turn any more
    than to this one, and dropping a real extraction is the worse error.
    """
    kept = []
    for ent in entities:
        start = int(ent.get("start", -1))
        if start < 0:
            kept.append(ent)
        elif start >= offset:
            kept.append({**ent, "start": start - offset, "end": int(ent.get("end", 0)) - offset})
    return kept


def _attributed(
    relations: list[dict[str, Any]],
    text: str,
    spans: list[tuple[int, int, str]],
    speakers: set[str],
) -> list[dict[str, Any]]:
    """Relations whose SUBJECT the asserting turn actually supports.

    Resolving deixis per turn fixes pronouns but not crossed subjects: when two
    people assert parallel facts in one window ("Jon: lost my job as a banker" /
    "Gina: I also lost my job at Door Dash") the extractor links both names to
    both objects. The guard locates the object's turn and keeps only the nearest
    person named in it — presence alone is not enough, precisely because deixis
    resolution writes the addressee's name into the line.
    """
    from graphknows.ingestion.stm._attribution import admits

    kept = [r for r in relations if admits(str(r["head"]), str(r["tail"]), text, spans, speakers)]
    if len(kept) != len(relations):
        log.info(
            "attribution guard dropped %d of %d relations (subject not supported by the "
            "asserting turn)",
            len(relations) - len(kept),
            len(relations),
        )
    return kept


class _STMIngestHandler:
    """Ingest parsed text into the golden GraphStore (state='raw').

    Attributes:
        _store: GraphStore for the single per-namespace database.
        _extractor: Entity extractor exposing extract(text) -> ExtractionResult.
        _embed_fn: Callable that takes text and returns list[float] embedding.
        _rel_extractor: Optional relation extractor (llm_assisted).
        _ontology: Optional OntologyIndex (ctor arg ``ontology_index``); its
            CLASSES are the per-chunk entity label hints, the chunk
            MATCHES_CLASS tags and the entity INSTANCE_OF edges. The bundled
            CCO digest by default.
        _relations: Optional OntologyIndex (ctor arg ``relation_index``); its
            PROPERTIES are the relation vocabulary offered to the extractor and
            the domain/range gate over what comes back. Defaults to the same
            bundled digest (``settings.relation_ontology_source``) but stays a
            separate index, so a deployment can point relations at a curated
            vocabulary without replacing the classes that type entities.
    """

    def __init__(
        self,
        store: Any,
        extractor: Any,
        embed_fn: Any | None = None,
        rel_extractor: Any | None = None,
        enable_frames: bool = False,
        ontology_index: Any | None = None,
        relation_index: Any | None = None,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._rel_extractor = rel_extractor
        self._enable_frames = enable_frames
        self._ontology = ontology_index
        self._relations = relation_index
        # Predicate mapping uses char-n-gram TF-IDF plus a surface-alias table
        # ("is friends with" -> friend_of), which suits multi-word predicates.
        # Class mapping deliberately does NOT: on single-word type labels the
        # n-gram similarity is dominated by shared suffixes and confidently
        # returns nonsense (LOCATION -> Organization, ANIMAL -> Goal), so types
        # are matched against the class DEFINITIONS by embedding instead.
        self._property_mapper: Any = None
        self._uri_by_label: dict[str, str] = {}
        self._type_cache: dict[str, tuple[str, str, float]] = {}
        # Entity type -> RELATION-profile class, memoised the same way and kept
        # separate: the two vocabularies are the same file by default but need
        # not be, and a type resolves against each independently.
        self._profile_class_cache: dict[str, tuple[str, float]] = {}
        # Recent speaker-attributed turns per session, for the extraction window
        # (see ingest_turn). Bounded by ``extract_window``; a session that ends
        # leaves at most that many strings behind. Read here rather than beside
        # the ontology settings below, because the window is independent of
        # whether an ontology is configured.
        from graphknows.settings import GraphKnowsSettings as _Settings

        _cfg = _Settings()
        self._extract_window: int = _cfg.extract_window
        self._window_overlap: int = _cfg.extract_overlap
        # Turn dicts, not strings: the deixis resolver and the attribution guard
        # both need each turn's speaker, and flattening them loses exactly that.
        self._recent_turns: dict[str, deque[dict[str, str]]] = {}
        # Lexical label selection over the SKOS scheme. Replaces embedding the
        # chunk against class definitions, which measured 11% of tags sharing a
        # word with their own chunk. Built lazily: the lexical index needs the
        # corpus's document frequencies, which only exist once turns arrive.
        self._lexical: bool = _cfg.lexical_labels
        self._scheme: Any = None
        self._lex_index: Any = None
        self._corpus_seen: list[str] = []
        self._lex_frozen: bool = False
        # rdfs:domain / rdfs:range per property label (both multi-valued), and
        # each class's parent chain. These are the ingredients of BOTH halves of
        # the ontology's job now: selecting which properties a chunk is offered
        # (_relation_labels) and gating which of the extracted ones get annotated
        # (_domain_admits / _range_admits). All three are already parsed by the
        # loader; the predicate mapper is a pure string matcher and never sees
        # them. They come from the RELATION vocabulary. The parent chain is
        # load-bearing and was inert until CCO: the schema.org digest declared
        # no subClassOf at all, so _class_ancestors degenerated to the class
        # itself and a property declared on a supertype was never evoked by a
        # subtype. CCO ships the hierarchy (Person -> Animal -> ...), so the
        # walk now reaches those properties.
        self._domain_by_property: dict[str, tuple[str, ...]] = {}
        self._range_by_property: dict[str, tuple[str, ...]] = {}
        self._parents_by_class: dict[str, tuple[str, ...]] = {}
        # Cosine floor for the per-chunk label hints; see
        # ``GraphKnowsSettings.ontology_hint_min_sim`` for the measurement.
        self._hint_min_sim: float = 0.0
        if ontology_index is not None:
            from graphknows.settings import GraphKnowsSettings

            settings = GraphKnowsSettings()
            self._hint_min_sim = settings.ontology_hint_min_sim
            self._uri_by_label = {t.label.casefold(): t.uri for t in ontology_index.classes}
            if self._relations is None:
                from graphknows.ontology import load_ontology_index

                self._relations = load_ontology_index(settings.relation_ontology_source)
                if self._relations is None:
                    # A bundled asset that will not load is broken, not absent —
                    # but relations are enrichment, so this degrades to the
                    # entity ontology's properties rather than failing ingest.
                    # Loudly: silence here reads exactly like "no relations in
                    # this corpus", which is the defect this layer keeps having.
                    log.error(
                        "Relation profile %r did not load; falling back to the entity "
                        "ontology's properties, which selects relations badly",
                        settings.relation_ontology_source,
                    )
                    self._relations = ontology_index
        if self._relations is not None:
            from graphknows.ingestion.extraction.relations.filter import OntologyPredicateMapper

            self._property_mapper = OntologyPredicateMapper(self._relations.property_labels)
            self._domain_by_property = {
                t.label.casefold(): t.domain for t in self._relations.properties if t.domain
            }
            self._range_by_property = {
                t.label.casefold(): t.range for t in self._relations.properties if t.range
            }
            self._parents_by_class = {
                t.label.casefold(): t.parents for t in self._relations.classes
            }
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
        infer: bool = True,
    ) -> IngestResult:
        """Ingest raw text into the GraphStore."""
        parsed = PlainTextParser().parse(text, source_hint=source_hint, title=title)
        return await self._ingest_parsed(parsed, session_id, metadata or {}, infer=infer)

    async def ingest_parsed(
        self,
        parsed: ParsedFile,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> IngestResult:
        """Ingest an already-parsed ParsedFile into the GraphStore."""
        return await self._ingest_parsed(parsed, session_id, metadata or {}, infer=infer)

    async def _ingest_parsed(
        self, parsed: ParsedFile, session_id: str, metadata: dict[str, Any], infer: bool = True
    ) -> IngestResult:
        t0 = time.monotonic()

        await self._ensure_session_and_file(session_id, parsed, metadata, t0)

        # Pre-compute cleaned texts so they are identical for both embedding and
        # writing (no double-clean).
        cleaned_texts = [self._clean_text(seg.text) for seg in parsed.segments]

        use_batch, batch_embeddings = self._precompute_embeddings(cleaned_texts)

        # Bounded chunk-level concurrency. Default 1 (sequential): the
        # ENTITY(name_norm) index is UNIQUE, so concurrent MERGEs of a shared
        # entity name across chunks can race on the unique key. Opt into
        # parallelism by setting INGEST_CHUNK_CONCURRENCY>1.
        cap = int(os.environ.get("INGEST_CHUNK_CONCURRENCY", "1"))

        # Per-chunk ontology label hints, reusing the embeddings above (no
        # re-embed). Empty when no ontology is configured, which is the default.
        ont_classes = [self._match_classes(e) for e in batch_embeddings]
        entity_hints = self._label_hints(batch_embeddings)

        # Batch NER for ALL chunks in one GLiNER2 forward pass when supported.
        # ``infer=False`` skips extraction entirely (chunks + embeddings only) —
        # the entity/relation extraction and its per-relation graph writes are the
        # ingest cost that never touches the vector index.
        batch_extractions: list[ExtractionResult] | None = None
        if (
            infer
            and self._extractor is not None
            and hasattr(type(self._extractor), "extract_batch")
            and cleaned_texts
        ):
            # Pass 1 is entities only: the relation vocabulary is derived FROM
            # the types this pass finds (see _relation_labels), so it cannot be
            # known before it runs.
            first_pass: list[ExtractionResult] = await asyncio.to_thread(
                self._extractor.extract_batch, cleaned_texts, entity_hints, None
            )
            batch_extractions = await self._relation_pass(cleaned_texts, entity_hints, first_pass)

        for idx, seg in enumerate(parsed.segments):
            seg.metadata["chunk_index"] = idx

        async def _process_chunk(idx: int) -> int:
            seg = parsed.segments[idx]
            clean = cleaned_texts[idx]
            embedding: list[float] = batch_embeddings[idx] if use_batch else self._embed_fn(clean)
            if not infer:
                extraction = ExtractionResult()
            elif batch_extractions is not None:
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
                ts=str(seg.metadata.get("timestamp") or metadata.get("timestamp") or ""),
                embedding=embedding,
                extraction=extraction,
                temporal=temporal,
                memory_class=seg.metadata.get("memory_class")
                or metadata.get("memory_class")
                or "raw_evidence",
                ontology_classes=ont_classes[idx] if idx < len(ont_classes) else [],
            )

        async def _run_all() -> tuple[list[int], list[str]]:
            """Ingest every chunk, surviving individual failures.

            One bad chunk must not abandon the rest of the file: the message
            path (``STMService.ingest_messages``) already records a per-turn
            error and continues, and both paths return the same
            ``IngestResult.errors`` contract.
            """
            sem = asyncio.Semaphore(cap)

            async def _with_sem(idx: int) -> int:
                async with sem:
                    return await _process_chunk(idx)

            results = await asyncio.gather(
                *[_with_sem(i) for i in range(len(parsed.segments))],
                return_exceptions=True,
            )
            counts: list[int] = []
            failures: list[str] = []
            for idx, res in enumerate(results):
                if isinstance(res, BaseException):
                    log.warning("ingest chunk %d failed: %s", idx, res)
                    failures.append(f"chunk {idx}: {res}")
                else:
                    counts.append(res)
            return counts, failures

        counts, errors = await _run_all()
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
            errors=errors,
        )

    async def _relation_pass(
        self,
        texts: list[str],
        entity_hints: list[list[str]] | None,
        first_pass: list[ExtractionResult],
    ) -> list[ExtractionResult]:
        """Re-extract with the relation labels pass 1's entity types selected.

        The second forward pass is the price of selecting properties from what
        the text actually contains rather than from what it is topically near;
        the FrameNet layer already pays the same price for the same reason. It
        is only paid where it can produce something: a chunk whose types evoke
        no property keeps its pass-1 result and is not re-extracted.
        """
        if self._relations is None or not texts:
            return first_pass
        hints = [self._relation_labels(r.entities) for r in first_pass]
        todo = [i for i, h in enumerate(hints) if h]
        if not todo:
            log.info(
                "relation pass skipped for all %d chunks: no entity types evoked a property",
                len(texts),
            )
            return first_pass
        t0 = time.monotonic()
        redone = await asyncio.to_thread(
            self._extractor.extract_batch,
            [texts[i] for i in todo],
            [entity_hints[i] for i in todo] if entity_hints else None,
            [hints[i] for i in todo],
        )
        log.info(
            "relation pass: %d/%d chunks re-extracted in %.2fs (%.0f ms/chunk)",
            len(todo),
            len(texts),
            time.monotonic() - t0,
            1000 * (time.monotonic() - t0) / len(todo),
        )
        merged = list(first_pass)
        for idx, result in zip(todo, redone, strict=True):
            merged[idx] = result
        return merged

    async def ingest_turn(
        self,
        session_id: str,
        file_id: str,
        seq: int,
        role: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        speaker: str = "",
    ) -> int:
        """Write one conversation turn as a CHUNK(kind='turn'). Returns entity count.

        Phase-1 bridge: gives the messages path the same full extraction (ENTITY
        + MENTIONS + evidenced relations + temporal) the text/chunk path has.
        ``infer=False`` writes the embedded turn without entity/relation extraction.

        ``speaker`` is who actually said it ("Gina"); it falls back to ``role``,
        which is all this had before — so a two-human conversation recorded every
        chunk's speaker as "user".

        Whether a given turn is worth extracting from is the CALLER's decision,
        expressed through ``infer`` — this layer applies no content policy of
        its own.
        """
        metadata = metadata or {}
        clean = self._clean_text(text) or text
        if not clean.strip():
            # Dropping the turn is fine; dropping it silently is not. From outside,
            # "200 turns in, 0 chunks out" reads identically whether this layer made a
            # content decision or the extractor was wedged. Name the turn so the
            # specific gap is findable.
            log.info(
                "ingest_turn: dropping empty turn (session=%s seq=%s role=%s) - "
                "no non-whitespace text after cleaning",
                session_id,
                seq,
                role,
            )
            return 0
        # The retrieval surface carries the attribution the utterance itself
        # omits: "I'm opening a dance studio" is a fact about Gina and names her
        # nowhere, so neither the embedding nor a lexical scan of `clean` can be
        # reached by a question that asks about Gina. The EXTRACTOR still sees
        # `clean` — putting the name in its input is what minted speakers as
        # ENTITY nodes — and CHUNK.text stays verbatim.
        surface = search_surface(clean, speaker or role)
        # One document per TURN for the lexical index's document frequencies.
        # Windows overlap, so counting them instead reports every content word as
        # ubiquitous and the filter empties the whole vocabulary.
        if self._lexical and not self._lex_frozen:
            self._corpus_seen.append(clean)
        embedding = self._embed_fn(surface)
        # Same ontology steering as the chunk path — the turn's own embedding
        # selects the candidate entity labels, so no extra embedding call.
        ent_hints = self._label_hints([embedding])
        entity_labels = ent_hints[0] if ent_hints else None
        # The extraction WINDOW. A turn read alone loses what the turns before it
        # established: "She said yes" asserts something only if the previous turn
        # named her. The window prepends the recent speaker-attributed turns so
        # the extractor resolves across the turn boundary, while ``clean`` — this
        # turn alone — remains what is stored, embedded and retrieved.
        #
        # ``offset`` is where this turn starts inside that text, and it is what
        # keeps provenance honest: only spans at or past it become this chunk's
        # entities. Without that filter every turn would re-mint the previous
        # four turns' entities and hang MENTIONS edges for them on the wrong
        # chunk. Relations are deliberately NOT filtered — a triplet whose head
        # came from an earlier turn is the entire point, and its endpoints merge
        # by name onto the ENTITY nodes those turns already created.
        window_text, offset, spans = self._extraction_window(session_id, clean, speaker or role)
        # Lexical selection over the SKOS scheme, when it is warm. It sees the
        # WINDOW, not the turn: the labels a chunk needs are evoked by the whole
        # passage the extractor reads.
        lex_ents, lex_rels = self._lexical_labels(window_text)
        if lex_ents:
            entity_labels = lex_ents
        extraction = (
            await asyncio.to_thread(
                self._extract,
                window_text,
                entity_labels,
                # Pass 1: entities only. The relation vocabulary is selected from
                # the types this pass finds, so it cannot precede it.
                None,
                # The turn's speaker anchors its first-person clauses. Without
                # it the extractor can only scrape a speaker back out of the
                # text, so a caller that keeps speaker OUT of the content —
                # which is what stops speakers being minted as entities — pays
                # for it in lost relations.
                speaker or role,
            )
            if infer
            else ExtractionResult()
        )
        # Lexically-evoked properties win over the domain/range-only selection:
        # a property whose own name is spoken is the strongest evidence there is.
        if infer and (rel_labels := lex_rels or self._relation_labels(extraction.entities)):
            t_rel = time.monotonic()
            extraction = await asyncio.to_thread(
                self._extract, window_text, entity_labels, rel_labels, speaker or role
            )
            log.debug(
                "relation pass on turn seq=%s: %d properties offered, %.0f ms",
                seq,
                len(rel_labels),
                1000 * (time.monotonic() - t_rel),
            )
        if infer and offset:
            extraction.entities = _entities_from(extraction.entities, offset)
        if infer and spans:
            extraction.relations = _attributed(
                extraction.relations, window_text, spans, {t[2] for t in spans if t[2]}
            )
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
            speaker=speaker or role,
            search_text=surface,
            # The same anchor the relative dates resolve against is what dates
            # the chunk at retrieval. Until now nothing populated CHUNK.ts, so
            # the column was written, indexed and selected as "" everywhere.
            ts=str(anchor or ""),
            embedding=embedding,
            extraction=extraction,
            temporal=temporal,
            memory_class=metadata.get("memory_class") or "raw_evidence",
            ontology_classes=self._match_classes(embedding) if infer else [],
        )

    # ------------------------------------------------------------------
    # Private step methods
    # ------------------------------------------------------------------

    def _ensure_scheme(self) -> bool:
        """Build the SKOS scheme and this corpus's lexical index, once.

        The index needs document frequencies, and those need text, so it cannot
        be built at construction. It is rebuilt while the corpus is still small
        and then frozen — a frequency estimated from 40 turns is already stable
        enough to tell "dance" from "the", which is all it has to do.
        """
        if not self._lexical:
            return False
        # Lexical selection needs a parser for lemmas and POS. A caller that
        # injected a bare extractor (tests, and any embedding-only composition)
        # simply keeps the previous behaviour instead of failing.
        if getattr(self._extractor, "nlp", None) is None:
            self._lexical = False
            log.info("Lexical label selection off: the extractor exposes no spaCy pipeline")
            return False
        if self._scheme is None:
            from graphknows.ontology.scheme import load_scheme
            from graphknows.settings import GraphKnowsSettings

            cfg = GraphKnowsSettings()
            self._scheme = load_scheme(cfg.ontology_source, cfg.overlay_source)
        # Built EAGERLY from whatever text has arrived, then rebuilt once at the
        # warmup mark. Waiting for the full warmup would leave every corpus's
        # first turns with no lexical selection at all, and a short session with
        # none ever — which is how this first ran and produced zero relations.
        # `usable_index` needs the document count to know when a frequency is
        # meaningful, so a small corpus simply filters nothing.
        if self._corpus_seen and (self._lex_index is None or not self._lex_frozen):
            from graphknows.ontology.scheme import build_index

            self._lex_index = build_index(self._scheme, self._corpus_seen, self._extractor.nlp)
            self._lex_frozen = len(self._corpus_seen) >= _LEXICAL_WARMUP
        return self._lex_index is not None

    def _lexical_labels(self, text: str) -> tuple[list[str] | None, dict[str, str] | None]:
        """``(entity labels, relation labels)`` for one window, lexically selected."""
        if not self._ensure_scheme():
            return None, None
        from graphknows.ontology import entity_labels, relation_labels

        doc = self._extractor.nlp(text)
        ents = entity_labels(doc, self._scheme, self._lex_index)
        rels = relation_labels(doc, set(ents), self._scheme, self._lex_index)
        # relex takes {label: definition}; the definition is what it reads.
        spec = {r: (self._scheme[r].definition or r) for r in rels if r in self._scheme}
        return ents, spec or None

    def _extraction_window(
        self, session_id: str, clean: str, speaker: str
    ) -> tuple[str, int, list[tuple[int, int, str]]]:
        """``(text to extract from, where this turn starts, per-turn spans)``.

        The turn is always the tail of the window, so a zero offset means the
        window is off (``extract_window <= 1``) or this is the session's first
        turn — and in both cases the returned text IS ``clean``, byte for byte,
        which makes ``extract_window=1`` an exact replay of the pre-window
        behaviour rather than an approximation of it.

        **Deixis is resolved per turn** before the lines are joined. A window has
        many speakers and one text, so "I" in one line and "I" in the next are
        different people; leaving them unresolved is what put
        ``Jon -[works for]-> Door Dash`` in the graph when GINA lost that job.
        First person becomes the line's own speaker, second person the
        addressee — skipped entirely above two participants, where the addressee
        is a guess and a wrong name is worse than a pronoun.

        The returned SPANS are what lets :meth:`_write_relations` tell which turn
        asserted what; without them the attribution guard cannot locate anything.

        Every turn joins the buffer, including ones the caller declined to
        extract from: a greeting asserts nothing itself but still establishes who
        is being addressed, which is what a later turn's pronoun needs.
        """
        from graphknows.ingestion.stm._attribution import turn_spans
        from graphknows.ingestion.stm._deixis import resolve_line

        window = self._extract_window
        # Overlap is expressed as the buffer's depth: keeping `window - 1` prior
        # turns means consecutive windows share that many turns, so no pair of
        # adjacent turns ever falls across a boundary unseen. Measured at equal
        # coverage: 2 turns of overlap yields +35% distinct edges.
        recent = self._recent_turns.setdefault(session_id, deque(maxlen=max(window - 1, 0)))
        turns = [*recent, {"speaker": speaker, "text": clean}]
        speakers = list(dict.fromkeys(t["speaker"] for t in turns if t["speaker"]))
        for turn in turns:
            others = [s for s in speakers if s != turn["speaker"]]
            turn["resolved"] = resolve_line(
                turn["text"], turn["speaker"], others[0] if len(others) == 1 else ""
            )
        recent.append({"speaker": speaker, "text": clean})

        if window <= 1 or len(turns) == 1:
            return clean, 0, []
        lines = [
            f"{t['speaker']}: {t['resolved']}" if t["speaker"] else t["resolved"] for t in turns
        ]
        text = "\n".join(lines)
        offset = len(text) - len(lines[-1]) + (len(speaker) + 2 if speaker else 0)
        return text, offset, turn_spans(turns, text)

    async def _ensure_session_and_file(
        self,
        session_id: str,
        parsed: ParsedFile,
        metadata: dict[str, Any],
        t0: float,
    ) -> None:
        await self._store.ensure_session(session_id, started_at=str(t0))
        await self._store.ensure_file(file_id=parsed.source_id, session_id=session_id)

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
        ts: str,
        embedding: list[float],
        search_text: str = "",
        extraction: ExtractionResult,
        temporal: list[dict[str, str]],
        memory_class: str,
        ontology_classes: list[tuple[str, float]] | None = None,
    ) -> int:
        """Write one chunk's CHUNK + ENTITY/MENTIONS + RELATION + TEMPORAL. Returns entity count."""
        filtered = graph_entities(extraction.entities)
        await self._store.write_chunk(
            {
                "chunk_id": chunk_id,
                "session_id": session_id,
                "file_id": file_id,
                "text": text,
                "search_text": search_text or text,
                "seq": seq,
                "kind": kind,
                "speaker": speaker,
                "ts": ts,
                "memory_class": memory_class,
                "embedding": embedding,
            }
        )
        if filtered:
            await self._store.write_entities_bulk(session_id, chunk_id, filtered)

        # Ontology layer — tag the chunk with the classes it evokes and link each
        # entity to its class. Both are additive: an entity that maps to nothing
        # keeps its extracted type and is written exactly as before.
        await self._link_ontology(chunk_id, filtered, ontology_classes or [])

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

        await self._write_relations(extraction, chunk_id, text, speaker)

        iso_dates = [t["iso"] for t in temporal if t.get("iso")]
        if iso_dates:
            await self._store.write_temporal(chunk_id, iso_dates)
        return len(filtered)

    async def _write_relations(
        self, extraction: ExtractionResult, chunk_id: str, text: str, speaker: str = ""
    ) -> dict[str, float]:
        """Write the chunk's relation edges — deterministic pass, then DSPy.

        The ontology ANNOTATES rather than filters: a predicate that maps to an
        ontology property records that mapping, and one that maps to nothing is
        still written under its extracted label. A filter here would reject
        every open-vocab label, which is why one was refused before.

        Temporal/numeric ENDPOINTS are a different matter: write_relation MERGEs
        its endpoints as ENTITY nodes with no filtering, which is how dates
        became ENTITY vertices with community_ids and fragmented the topic
        layer. Both loops share the same endpoint gate; the date signal is not
        lost — it lives on the chunk's MENTIONS_DATE/TEMPORAL link.

        Returns the chunk's conformance report (see :meth:`_report`), which is
        also logged — a gate nobody counts is indistinguishable from a gate that
        never fires, which is how this layer sat at zero unnoticed.
        """
        counts = dict.fromkeys(
            ("blocked_domain", "blocked_range", "unsupported", "pruned_endpoint"), 0
        )
        # "Unsupported by the text" per the spec: an endpoint that corresponds to
        # no extracted span. Counted over every extracted relation, including the
        # ones the endpoint gate drops, so the rates describe what the extractor
        # produced rather than what survived. COUNTED, NOT PRUNED — pruning on
        # this would be a retrieval-affecting change needing its own A/B.
        # The resolved SPEAKER counts as a supported endpoint. SVO rewrites a
        # first-person subject to the speaker's name, so "My collection has
        # twenty pieces" spoken by Gina yields head "Gina" — by design a name
        # that appears in no span of that chunk. Judging it against spans alone
        # made the rate read 1.000 on every conversational chunk regardless of
        # quality, i.e. a metric that measures nothing.
        spans = {str(e.get("name", "")).casefold() for e in extraction.entities}
        if speaker.strip():
            spans.add(speaker.strip().casefold())
        for rel in extraction.relations:
            if spans and not (
                str(rel["head"]).casefold() in spans and str(rel["tail"]).casefold() in spans
            ):
                counts["unsupported"] += 1
            if not is_graph_entity_name(rel["head"], rel.get("head_type", "")) or (
                not is_graph_entity_name(rel["tail"], rel.get("tail_type", ""))
            ):
                counts["pruned_endpoint"] += 1
                continue
            ont_property, _ = self._map_predicate(
                rel["relation"], rel.get("tail_type", ""), rel.get("head_type", "")
            )
            # A refusal is a verdict about this edge, not a lack of one, and it
            # has to outlive the chunk that made it: the same edge is written
            # once per chunk asserting it, and those chunks disagree about the
            # endpoints' types. See write_relation's ontology_blocked.
            domain_blocked = self._domain_blocked(rel["relation"], rel.get("head_type", ""))
            range_blocked = self._range_blocked(rel["relation"], rel.get("tail_type", ""))
            counts["blocked_domain"] += domain_blocked
            counts["blocked_range"] += range_blocked
            blocked = domain_blocked or range_blocked
            await self._store.write_relation(
                rel["head"],
                rel["relation"],
                rel["tail"],
                fact=rel.get("evidence", "") or text,
                chunk_id=chunk_id,
                confidence=rel.get("score", 0.5),
                source=rel.get("source", "gliner"),
                src_type=rel.get("head_type", ""),
                dst_type=rel.get("tail_type", ""),
                ontology_property=ont_property,
                verifier_score=rel.get("verifier_score"),
                ontology_blocked=blocked,
                speaker=speaker,
            )

        # DSPy ontology-guided relations (llm_assisted mode only).
        if self._rel_extractor is not None and len(extraction.entities) >= 2:
            for subj, rel_name, obj, props in await self._rel_extractor.extract_relations(
                text=text
            ):
                if not is_graph_entity_name(str(subj)) or not is_graph_entity_name(str(obj)):
                    continue
                await self._store.write_relation(
                    subj,
                    rel_name,
                    obj,
                    fact=props.get("evidence", "") or text,
                    chunk_id=chunk_id,
                    # The DSPy path carries no per-triplet confidence: the LLM's
                    # self-reported score was never copied into props, so this
                    # read was the literal constant 0.5 for the layer's whole
                    # life. Written as a constant now rather than a lookup that
                    # looks like it varies.
                    confidence=0.5,
                    source="dspy",
                )

        return self._report(chunk_id, len(extraction.relations), counts)

    @staticmethod
    def _report(chunk_id: str, total: int, counts: dict[str, int]) -> dict[str, float]:
        """Per-chunk ontology conformance report, logged and returned.

        ``conformance`` is the share of extracted relations no domain/range
        refusal touched; ``hallucination`` the share with an endpoint that
        matches no extracted span. A chunk with no relations violated nothing, so
        it reports 1.0/0.0 rather than a division by zero.

        Logged at INFO only when a rate is imperfect: on a clean corpus this
        would otherwise be one line per chunk saying nothing, and the DEBUG line
        still distinguishes "gate ran, found nothing" from "gate never ran".
        """
        report: dict[str, float] = {"relations": total, **counts}
        report["conformance"] = (
            1.0
            if not total
            else (total - counts["blocked_domain"] - counts["blocked_range"]) / total
        )
        report["hallucination"] = 0.0 if not total else counts["unsupported"] / total
        line = (
            "ontology conformance chunk=%s relations=%d conformance=%.3f hallucination=%.3f "
            "blocked_domain=%d blocked_range=%d unsupported=%d pruned_endpoint=%d"
        )
        args = (
            chunk_id,
            total,
            report["conformance"],
            report["hallucination"],
            counts["blocked_domain"],
            counts["blocked_range"],
            counts["unsupported"],
            counts["pruned_endpoint"],
        )
        if report["conformance"] < 1.0 or report["hallucination"] > 0.0:
            log.info(line, *args)
        else:
            log.debug(line, *args)
        return report

    def _precompute_embeddings(self, texts: list[str]) -> tuple[bool, list[list[float]]]:
        """Embed every chunk up front when possible. Returns ``(precomputed, embeddings)``.

        Two reasons to do this before the per-chunk loop: the default embedder
        batches into one HTTP round-trip, and ontology label hints are derived
        from a chunk's embedding and must exist *before* extraction runs. An
        overridden ``embed_fn`` (tests) still gets per-chunk calls unless an
        ontology forces the precompute, preserving the injection contract.
        """
        if not texts:
            return False, []
        from graphknows.storage.embedder import embed as embed_batch
        from graphknows.storage.embedder import embed_one

        if self._embed_fn is embed_one:
            import numpy as np  # embed_batch returns np.ndarray

            arr: np.ndarray = embed_batch(texts)
            return True, [arr[i].tolist() for i in range(len(texts))]
        if self._ontology is not None:
            return True, [self._embed_fn(t) for t in texts]
        return False, []

    # ------------------------------------------------------------------
    # Ontology injection (mode-independent; inert when no ontology is set)
    # ------------------------------------------------------------------

    def _match_classes(self, embedding: list[float]) -> list[tuple[str, float]]:
        """Ontology classes this chunk evokes, as ``[(uri, sim)]`` for tagging.

        Uses the stricter TAG floor: these become persisted MATCHES_CLASS edges,
        which retrieval trusts, whereas a label hint the extractor finds no span
        for simply goes unused.
        """
        if self._ontology is None or not embedding:
            return []
        from graphknows.ontology.catalog import TAG_MIN_SIM

        matches = self._ontology.match_classes(embedding, top_k=_ONT_TAG_TOP_K, min_sim=TAG_MIN_SIM)
        return [(term.uri, sim) for term, sim in matches]

    def _label_hints(self, embeddings: list[list[float]]) -> list[list[str]] | None:
        """Per-chunk ontology CLASS labels, offered as candidate entity types.

        ``None`` when no ontology is configured, which keeps the extractor call
        byte-identical to the pre-ontology path.

        Gated by top-k AND a cosine floor (``ontology_hint_min_sim``). Top-k
        alone was enough while the bundled ontology had 11 classes; against
        CCO's 1401 it hands twelve arbitrary labels to a chunk that says
        "Thanks, Gina!". A chunk with nothing above the floor gets an EMPTY hint
        list — a distinct, counted outcome, not twelve bad guesses — and the
        extractor falls back to its base inventory for that chunk.

        Property hints used to be selected here the same way, and that is the
        half that did not work: cosine over 945 object properties ranks
        ``liveBlogUpdate`` (0.64) above every property "Still following my
        passion for dance" actually asserts, so the wrong terms OUTSCORE the
        right ones and no floor separates them. Relation properties are now
        selected structurally instead — see :meth:`_relation_labels`.
        """
        if self._ontology is None or not embeddings:
            return None
        floor = self._hint_min_sim
        entity_hints = [
            [t.label for t, _ in self._ontology.match_classes(e, min_sim=floor)] for e in embeddings
        ]
        if no_class := sum(1 for h in entity_hints if not h):
            log.info(
                "Ontology class hints below floor %.2f for %d/%d chunks",
                floor,
                no_class,
                len(entity_hints),
            )
        return entity_hints

    def _relation_labels(self, entities: list[dict[str, Any]]) -> dict[str, str]:
        """Relation properties this chunk's ENTITY TYPES evoke: ``{label: definition}``.

        The ontology analogue of the FrameNet layer: a frame is evoked by a
        trigger in the text and its core elements then define the schema to fill;
        here the extracted entity types evoke the properties whose declared
        domain and range they satisfy. Structural and anchored in the text —
        unlike a whole-chunk cosine, which on dialogue ranks by topic and
        measured 0 relations over 60 real turns.

        Both endpoints must be present. A chunk holding one Person is offered
        ``knows`` and ``likes`` but not ``worksFor``: with no Organization in it,
        an extractor offered ``worksFor`` could only invent the other end.

        Capped at ``_RELATION_LABEL_CAP``, round-robin over the (domain, range)
        pairs rather than in profile order — nine interpersonal Person->Person
        properties sit at the top of the vocabulary and would otherwise take
        every slot the moment a person is mentioned, deleting the event, place
        and employment properties from the offer entirely.

        Offering nothing is a real outcome, and a logged one: an unlogged empty
        vocabulary is indistinguishable from a broken extractor downstream.
        """
        if self._relations is None or not entities:
            return {}
        present: set[str] = set()
        for entity_type in {str(e.get("type", "")) for e in entities}:
            label, _score = self._map_to_profile_class(entity_type)
            if label:
                present |= self._class_ancestors(label)

        offered = [
            p
            for p in self._relations.properties
            if p.domain
            and p.range
            and {d.casefold() for d in p.domain} & present
            and {r.casefold() for r in p.range} & present
        ]
        if not offered:
            log.info(
                "no relation property matched the extracted types %s (mapped to %s)",
                sorted({str(e.get("type", "")) for e in entities}),
                sorted(present) or "nothing in the profile",
            )
            return {}

        groups: dict[tuple[tuple[str, ...], tuple[str, ...]], list[Any]] = {}
        for prop in offered:
            groups.setdefault((prop.domain, prop.range), []).append(prop)
        interleaved = [p for row in zip_longest(*groups.values()) for p in row if p is not None][
            :_RELATION_LABEL_CAP
        ]
        return {p.label: p.definition or p.label for p in interleaved}

    def _map_to_profile_class(self, entity_type: str) -> tuple[str, float]:
        """Map an extracted entity type onto a RELATION-profile class: ``(label, score)``.

        Same two stages as :meth:`_map_to_class` — exact label first ("PERSON" ->
        ``Person``), then the type string embedded against the class definitions
        — but over the RELATION vocabulary's classes, which is where the
        properties' declared domain and range are written.

        ``("", 0.0)`` means the type names nothing the profile can type a
        relation endpoint with, and contributes no property to the offer.
        """
        if self._relations is None or not entity_type:
            return "", 0.0
        if (hit := self._profile_class_cache.get(entity_type)) is not None:
            return hit
        label = next(
            (
                t.label
                for t in self._relations.classes
                if t.label.casefold() == entity_type.casefold()
            ),
            "",
        )
        if label:
            result = (label, 1.0)
        else:
            from graphknows.storage.embedder import embed_one

            matches = self._relations.match_classes(
                embed_one(entity_type), top_k=1, min_sim=_ONT_CLASS_MIN_SIM
            )
            result = (matches[0][0].label, matches[0][1]) if matches else ("", 0.0)
        self._profile_class_cache[entity_type] = result
        return result

    def _map_to_class(self, entity_type: str) -> tuple[str, str, float]:
        """Map a free-text entity type onto an ontology class: ``(uri, label, score)``.

        An exact label hit scores 1.0 — the common case, because the ontology's
        own labels were injected into the extractor's vocabulary. Otherwise the
        type is embedded and matched against the class definitions. Results are
        memoised per type string: types repeat across every chunk, so this costs
        one embedding per DISTINCT type in the corpus, not one per entity.

        ``("", "", 0.0)`` means unmapped, and an unmapped entity is still
        written with its extracted type: this is enrichment, not a filter.
        """
        if self._ontology is None or not entity_type:
            return "", "", 0.0
        if (hit := self._type_cache.get(entity_type)) is not None:
            return hit
        uri = self._uri_by_label.get(entity_type.casefold())
        if uri:
            # Canonical ontology casing, not the extractor's ("PERSON" -> "Person").
            label = next((t.label for t in self._ontology.classes if t.uri == uri), entity_type)
            result = (uri, label, 1.0)
        else:
            from graphknows.storage.embedder import embed_one

            matches = self._ontology.match_classes(
                embed_one(entity_type), top_k=1, min_sim=_ONT_CLASS_MIN_SIM
            )
            result = (
                (matches[0][0].uri, matches[0][0].label, matches[0][1])
                if matches
                else ("", "", 0.0)
            )
        self._type_cache[entity_type] = result
        return result

    async def _link_ontology(
        self,
        chunk_id: str,
        entities: list[dict[str, Any]],
        ontology_classes: list[tuple[str, float]],
    ) -> None:
        """Write the chunk's MATCHES_CLASS tags and its entities' INSTANCE_OF edges."""
        if self._ontology is None:
            return
        if ontology_classes:
            await self._store.write_class_matches(chunk_id, ontology_classes)
        mappings = []
        for ent in entities:
            uri, label, score = self._map_to_class(str(ent.get("type", "")))
            if uri:
                mappings.append({"name": ent["name"], "uri": uri, "label": label, "score": score})
        if mappings:
            await self._store.write_instance_of(mappings)

    def _map_predicate(
        self, predicate: str, object_type: str = "", subject_type: str = ""
    ) -> tuple[str, float]:
        """Map a relation predicate onto an ontology property label.

        Gated by the property's declared ``rdfs:domain`` and ``rdfs:range``: the
        surface-alias table maps the bare predicate "has" onto ``owns``, which is
        right for a possession and wrong for a person, and the mapper is a pure
        string matcher that never sees either endpoint. The ontology already
        carries what settles it, so domain and range decide rather than a special
        case per alias.
        """
        if self._property_mapper is None or not predicate:
            return "", 0.0
        label, score = self._property_mapper.map_predicate(predicate)
        if label and not (
            self._domain_admits(label, subject_type) and self._range_admits(label, object_type)
        ):
            return "", 0.0
        return (label or ""), score

    def _domain_blocked(self, predicate: str, subject_type: str) -> bool:
        """Did the domain gate REFUSE to annotate this predicate for this subject?

        Same contract as :meth:`_range_blocked`: a refusal is a verdict that has
        to survive later writes of the same edge from chunks that typed the
        subject differently, whereas an unmappable predicate is mere silence.
        """
        if self._property_mapper is None or not predicate:
            return False
        label, _score = self._property_mapper.map_predicate(predicate)
        return bool(label) and not self._domain_admits(label, subject_type)

    def _domain_admits(self, property_label: str, subject_type: str) -> bool:
        """Can a subject of ``subject_type`` satisfy the property's ``rdfs:domain``?

        The subject-side mirror of :meth:`_range_admits`, and enrichment in the
        same way: an undeclared domain or an unmappable subject type ADMITS, and
        a refusal drops only the ``ontology_property`` annotation.

        Enforced only when ``Person`` is among the declared domains — the same
        narrowing the range gate carries, for the same reason: every other
        domain miss is the ontology being narrower than the data, and
        personal.ttl declares a Person domain on 11 of its 13 properties.
        "London works_as Analytical Engine" is the defect shape: a property that
        requires a human subject, asserted by a place.

        Domains are multi-valued (schema.org's ``domainIncludes``), so the check
        is ANY-match: ``sponsor`` declares Organization, Person, Event and more,
        and satisfying one of them is conformance.

        Unlike the range gate this needs NO confidence floor. That gate concludes
        "the object IS a person", so a wrong ``-> Person`` mapping (``PRODUCT``
        scores 0.580) would delete a correct edge. This one concludes "the
        subject is NOT a person", which every non-person class supports equally —
        whether ``GPE`` landed on ``Event`` (0.494) or ``Place`` does not change
        the verdict. Censused against personal.ttl with the baked embedder, no
        type denoting a person maps to a non-person class: PERSON 1.000 and
        noun.person 0.840 both reach Person, while LOCATION (Place, 0.653),
        noun.location (Place, 0.843), OCCUPATION (Activity, 0.613) and
        ORGANIZATION (1.000) are all correctly not people. ``_map_to_class``
        already refuses to return a class below ``_ONT_CLASS_MIN_SIM``.
        """
        declared = {
            d.casefold() for d in self._domain_by_property.get(property_label.casefold(), ())
        }
        if _PERSON_CLASS not in declared:
            return True
        subj_label, _score = self._map_to_profile_class(subject_type)
        if not subj_label:
            return True  # nothing to judge the subject by — leave it alone
        return bool(declared & self._class_ancestors(subj_label))

    def _range_blocked(self, predicate: str, object_type: str) -> bool:
        """Did the range gate REFUSE to annotate this predicate for this object?

        Distinct from "there was nothing to map": an unmappable predicate is
        silence, a refused one is a verdict that must survive later writes of the
        same edge from other chunks.
        """
        if self._property_mapper is None or not predicate:
            return False
        label, _score = self._property_mapper.map_predicate(predicate)
        return bool(label) and not self._range_admits(label, object_type)

    def _range_admits(self, property_label: str, object_type: str) -> bool:
        """Can an object of ``object_type`` satisfy the property's ``rdfs:range``?

        Enrichment, never a filter: an undeclared range or an unmappable object
        type ADMITS, and a blocked mapping drops only the ``ontology_property``
        annotation — the relation is still written under its extracted predicate.

        A satisfied range admits. An UNsatisfied one admits too, unless the
        object is a person. Enforcing every declared range was measured on
        conv-30 (369 turns, namespace ``lg30o``) and is far too strict: of 27
        ``owns`` edges, 26 are correct (``collection``, ``store``, ``fashion
        pieces``, ``business``) and carry types — PLACE, PRODUCT, ORGANIZATION,
        OCCUPATION — that the bundled ontology's narrow ``rdfs:range
        :Possession`` does not admit. Full enforcement destroyed all 26 to catch
        the one real error.

        That one error is ``Jon owns students``, and every instance of this
        defect has the same shape: a property whose range excludes people,
        asserted over a person. The surface-alias table maps the bare predicate
        "has" onto ``owns``, so "my sister Maria" and "my mother" become
        ownership claims about humans. Restricting the gate to person-objects
        removes exactly those and keeps the rest.
        """
        declared = {
            r.casefold() for r in self._range_by_property.get(property_label.casefold(), ())
        }
        if not declared:
            return True
        obj_label, score = self._map_to_profile_class(object_type)
        if not obj_label:
            return True
        ancestors = self._class_ancestors(obj_label)
        if declared & ancestors:  # any-match: one satisfied range is conformance
            return True
        if score < _RANGE_GATE_MIN_CLASS_SIM:
            return True  # not confident the object is a person — leave it alone
        return _PERSON_CLASS not in ancestors

    def _class_ancestors(self, label: str) -> set[str]:
        """``label`` plus its transitive rdfs:subClassOf parents, casefolded."""
        seen: set[str] = set()
        frontier = [label.casefold()]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(p.casefold() for p in self._parents_by_class.get(current, ()))
        return seen

    @staticmethod
    def _dedupe_nested(fills: list[str]) -> list[str]:
        """Drop fillers wholly contained in another filler of the same role.

        The extractor emits overlapping spans of one phrase, and each became a
        separate role fill on the same frame: "Goal=shopping experience,
        Goal=special shopping experience, Goal=make a special shopping
        experience" is one goal stated three times. Keeps the SHORTEST of a
        nested set — it is the noun phrase; the longer ones have picked up the
        verb and modifiers around it.
        """
        kept: list[str] = []
        for fill in sorted(fills, key=len):
            low = fill.casefold()
            if not any(k.casefold() in low for k in kept):
                kept.append(fill)
        return kept

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
            text,
            primary,
            speakers,
            gliner,
            lambda t: lu_candidate_frames(t, nlp, embedding=embedding),
        )
        # Role fillers become ENTITY nodes, so they must clear the SAME junk gate
        # every other minting path uses. This was the one path that did not call
        # it: measured on LoCoMo conv-30, it minted 624 ENTITY nodes with no type,
        # no confidence and no MENTIONS edge to any chunk — 30% of the graph —
        # and fed them to the generator's fact-sheet ("Buyer=Hey Gina!",
        # "Speaker=Yeah", "Cognizer=Wow").
        out: list[dict[str, Any]] = []
        for inst in instances:
            roles = {
                role: self._dedupe_nested([f for f in fills if is_graph_entity_name(f)])
                for role, fills in inst.roles.items()
            }
            roles = {r: f for r, f in roles.items() if f}
            # The frame's own gate: an instance under two filled roles is not a
            # fact, it is a frame label with a stray span attached.
            if len(roles) < 2:
                continue
            out.append(
                {
                    "frame": inst.frame,
                    "match_sim": inst.match_sim,
                    "role_confidence": inst.role_confidence,
                    "roles": roles,
                }
            )
        return out

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
        return strip_content_heading(text)

    def _extract(
        self,
        clean_text: str,
        extra_entity_labels: list[str] | None = None,
        extra_relation_labels: dict[str, str] | None = None,
        speaker: str = "",
    ) -> ExtractionResult:
        if self._extractor is not None:
            from typing import cast

            return cast(
                ExtractionResult,
                self._extractor.extract(
                    clean_text, extra_entity_labels, extra_relation_labels, speaker
                ),
            )
        return ExtractionResult()
