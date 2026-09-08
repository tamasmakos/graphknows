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
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.extraction.entities.hygiene import (
    graph_entities,
    is_graph_entity_name,
)
from graphknows.ingestion.parsers.text import (
    ParsedFile,
    PlainTextParser,
    search_surface,
    strip_content_heading,
)
from graphknows.models.ingest import IngestResult
from graphknows.storage.arcadedb.graph_store import RelationWrite

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings

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
# 25 = the bundled personal profile's full size, so a vocabulary designed to be
# offered WHOLE is never truncated, and the documented <=25 training budget of
# GLiNER/GLiREL. Raised from 15 alongside the proportional allocation below: the
# cap alone was not the defect, the flat round-robin was.
_RELATION_LABEL_CAP = 25

# Turns to accumulate before the lexical index is frozen. Document frequency is
# what separates a discriminating altLabel ("dance") from a ubiquitous one
# ("about"), and a handful of turns is not enough to tell them apart; 40 is,
# and waiting longer only delays every chunk before it.
_LEXICAL_WARMUP = 40

# Entity types that carry no discriminating meaning: the extractor's fallback
# when a minted ENTITY has no nominal referent for _map_to_profile_class to
# resolve (see #160). Measured on conv-30: of 334 entities, 97 were typed
# literally ENTITY and 99 EVENT — 59% with no type a relation's domain/range
# gate can match against.
_UNTYPED_ENTITY_TYPES = frozenset({"", "ENTITY", "EVENT"})

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


# The class set offered to the definition-described typing pass. A fixed
# mid-level list, not a computed one: ranking CCO by descendant count — the
# obvious derivation — optimises for generality and returns `entity` /
# `continuant` / `generically dependent continuant`, which type everything and
# discriminate nothing. It also broke the relation gate downstream, because an
# abstract class satisfies nearly every property's domain and range. This is the
# set that measured best (22/26 surfaces typed, against 14/24 for the mapped
# path); the whole schema is one encoder input, so it stays short by design.
_TYPING_CLASSES: tuple[str, ...] = (
    "Person",
    "Organization",
    "Family",
    "Group of Agents",
    "Facility",
    "Geospatial Region",
    "Artifact",
    "Information Content Entity",
    "Act",
    "Act of Communication",
    "Occupation Role",
    "Skill",
    "Plan",
    "Disease",
)


def _allocate_offer(
    groups: dict[tuple[tuple[str, ...], tuple[str, ...]], list[Any]], cap: int
) -> list[Any]:
    """Choose up to *cap* properties across (domain, range) groups.

    Every group keeps at least one slot, then the remainder is shared in
    proportion to how many applicable properties each group actually holds.

    The floor is the original round-robin's intent: nine interpersonal
    Person->Person properties sit at the top of the profile and would otherwise
    take every slot the moment a person is mentioned, deleting the event, place
    and employment properties from the offer.

    Proportionality is what a FLAT round-robin got backwards on a large
    vocabulary. Measured on the bundled CCO digest for a person-centric chunk:
    99 applicable properties across 7 groups, 89 of them Person->Person, and
    zip_longest handed that group one slot per round — so **5 of 87 familial
    predicates were offered** while ten of the fifteen slots went to BFO
    mereology (`has member part`, `continuant part of`, `is successor of`,
    `uses`, `is predecessor of`). A chunk about someone's family was offered
    almost none of the vocabulary for describing families.

    Within-group order is preserved, so an embedding rank applied by
    :meth:`_rank_within_groups` still decides which members of a group are kept.
    """
    if not groups:
        return []
    ordered = sorted(groups.values(), key=len, reverse=True)
    # Below one slot each, the biggest groups are the ones worth keeping.
    if len(ordered) >= cap:
        return [g[0] for g in ordered[:cap]]

    total = sum(len(g) for g in ordered)
    remaining = cap - len(ordered)
    quota = [1] * len(ordered)
    for i, group in enumerate(ordered):
        # -1 because the floor already granted one.
        quota[i] += int(remaining * (len(group) - 1) / max(total - len(ordered), 1))
    # Integer division leaves slots unspent; hand them to the largest groups.
    for i in range(len(ordered)):
        if sum(quota) >= cap:
            break
        quota[i] += 1
    # Emit round-robin so the interleaved ORDER survives — a caller that ranked
    # within groups by embedding fit expects its best match to lead. The quotas
    # decide how many each group contributes; the interleave decides when.
    pools = [list(group[:n]) for group, n in zip(ordered, quota, strict=True)]
    out: list[Any] = []
    while len(out) < cap and any(pools):
        for pool in pools:
            if pool:
                out.append(pool.pop(0))
                if len(out) >= cap:
                    break
    return out


def _group_ranking_enabled() -> bool:
    """Whether a (domain, range) group is ordered by chunk fit (default off).

    Read at call time so the eval A/B can flip it per subprocess, matching
    ``GRAPHKNOWS_FRAME_SRL``. Delete the knob once the arm has decided it.
    """
    from graphknows.settings import get_settings

    return get_settings().relation_group_ranking


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


def _decoded_instances(frames: list[dict[str, Any]]) -> list[Any]:
    """``ExtractionResult.frames`` as frame instances, filled, with no model run.

    The fills arrived with the chunk's entities and relations, in the one call it
    gets. The frame and its trigger offset are the LOCAL candidate's own — the
    decoder was offered them and could not propose others — so nesting still
    computes over trigger offsets exactly as on the parsed path, and every gate
    in :meth:`_STMIngestHandler._frame_instances` applies unchanged.
    """
    from graphknows.ingestion.extraction.relations.frame_srl import FrameInstance

    return [
        FrameInstance(
            frame=str(filled["frame"]),
            roles={str(role): list(fills) for role, fills in (filled.get("roles") or {}).items()},
            match_sim=1.0,
            role_confidence=1.0,
            trigger_offset=int(filled.get("trigger_offset", -1)),
        )
        for filled in frames
    ]


@dataclass(frozen=True)
class ExtractedChunk:
    """One :meth:`_STMIngestHandler._write_extracted` call's fields, bundled.

    A single chunk's CHUNK row plus everything already computed upstream for
    it — entities/relations, temporal facts, ontology classes, frame triggers.
    """

    session_id: str
    file_id: str
    chunk_id: str
    text: str
    seq: int
    kind: str
    speaker: str
    ts: str
    embedding: list[float]
    extraction: ExtractionResult
    temporal: list[dict[str, str]]
    memory_class: str
    search_text: str = ""
    ontology_classes: list[tuple[str, float]] | None = None
    srl_hits: list[Any] | None = None
    decode_state: str = ""


class _STMIngestHandler:
    """Ingest parsed text into the golden GraphStore (state='raw').

    Attributes:
        _store: GraphStore for the single per-namespace database.
        _extractor: Entity extractor exposing extract(text) -> ExtractionResult.
        _embed_fn: Callable that takes text and returns list[float] embedding.
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
        enable_frames: bool = False,
        ontology_index: Any | None = None,
        relation_index: Any | None = None,
        settings: GraphKnowsSettings | None = None,
    ) -> None:
        from graphknows.settings import get_settings

        # The instance the caller was built with, not a fresh read of the
        # environment: an explicit ``Memory(settings=s)`` must win. Only a
        # caller that hands none (a bare STMService in tests) falls back to
        # the process accessor.
        self._settings = settings if settings is not None else get_settings()
        self._store = store
        self._extractor = extractor
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
        # Entity type -> RELATION-profile class, memoised per type string:
        # types repeat across every chunk, so this costs one embedding per
        # DISTINCT ambiguous type in the corpus.
        self._profile_class_cache: dict[str, tuple[str, float]] = {}
        # Built on first use from the loaded ontology, then reused: the schema is
        # the same for every chunk, and rebuilding it per chunk would re-scan the
        # class list 1437 entries deep for no gain.
        self._typing_schema: dict[str, str] | None = None
        # Recent speaker-attributed turns per session, for the extraction window
        # (see ingest_turn). Bounded by ``extract_window``; a session that ends
        # leaves at most that many strings behind. Read here rather than beside
        # the ontology settings below, because the window is independent of
        # whether an ontology is configured.
        _cfg = self._settings
        self._extract_window: int = _cfg.extract_window
        self._window_overlap: int = _cfg.extract_overlap
        self._window_chunks: bool = _cfg.window_chunks
        # Turn dicts, not strings: the deixis resolver and the attribution guard
        # both need each turn's speaker, and flattening them loses exactly that.
        self._recent_turns: dict[str, deque[dict[str, str]]] = {}
        # Turns seen since this session last emitted a chunk. With window
        # chunking a turn does not get its own chunk; it joins the next window,
        # which is emitted every ``_chunk_stride`` turns.
        self._since_emit: dict[str, int] = {}
        # Whether any turn folded into the pending window asked to be extracted.
        # The emitted chunk carries those turns' text, so it must carry their
        # extract/skip decision too -- not just the emitting turn's.
        self._pending_infer: dict[str, bool] = {}
        # Chunks actually written, so the service can report the real count
        # instead of assuming one per message.
        self._chunks_written: int = 0
        # Every gate that can drop something, tallied for the session. These
        # existed as per-chunk log lines only: `_write_relations` built a full
        # report and the caller discarded the return value, so the numbers were
        # computed and thrown away — the same defect as verifier_score. An
        # abstention nobody can count is indistinguishable from an extractor
        # that found nothing.
        self._abstentions: dict[str, int] = defaultdict(int)
        # Run-permanent decoder failures — rejected credentials, a request the
        # provider refuses outright — kept apart from the gates above: no chunk
        # declined to emit, the run could not decode at all, so they belong on
        # `IngestResult.errors` (FR-021). Drained per report, not accumulated.
        self._decode_errors: list[str] = []
        # Decodes given up on, gated and run-permanent alike. Counted rather
        # than derived from the two above because `_decode_errors` de-duplicates
        # its messages: a rejected key that costs ten chunks appends once, so
        # its length cannot tell a re-drive which chunk just failed (FR-024).
        self._decode_failures: int = 0
        # Accepted MATCHES_CLASS sims for this session, so the spread that
        # issue #158 measured (a degenerate ~0.02 band) can be reported rather
        # than only ever inferred from a bug report.
        self._tag_sims: list[float] = []
        # Set once we have logged that class tagging is off for lack of a
        # lexical index -- otherwise every subsequent chunk repeats the same
        # INFO line for the whole ingest run.
        self._logged_unanchored_class_tag: bool = False
        # Lexical label selection over the SKOS scheme. Replaces embedding the
        # chunk against class definitions, which measured 11% of tags sharing a
        # word with their own chunk. Built lazily: the lexical index needs the
        # corpus's document frequencies, which only exist once turns arrive.
        self._lexical: bool = _cfg.lexical_labels
        self._scheme: Any = None
        self._lex_index: Any = None
        # The overlay's curated coarse-type -> CCO-class mapping (issue #154):
        # populated in `_ensure_scheme` alongside `_scheme`.
        self._coarse_aliases: dict[str, str] = {}
        # Overlay-only classes (in `_scheme` but not in `self._ontology.classes`,
        # e.g. "Dance Studio") as their own tiny index, so the anchor law's
        # candidate set and the ranked vocabulary agree — see `_match_classes`.
        # ``None`` until `_ensure_scheme` builds it, and stays ``None`` when
        # every scheme class is already covered by `self._ontology.classes`.
        self._overlay_index: Any = None
        self._corpus_seen: list[str] = []
        self._lex_frozen: bool = False
        # Corpus size the lexical index was last built at. `_ensure_scheme` is
        # called from both `_lexical_labels` and `_match_classes` per chunk;
        # without this guard every one of those calls would re-run
        # `build_index` — a full spaCy re-parse of the corpus — even though
        # the corpus has not grown since the last build this same chunk.
        self._lex_built_at: int = -1
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
        self._broader_by_class: dict[str, tuple[str, ...]] = {}
        # Cosine floor for the per-chunk label hints; see
        # ``GraphKnowsSettings.ontology_hint_min_sim`` for the measurement.
        self._hint_min_sim: float = 0.0
        if ontology_index is not None:
            settings = self._settings
            self._hint_min_sim = settings.ontology_hint_min_sim
            if self._relations is None:
                from graphknows.symbolic.ontology import load_ontology_index

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
            self._broader_by_class = {
                t.label.casefold(): t.parents_nontransitive for t in self._relations.classes
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

        use_batch, batch_embeddings = await asyncio.to_thread(
            self._precompute_embeddings, cleaned_texts
        )

        # Bounded chunk-level concurrency. Default 1 (sequential): the
        # ENTITY(name_norm) index is UNIQUE, so concurrent MERGEs of a shared
        # entity name across chunks can race on the unique key. Opt into
        # parallelism by setting INGEST_CHUNK_CONCURRENCY>1.
        cap = int(os.environ.get("INGEST_CHUNK_CONCURRENCY", "1"))

        self._seed_corpus(cleaned_texts)

        # Per-chunk ontology label hints, reusing the embeddings above (no
        # re-embed). Empty when no ontology is configured, which is the default.
        ont_classes = [
            self._match_classes(e, t) for e, t in zip(batch_embeddings, cleaned_texts, strict=False)
        ]
        entity_hints = self._label_hints(batch_embeddings)

        # Batch NER for ALL chunks in one GLiNER2 forward pass when supported.
        # ``infer=False`` skips extraction entirely (chunks + embeddings only) —
        # the entity/relation extraction and its per-relation graph writes are the
        # ingest cost that never touches the vector index.
        # Frame parsing for ALL chunks in one batched pass, for the same reason
        # the entity pass below batches: the parser is the ingest cost, and
        # per-chunk calls hand it ~14 sentences at a time. Overlapping windows
        # (5 turns, stride 3) also share most of their text, so this collapses
        # the ~1.7x duplicate parses into one. The hits are kept rather than
        # discarded: the per-chunk stage used to look them up again, and a
        # decoder that gets one call per chunk needs them BEFORE it decodes.
        batch_srl_hits = await self._prewarm_frame_srl(cleaned_texts, infer)

        batch_extractions: list[ExtractionResult] | None = None
        if (
            infer
            and self._extractor is not None
            and hasattr(type(self._extractor), "extract_batch")
            and cleaned_texts
        ):
            batch_extractions = await self._batch_extract(
                cleaned_texts,
                entity_hints,
                batch_srl_hits,
                batch_embeddings if use_batch else None,
            )

        for idx, seg in enumerate(parsed.segments):
            seg.metadata["chunk_index"] = idx

        async def _process_chunk(idx: int) -> int:
            seg = parsed.segments[idx]
            clean = cleaned_texts[idx]
            embedding: list[float] = (
                batch_embeddings[idx] if use_batch else await self._aembed(clean)
            )
            if not infer:
                extraction = ExtractionResult()
            elif batch_extractions is not None:
                extraction = batch_extractions[idx]
            else:
                extraction = await asyncio.to_thread(self._extract, clean)
            temporal = self._resolve_temporal(extraction.entities, seg, metadata)
            chunk_id = self._make_chunk_id(session_id, parsed.source_id, seg)
            return await self._write_extracted(
                ExtractedChunk(
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
                    srl_hits=batch_srl_hits[idx],
                )
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
            errors=errors + self._drain_decode_errors(),
            abstentions=self._report_abstentions(),
        )

    async def redecode(self, session_id: str) -> dict[str, Any]:
        """Re-drive this session's decode-failed chunks through the decoder (FR-024).

        The text already in the graph is the input — no source is re-read and no
        chunk is re-cut — and every write goes back through
        :meth:`_write_extracted`, so a re-drive and an ordinary ingest produce
        the same graph. A chunk whose decode answers is rewritten ``'ok'``; one
        that fails again keeps ``'failed'``, which is what makes a second run
        find nothing and report ``attempted: 0``.
        """
        rows = await self._store.failed_decode_chunks(session_id)
        # ``kind`` and ``memory_class`` are outside the re-drive query's
        # projection, and _write_extracted rewrites the whole CHUNK row: read
        # back rather than default, or a re-decoded turn becomes kind='text'
        # and drops out of every turn-scoped read.
        stored = await self._store.get_chunks_by_ids([r["chunk_id"] for r in rows])
        recovered = 0
        for row in rows:
            failed = await self._redecode_chunk(session_id, row, stored.get(row["chunk_id"], {}))
            recovered += 0 if failed else 1
        return {
            "session_id": session_id,
            "attempted": len(rows),
            "recovered": recovered,
            "still_failed": len(rows) - recovered,
            "abstentions": self._report_abstentions(),
            "errors": self._drain_decode_errors(),
        }

    async def _redecode_chunk(
        self, session_id: str, row: dict[str, Any], stored: dict[str, Any]
    ) -> bool:
        """Decode one stored chunk and rewrite it. Returns whether it failed again.

        ``_extract`` contains a decoder failure rather than raising it, so the
        outcome is read off :attr:`_decode_failures` — the one counter every
        given-up decode passes through.
        """
        text = str(row.get("text") or "")
        speaker = str(row.get("speaker") or "")
        ts = str(row.get("ts") or "")
        before = self._decode_failures
        extraction = await asyncio.to_thread(self._extract, text, None, None, speaker)
        failed = self._decode_failures > before
        embedding = await self._aembed(text)
        await self._write_extracted(
            ExtractedChunk(
                session_id=session_id,
                # The CHUNK already carries its PART_OF edge; an empty file_id is
                # what stops this write from re-asserting one it was not told.
                file_id="",
                chunk_id=row["chunk_id"],
                text=text,
                seq=int(row.get("seq") or 0),
                kind=str(stored.get("kind") or "text"),
                speaker=speaker,
                ts=ts,
                embedding=embedding,
                extraction=extraction,
                temporal=self._resolve_temporal_entities(extraction.entities, ts or None),
                memory_class=str(stored.get("memory_class") or "raw_evidence"),
                ontology_classes=self._match_classes(embedding, text),
                decode_state="failed" if failed else "ok",
            )
        )
        return failed

    @property
    def _decodes_in_one_call(self) -> bool:
        """Whether the decoder labels a chunk in a single call.

        Presence of ``gliner`` is the capability signal the frame layer already
        reads (contracts/decoder.md §1): a decoder that carries the local span
        model can be asked twice — entities, then relations selected from what
        the first pass typed — while one that does not must be handed every
        closed vocabulary up front.

        Asked of the CLASS, as the ``extract_batch`` probe beside it is: the
        local extractor's ``gliner`` is a lazy-loading property, and answering
        this question must not be what loads a model.
        """
        return not hasattr(type(self._extractor), "gliner")

    async def _batch_extract(
        self,
        texts: list[str],
        entity_hints: list[list[str]] | None,
        srl_hits: list[list[Any]],
        embeddings: list[list[float]] | None,
    ) -> list[ExtractionResult]:
        """Label every chunk up front: one decode call each, or the two local passes."""
        if self._decodes_in_one_call:
            return await self._decode_batch(texts, entity_hints, srl_hits, embeddings)
        # Pass 1 is entities only: the relation vocabulary is derived FROM
        # the types this pass finds (see _relation_labels), so it cannot be
        # known before it runs.
        first_pass: list[ExtractionResult] = await asyncio.to_thread(
            self._extractor.extract_batch, texts, entity_hints, None
        )
        return await self._relation_pass(texts, entity_hints, first_pass, embeddings)

    async def _decode_batch(
        self,
        texts: list[str],
        entity_hints: list[list[str]] | None,
        srl_hits: list[list[Any]],
        embeddings: list[list[float]] | None,
    ) -> list[ExtractionResult]:
        """Decode every chunk once, against vocabularies built without a first pass.

        :meth:`_relation_pass` selects the property spec FROM pass 1's entity
        types; a single-call decoder cannot wait for that, so the types come from
        spaCy's observation of the chunk instead (research.md R2) and the frame
        candidates from the parse already done in the batch stage (R3). Every
        vocabulary is closed before the call, and no local extraction model runs
        to build one.
        """
        hints = entity_hints or [[] for _ in texts]
        relation_spec = [
            self._decode_relation_labels(text, hints[i], embeddings[i] if embeddings else None)
            for i, text in enumerate(texts)
        ]
        candidates = [
            self._frame_candidates(text, srl_hits[i], embeddings[i] if embeddings else None)
            for i, text in enumerate(texts)
        ]

        def _decode() -> list[ExtractionResult]:
            return self._extractor.extract_batch(
                texts, entity_hints, relation_spec, frame_candidates=candidates
            )

        before = self._decoder_abstentions()
        try:
            results = await asyncio.to_thread(_decode)
        except Exception as exc:
            # Only a run-permanent failure reaches here: the decoder contains and
            # counts the chunk-level ones itself. Every chunk still gets written,
            # undecoded, so an expired key costs a report line and a re-drive
            # rather than a silently short graph (FR-018, FR-021).
            self._note_decode_failure(exc)
            results = [ExtractionResult() for _ in texts]
        self._merge_decoder_abstentions(before)
        return results

    def _decode_relation_labels(
        self, text: str, class_hints: list[str], embedding: list[float] | None
    ) -> dict[str, str]:
        """This chunk's property spec, derived without a first extraction pass.

        spaCy's observed NER labels for the chunk, unioned with its ontology
        class hints, stand in for the entity types the two-pass path takes from
        pass 1. They then go through the unchanged :meth:`_relation_labels`
        machinery, so the offered spec is built by one code path in both modes —
        only the type source differs (research.md R2). That source is coarser
        than relex's, which is a measurable difference the relation-section
        ablation exists to attribute.
        """
        types = {str(ent.label_) for ent in self._extractor.parse(text).ents} | set(class_hints)
        return self._relation_labels([{"type": t} for t in sorted(types)], embedding)

    def _frame_candidates(
        self, text: str, srl_hits: list[Any], embedding: list[float] | None
    ) -> list[Any]:
        """The frames this chunk evokes, with their core FEs, for the decode request.

        Selection stays LOCAL and stays shared with the two-pass path: the parsed
        SRL triggers when the parser ran, the LU triggers otherwise — the same two
        sources :meth:`_frame_instances` picks between, capped the same way. Only
        the role FILLS move to the decoder, so a frame regression is attributable
        to the fill rather than to a changed candidate set (research.md R3).
        """
        if not self._enable_frames:
            return []
        from graphknows.ingestion.extraction.llm.schema import FrameCandidate
        from graphknows.ingestion.extraction.relations.frame_srl import (
            _MAX_CORE_FES,
            MIN_CORE_ROLES,
            frame_ontology,
            lu_triggers,
        )

        if srl_hits:
            evoked = [(h.frame, h.trigger, int(h.offset)) for h in srl_hits]
        else:
            evoked = [
                (frame, trig.trigger, -1)
                for trig in lu_triggers(
                    text, self._extractor.nlp, embedding=embedding, doc=self._extractor.parse(text)
                )
                # Same top_k the local fill attempts per trigger.
                for frame, _sim in trig.candidates[:3]
            ]
        candidates: list[Any] = []
        for frame, trigger, offset in evoked:
            core = tuple(frame_ontology(frame).get("core_fes", ())[:_MAX_CORE_FES])
            # A frame with fewer core FEs than the fillability gate needs cannot
            # produce a kept instance — the local path skips it for the same reason.
            if len(core) >= MIN_CORE_ROLES:
                candidates.append(FrameCandidate(frame, trigger, offset, core))
        return candidates

    async def _relation_pass(
        self,
        texts: list[str],
        entity_hints: list[list[str]] | None,
        first_pass: list[ExtractionResult],
        embeddings: list[list[float]] | None = None,
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
        hints = [
            self._relation_labels(r.entities, embeddings[i] if embeddings else None)
            for i, r in enumerate(first_pass)
        ]
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

    def _window_decision(
        self, session_id: str, infer: bool, last_in_batch: bool
    ) -> tuple[bool, bool]:
        """Whether this turn emits the pending window, and under which flag.

        With window chunking a turn does not get its own chunk: it is folded
        into the window a later turn emits. The emitted chunk carries the folded
        turns' text, so it carries their extract/skip decision too -- extract
        when ANY of them asked for it. Taking only the emitting turn's flag is
        how one trailing ``infer=False`` turn (tool output, a greeting) silenced
        extraction for everything said before it.

        Returns ``(emit_now, infer_for_the_window)``.
        """
        pending = self._since_emit.get(session_id, 0) + 1
        if pending < self._chunk_stride and not last_in_batch:
            self._since_emit[session_id] = pending
            self._pending_infer[session_id] = self._pending_infer.get(session_id, False) or infer
            return False, infer
        self._since_emit[session_id] = 0
        return True, infer or self._pending_infer.pop(session_id, False)

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
        last_in_batch: bool = True,
    ) -> int:
        """Ingest one conversation turn. Returns entity count.

        With ``window_chunks`` on (the default) a turn does not become its own
        CHUNK: it joins a 5-turn sliding window emitted every
        :attr:`_chunk_stride` turns, and this returns 0 for the turns in
        between. ``last_in_batch`` forces the pending window out so the tail of
        a conversation is never dropped — it defaults True so a caller ingesting
        one turn at a time still gets that turn stored.

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
        window_text, offset, spans, verbatim = self._extraction_window(
            session_id, clean, speaker or role
        )

        # With window chunking, a turn does not get its own chunk — it joins the
        # window that will be emitted `_chunk_stride` turns from the last one.
        # Returning here skips this turn's embedding AND its extraction pass,
        # which is why windowing makes ingest cheaper rather than dearer: the
        # extractor runs once per window instead of once per turn over
        # substantially the same text.
        if self._window_chunks:
            emit, infer = self._window_decision(session_id, infer, last_in_batch)
            if not emit:
                return 0

        # What is stored, embedded and retrieved. A multi-turn window carries
        # the turns around this one, so a question about a fact stated two turns
        # ago can reach the chunk that also holds the answer; its lines are
        # already "Speaker: said", so it is its own retrieval surface. A window
        # of one is the turn byte for byte, and still needs `surface` — the
        # attribution the utterance itself omits is the whole reason that exists.
        windowed = self._window_chunks and verbatim != clean
        chunk_text = verbatim if windowed else clean
        chunk_surface = verbatim if windowed else surface
        embedding = await self._aembed(chunk_surface)
        # Same ontology steering as the chunk path — the chunk's own embedding
        # selects the candidate entity labels, so no extra embedding call.
        ent_hints = self._label_hints([embedding])
        entity_labels = ent_hints[0] if ent_hints else None
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
        if infer and (
            rel_labels := lex_rels or self._relation_labels(extraction.entities, embedding)
        ):
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
        # The provenance filter keeps a turn-chunk from re-minting the previous
        # four turns' entities and hanging their MENTIONS on the wrong chunk.
        # When the chunk IS the window that reasoning inverts: every entity in
        # the window really is mentioned by this chunk, and filtering would drop
        # all but the last turn's.
        if infer and offset and not self._window_chunks:
            extraction.entities = _entities_from(extraction.entities, offset)
        if infer and spans:
            extraction.relations = _attributed(
                extraction.relations, window_text, spans, {t[2] for t in spans if t[2]}
            )
        anchor = metadata.get("timestamp") or metadata.get("anchor_date")
        temporal = self._resolve_temporal_entities(extraction.entities, anchor)
        chunk_id = hashlib.sha256(
            f"{session_id}:{file_id}:{seq}:{chunk_text}".encode()
        ).hexdigest()[:16]
        self._chunks_written += 1
        return await self._write_extracted(
            ExtractedChunk(
                session_id=session_id,
                file_id=file_id,
                chunk_id=chunk_id,
                text=chunk_text,
                seq=seq,
                kind="turn",
                speaker=speaker or role,
                search_text=chunk_surface,
                # The same anchor the relative dates resolve against is what dates
                # the chunk at retrieval. Until now nothing populated CHUNK.ts, so
                # the column was written, indexed and selected as "" everywhere.
                ts=str(anchor or ""),
                embedding=embedding,
                extraction=extraction,
                temporal=temporal,
                memory_class=metadata.get("memory_class") or "raw_evidence",
                ontology_classes=self._match_classes(embedding, chunk_text) if infer else [],
            )
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
            from graphknows.symbolic.ontology.scheme import load_coarse_aliases, load_scheme

            cfg = self._settings
            self._scheme = load_scheme(cfg.ontology_source, cfg.overlay_source)
            self._coarse_aliases = load_coarse_aliases(cfg.overlay_source)
            self._overlay_index = self._build_overlay_index()
        # Built EAGERLY from whatever text has arrived, then rebuilt once at the
        # warmup mark. Waiting for the full warmup would leave every corpus's
        # first turns with no lexical selection at all, and a short session with
        # none ever — which is how this first ran and produced zero relations.
        # `usable_index` needs the document count to know when a frequency is
        # meaningful, so a small corpus simply filters nothing.
        if (
            self._corpus_seen
            and not self._lex_frozen
            and len(self._corpus_seen) != self._lex_built_at
        ):
            from graphknows.symbolic.ontology.scheme import build_index

            self._lex_index = build_index(self._scheme, self._corpus_seen, self._extractor.nlp)
            self._lex_built_at = len(self._corpus_seen)
            self._lex_frozen = len(self._corpus_seen) >= _LEXICAL_WARMUP
        return self._lex_index is not None

    def _lexical_labels(self, text: str) -> tuple[list[str] | None, dict[str, str] | None]:
        """``(entity labels, relation labels)`` for one window, lexically selected."""
        if not self._ensure_scheme():
            return None, None
        from graphknows.symbolic.ontology import entity_labels, relation_labels

        doc = self._extractor.parse(text)
        ents = entity_labels(doc, self._scheme, self._lex_index)
        rels = relation_labels(doc, set(ents), self._scheme, self._lex_index)
        # relex takes {label: definition}; the definition is what it reads.
        spec = {r: (self._scheme[r].definition or r) for r in rels if r in self._scheme}
        return ents, spec or None

    @property
    def _chunk_stride(self) -> int:
        """Turns between emitted window chunks — the window minus its overlap.

        Window 5 / overlap 2 gives stride 3: windows cover turns 1-5, 4-8, 7-11.
        Clamped to at least 1 so an overlap >= the window cannot emit forever.
        """
        return max(self._extract_window - self._window_overlap, 1)

    def _extraction_window(
        self, session_id: str, clean: str, speaker: str
    ) -> tuple[str, int, list[tuple[int, int, str]], str]:
        """``(text to extract from, where this turn starts, spans, verbatim window)``.

        The fourth element is the same window WITHOUT deixis resolution — what
        gets stored as the chunk when window chunking is on. Storage stays
        verbatim on purpose: the extractor benefits from "Gina lost Gina's job",
        a reader quoting it back does not.

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

        def _render(key: str) -> str:
            return "\n".join(f"{t['speaker']}: {t[key]}" if t["speaker"] else t[key] for t in turns)

        if window <= 1 or len(turns) == 1:
            # A one-turn window IS the turn, stored byte for byte. Speaker
            # prefixes appear only once there are several turns to tell apart;
            # who said this one is on CHUNK.speaker, and the attributed form is
            # what search_text already carries.
            return clean, 0, [], clean
        lines = [
            f"{t['speaker']}: {t['resolved']}" if t["speaker"] else t["resolved"] for t in turns
        ]
        text = "\n".join(lines)
        offset = len(text) - len(lines[-1]) + (len(speaker) + 2 if speaker else 0)
        return text, offset, turn_spans(turns, text), _render("text")

    async def _ensure_session_and_file(
        self,
        session_id: str,
        parsed: ParsedFile,
        metadata: dict[str, Any],
        t0: float,
    ) -> None:
        await self._store.ensure_session(session_id, started_at=str(t0))
        await self._store.ensure_file(file_id=parsed.source_id, session_id=session_id)

    async def _frame_payload(
        self, chunk: ExtractedChunk, filtered: list[dict[str, Any]]
    ) -> tuple[list[tuple[str, float]], list[dict[str, Any]], Any]:
        """Everything the frame layers need, computed before any write opens.

        Split out of :meth:`_write_extracted` so the model inference runs
        OUTSIDE the per-chunk transaction: an ArcadeDB HTTP transaction held
        open across SRL and frame-instance inference expires server-side, and
        the write that follows dies with a rollback that cannot find its own
        transaction.

        Returns the chunk's frame tags, its frame instances and their nesting.
        """
        # One parse serves both frame layers when SRL identification is on -- it
        # is the expensive step, and running it twice per chunk would double it.
        srl_hits = await self._srl_hits(chunk.text) if chunk.srl_hits is None else chunk.srl_hits
        if not self._enable_frames:
            return [], [], None

        from graphknows.symbolic.framenet.index import match_frames

        # Parsed triggers name the frames this chunk actually evokes; the
        # embedding match only ranks candidates by definition similarity. The
        # confidence is 1.0 because a trigger is an observation, not a guess --
        # EVOKED_BY.confidence weights the chunk at query time, so a parsed tag
        # should outweigh a cosine one.
        matches = (
            [(f, 1.0) for f in dict.fromkeys(h.frame for h in srl_hits)]
            if srl_hits
            else match_frames(chunk.embedding, top_k=_FRAME_TOP_K, min_sim=_FRAME_MIN_SIM)
        )
        # {name.casefold(): type} from this chunk's own already-typed entities --
        # the type veto's filler type comes from the graph's OWN entity typing,
        # not a second WordNet lookup.
        entity_types = {str(e["name"]).casefold(): str(e.get("type", "")) for e in filtered}
        instances, nesting = await asyncio.to_thread(
            self._frame_instances,
            chunk.text,
            chunk.chunk_id,
            chunk.speaker,
            chunk.embedding,
            srl_hits,
            entity_types,
            chunk.extraction.frames,
        )
        return matches, instances, nesting

    async def _write_extracted(self, chunk: ExtractedChunk) -> int:
        """Write one chunk's CHUNK + ENTITY/MENTIONS + RELATION + TEMPORAL. Returns entity count.

        Takes its fields bundled in ``chunk`` (an :class:`ExtractedChunk`)
        rather than as a flat parameter list.

        ``srl_hits`` are this chunk's parsed frame triggers when the batch stage
        already has them (``[]`` is a real answer there — no triggers found);
        ``None`` means the caller has no batch stage and this chunk is parsed here.

        ``decode_state`` is the chunk's decode marker (``''`` | ``'ok'`` |
        ``'failed'``); the default leaves it absent, which is every chunk no LLM
        decoder ever saw.
        """
        session_id = chunk.session_id
        file_id = chunk.file_id
        chunk_id = chunk.chunk_id
        text = chunk.text
        seq = chunk.seq
        kind = chunk.kind
        speaker = chunk.speaker
        ts = chunk.ts
        embedding = chunk.embedding
        search_text = chunk.search_text
        extraction = chunk.extraction
        temporal = chunk.temporal
        memory_class = chunk.memory_class
        ontology_classes = chunk.ontology_classes
        decode_state = chunk.decode_state
        filtered = graph_entities(extraction.entities)
        # Definition-described typing (default off). Runs BEFORE the entity write
        # so a surface only this pass saw is written as a first-class ENTITY
        # rather than dropped.
        typed = await self._definition_typing(text)
        if typed:
            known = {str(e.get("name", "")).casefold() for e in filtered}
            extra = [
                {"name": surface, "type": label, "confidence": score}
                for surface, (label, score) in typed.items()
                if surface.casefold() not in known
            ]
            if extra:
                filtered = graph_entities(list(extraction.entities) + extra)
        # Denominator and untyped count for the #160 share (see
        # _UNTYPED_ENTITY_TYPES above) — the metric the span-length gate
        # decision needs, computed once filtered is final so it counts exactly
        # what gets written below.
        self._abstentions["entities"] += len(filtered)
        self._abstentions["untyped_entities"] += sum(
            1 for e in filtered if str(e.get("type") or "").upper() in _UNTYPED_ENTITY_TYPES
        )
        # Everything the writes need is computed FIRST, outside the transaction:
        # model inference held inside an open ArcadeDB HTTP transaction expires it
        # server-side on a slow or cold machine (observed in CI, 2026-09-07).
        matches, instances, nesting = await self._frame_payload(chunk, filtered)

        # Every write this chunk makes is one unit of work: a failure anywhere
        # below must not leave a visible CHUNK with no relations, frames or
        # temporal facts for a reader to mistake for a chunk that had none.
        async with self._chunk_transaction():
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
                    "decode_state": decode_state,
                    "embedding": embedding,
                }
            )
            if filtered:
                await self._store.write_entities_bulk(session_id, chunk_id, filtered)

            # Ontology layer — tag the chunk with the classes it evokes.
            await self._link_ontology(chunk_id, ontology_classes or [])
            await self._link_senses(chunk_id, filtered)

            # FRAME layer — the chunk's evoked frames, matched above.
            tagged_frames: set[str] = set()
            if self._enable_frames and matches:
                await self._write_frame_tags(chunk_id, matches)
                tagged_frames = {name for name, _ in matches}

            # FRAME_INSTANCE layer — frame-guided semantic role labeling, run above.
            if self._enable_frames and instances:
                # An instance whose FRAME vertex does not exist gets no EVOKES
                # edge, so the fact cannot generalise through the taxonomy.
                # write_frames is idempotent (MERGE) and skips per-frame
                # embedding work for frames already materialised this process,
                # so re-tagging an already-tagged frame is cheap.
                # Unlike the SRL trigger above, these frames come from
                # lu_triggers and can rank below _FRAME_MIN_SIM — use
                # the instance's own match_sim as the tag confidence rather
                # than the trigger's 1.0, so the embedding guess the tag layer
                # already rejected doesn't outrank a real cosine match.
                frame_sims: dict[str, float] = {}
                for i in instances:
                    frame = str(i["frame"])
                    sim = float(i.get("match_sim") or 0.0)
                    frame_sims[frame] = max(sim, frame_sims.get(frame, 0.0))
                untagged = [(f, sim) for f, sim in frame_sims.items() if f not in tagged_frames]
                await self._write_frame_tags(chunk_id, untagged)
                await self._store.write_frame_instances(
                    session_id, chunk_id, instances, nesting=nesting
                )

            relation_report = await self._write_relations(extraction, chunk_id, text, speaker)
            for gate in ("blocked_domain", "blocked_range", "unsupported", "pruned_endpoint"):
                self._abstentions[gate] += int(relation_report.get(gate, 0))
            self._abstentions["relations"] += int(relation_report.get("relations", 0))

            iso_dates = [t["iso"] for t in temporal if t.get("iso")]
            if iso_dates:
                await self._store.write_temporal(chunk_id, iso_dates)
        return len(filtered)

    @asynccontextmanager
    async def _chunk_transaction(self) -> AsyncIterator[None]:
        """Run one chunk's writes inside the store's transaction.

        The rollback is counted and the exception re-raised: a chunk that was
        thrown away must be visible in ``IngestResult.abstentions``, or the
        graph is short a chunk and nothing downstream can tell.

        A store that cannot transact (the in-memory doubles the unit tests
        drive this handler with) writes exactly as it did before.
        """
        # Looked up on the class, not the instance: a test double whose
        # ``__getattr__`` answers everything would hand back a coroutine here.
        can_transact = hasattr(type(self._store), "transaction")
        try:
            async with self._store.transaction() if can_transact else nullcontext():
                yield
        except Exception:
            self._abstentions["chunk_write_rolled_back"] += 1
            raise

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

        Verification is a property of an EDGE, not of the producer that minted
        it, so both tiers route through one gate: the relex loop carries the
        score the extractor already assigned (scored there, where its text and
        offsets are freshest), and the DSPy loop is scored here with the same
        gate instance before it writes. A tier bolted onto a producer could
        only ever see that producer's candidates.

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
                RelationWrite(
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
            arr: np.ndarray = embed_batch(texts)
            return True, [arr[i].tolist() for i in range(len(texts))]
        if self._ontology is not None:
            return True, [self._embed_fn(t) for t in texts]
        return False, []

    async def _aembed(self, text: str) -> list[float]:
        """Embed one text off the event loop.

        ``_embed_fn`` blocks — an HTTP round-trip by default, a loaded model
        under an override — and every caller here is a coroutine. The batch
        sibling above stays synchronous and is handed to a thread whole, at its
        one call site, so a file pays a single hop instead of one per chunk.
        """
        embedding: list[float] = await asyncio.to_thread(self._embed_fn, text)
        return embedding

    # ------------------------------------------------------------------
    # Ontology injection (mode-independent; inert when no ontology is set)
    # ------------------------------------------------------------------

    def _seed_corpus(self, texts: list[str]) -> None:
        """Feed the file path's cleaned chunk texts into the lexical corpus.

        Only the turn path fed it before now, so a file ingest never had a
        lexical index and `_match_classes` abstained unconditionally.
        """
        if self._lexical and not self._lex_frozen:
            self._corpus_seen.extend(texts)

    def _build_overlay_index(self) -> Any:
        """Overlay-only SKOS classes (e.g. "Dance Studio"), embedded once.

        The anchor law in ``_match_classes`` admits anything in ``_scheme``,
        which is ontology + overlay, but ``self._ontology.classes`` is loaded
        from ``settings.ontology_source`` alone — so a class the overlay adds
        can be anchored and still never win, because it was never part of the
        vocabulary cosine ranks. This gives it its own tiny index, built the
        same way (:class:`~graphknows.symbolic.ontology.catalog.OntologyIndex`)
        so it is ranked and cached identically. ``None`` when nothing is left
        to add (no ontology configured, or every scheme class already has a
        same-labelled entry in ``self._ontology.classes``).
        """
        if self._ontology is None or self._scheme is None:
            return None
        from graphknows.symbolic.ontology.catalog import OntologyIndex
        from graphknows.symbolic.ontology.loader import OntologyTerm

        known = {t.label.casefold() for t in self._ontology.classes}
        terms = [
            OntologyTerm(uri=c.uri, label=c.pref_label, definition=c.definition, kind="class")
            for c in self._scheme.values()
            if c.kind == "class" and c.pref_label.casefold() not in known
        ]
        return OntologyIndex("overlay", terms) if terms else None

    def _match_classes(self, embedding: list[float], text: str) -> list[tuple[str, float]]:
        """Ontology classes this chunk evokes, as ``[(uri, sim)]`` for tagging.

        These become persisted MATCHES_CLASS edges, which retrieval trusts, so
        they obey the anchor law: the candidate set comes from lexical
        evidence, cosine only RANKS it.
        Absolute cosine over CCO's 1437 class definitions is uncalibrated (see
        the docstring of ``catalog.py``: median 0.409, p95 0.493 on real
        conversation) -- on conv-30 every unanchored tag landed in a 0.557-0.590
        band, a spread narrow enough that a BFO upper class like "Curvilinear
        Motion" outranked "Dance Studio" on a chunk that never mentions motion.
        No floor can separate a degenerate band; only the chunk's own words can
        -- so ``TAG_MIN_SIM`` is not applied here either: once the anchor has
        admitted a class, vetoing it again on the same uncalibrated cosine
        (0.487 for "Week", 0.457 for "Month", both lexically anchored) throws
        away evidence the anchor already certified, and miscounts it as
        ``unanchored_class_tag``.

        Ranked against ``self._ontology.classes`` PLUS ``_build_overlay_index``'s
        overlay-only classes: the anchor set comes from ``_scheme`` (ontology +
        overlay), so a class that exists only in the overlay -- "Dance Studio"
        -- must be rankable too, or it can be anchored and still never win.

        A chunk with no spaCy pipeline or no lexical index available abstains
        outright (``unanchored_class_tag``) -- there is no anchor to check
        against. Otherwise the anchor set is whatever ``lexical.evoked`` finds
        in this chunk's own text, cosine ranks the whole vocabulary (not just
        top-k, since the anchored classes are almost never in an unanchored
        top-k), and only matches inside the anchor survive, truncated to
        ``_ONT_TAG_TOP_K``.
        """
        if self._ontology is None or not embedding:
            return []
        from graphknows.symbolic.ontology.lexical import evoked

        if not self._ensure_scheme():
            self._abstentions["unanchored_class_tag"] += 1
            if not self._logged_unanchored_class_tag:
                log.info("Class tagging off: no lexical index available for anchoring")
                self._logged_unanchored_class_tag = True
            return []
        doc = self._extractor.parse(text)
        anchor = {label.casefold() for label in evoked(doc, self._scheme, self._lex_index, "class")}
        matches = self._ontology.match_classes(
            embedding, top_k=len(self._ontology.classes), min_sim=0.0
        )
        if self._overlay_index is not None:
            matches = matches + self._overlay_index.match_classes(
                embedding, top_k=len(self._overlay_index.classes), min_sim=0.0
            )
        matches.sort(key=lambda pair: pair[1], reverse=True)
        accepted = [(term.uri, sim) for term, sim in matches if term.label.casefold() in anchor][
            :_ONT_TAG_TOP_K
        ]
        if not accepted:
            self._abstentions["unanchored_class_tag"] += 1
        else:
            self._abstentions["class_tags"] += len(accepted)
            self._tag_sims.extend(sim for _, sim in accepted)
        return accepted

    def _report_abstentions(self) -> dict[str, int]:
        """Non-zero abstention counters for this session, logging the tag sim spread.

        Was two identical dict comprehensions (``_ingest_parsed`` and
        ``STMService.ingest_messages``); collecting the log line here too means
        the spread issue #158 measured -- a band narrower than ~0.05 means
        cosine ranked nothing inside the anchor set -- is reported from every
        caller rather than only ever recovered by re-running the bug.
        """
        if self._tag_sims:
            log.info(
                "MATCHES_CLASS sim spread: n=%d min=%.3f max=%.3f spread=%.3f",
                len(self._tag_sims),
                min(self._tag_sims),
                max(self._tag_sims),
                max(self._tag_sims) - min(self._tag_sims),
            )
        return {k: v for k, v in self._abstentions.items() if v}

    def _drain_decode_errors(self) -> list[str]:
        """This run's run-permanent decoder failures, then forgotten.

        Drained rather than accumulated: one handler outlives many ingests, and
        a rejected key re-reported on the next run — after it was replaced — is
        a false alarm.
        """
        errors, self._decode_errors = self._decode_errors, []
        return errors

    def _decoder_abstentions(self) -> dict[str, int]:
        """The decoder's own gate counters; empty for a decoder that keeps none.

        The response gates live in the decoder, because that is the only place
        the response exists (contracts/decoder.md §5). This handler is where
        they are *reported* from, so it reads them rather than recounting them.
        """
        return dict(getattr(self._extractor, "abstentions", {}))

    def _merge_decoder_abstentions(self, before: dict[str, int]) -> None:
        """Fold what the decoder counted since *before* into this run's report.

        The delta, not the total: the decoder's counter is cumulative over its
        own lifetime, and one decoder serves every ingest of the process.
        """
        for gate, count in self._decoder_abstentions().items():
            if (new := count - before.get(gate, 0)) > 0:
                self._abstentions[gate] += new

    def _note_decode_failure(self, exc: Exception) -> None:
        """Sort one failed decode into its gate, or onto the run's errors.

        Never re-raised: the chunk is written undecoded either way, which is what
        keeps a provider outage from opening a *hole* in the graph rather than a
        counted gap in it (FR-018, SC-004). Falling back to the local decoder is
        forbidden here as everywhere (FR-022), so there is no third branch.
        """
        from graphknows.ingestion.extraction.llm.retry import failure_gate

        self._decode_failures += 1
        if (gate := failure_gate(exc)) is not None:
            self._abstentions[gate] += 1
            log.warning("decode failed (%s: %s); abstained under %s", type(exc).__name__, exc, gate)
            return
        message = f"decoder: {type(exc).__name__}: {exc}"
        if message not in self._decode_errors:
            self._decode_errors.append(message)
        log.error("%s -- the run cannot decode; chunks are stored undecoded", message)

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

    def _rank_within_groups(self, groups: dict[Any, list[Any]], embedding: list[float]) -> None:
        """Order each (domain, range) group by how well the property fits THIS chunk.

        Only the order changes — every property that satisfied the gate is still
        offered — so this cannot admit a property the endpoints do not license.
        That distinction is the whole reason it is safe: ranking across the full
        vocabulary is the measured failure recorded on :meth:`_label_hints`
        (cosine puts ``liveBlogUpdate`` above anything a dance sentence asserts,
        with no floor separating right from wrong). Inside a group the endpoints
        have already done the filtering, so cosine only has to break a tie
        between properties that are all structurally admissible.

        Properties the index has no vector for sort last, and Python's sort is
        stable, so a group with no embeddings at all keeps its vocabulary order.
        """
        import numpy as np

        v = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(v))
        if v.size == 0 or norm == 0.0:
            return
        v = v / norm

        def fit(prop: Any) -> float:
            row = self._relations.term_embedding(prop.uri) if self._relations else []
            if not row:
                return -1.0
            e = np.asarray(row, dtype=np.float32).ravel()
            return float(e @ v) if e.shape == v.shape else -1.0

        for props in groups.values():
            props.sort(key=lambda p: -fit(p))

    def _relation_labels(
        self, entities: list[dict[str, Any]], embedding: list[float] | None = None
    ) -> dict[str, str]:
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

        Capped at ``_RELATION_LABEL_CAP`` and allocated across the (domain, range)
        pairs by :func:`_allocate_offer` — one slot per pair as a floor, the rest
        in proportion to how many applicable properties each pair holds.

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
        if embedding and _group_ranking_enabled():
            self._rank_within_groups(groups, embedding)
        offer = _allocate_offer(groups, _RELATION_LABEL_CAP)
        return {p.label: p.definition or p.label for p in offer}

    def _map_to_profile_class(self, entity_type: str) -> tuple[str, float]:
        """Map an extracted entity type onto a RELATION-profile class: ``(label, score)``.

        Exact label first ("PERSON" -> ``Person``), then the type string
        embedded against the class definitions
        — but over the RELATION vocabulary's classes, which is where the
        properties' declared domain and range are written. The anchor law does
        not apply here: this class is only ever read for the domain/range
        conformance gate, never persisted as a grounding.

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

    async def _link_ontology(
        self, chunk_id: str, ontology_classes: list[tuple[str, float]]
    ) -> None:
        """Write the chunk's MATCHES_CLASS tags for the ontology classes it evokes."""
        if self._ontology is None or not ontology_classes:
            return
        await self._store.write_class_matches(chunk_id, ontology_classes)

    async def _link_senses(self, chunk_id: str, entities: list[dict[str, Any]]) -> None:
        """Ground each entity to its WordNet most-frequent-sense (MFS), lazily.

        Sibling of :meth:`_link_ontology`: the per-chunk write seam for the
        WordNet plane (issue #142). Purely additive enrichment — a WSD miss
        or corpus error must never make an entity un-writable, so failures
        here are swallowed and counted, not raised.
        """
        from graphknows.symbolic.wordnet import WN_VERSION, most_frequent_sense

        try:
            rows: list[dict[str, Any]] = []
            for ent in entities:
                name = str(ent.get("name", ""))
                sense = most_frequent_sense(name)
                if sense is None:
                    self._abstentions["senses_ungrounded"] += 1
                    continue
                self._abstentions["senses_grounded"] += 1
                score = float(ent.get("confidence", ent.get("score", 1.0)) or 0.0)
                rows.append(
                    {
                        "name": name,
                        **sense,
                        "channel": "wordnet",
                        "version": WN_VERSION,
                        "matcher": "mfs",
                        "score": score,
                        "rank": 0,
                        "chunk_id": chunk_id,
                        # ponytail: surface name as span; a character offset
                        # is the upgrade path if two occurrences of one
                        # surface in a single chunk ever need distinguishing.
                        "span": name,
                    }
                )
            if rows:
                await self._store.write_senses(rows)
        except Exception:
            log.exception("WordNet sense grounding failed for chunk %s", chunk_id)

    async def _definition_typing(self, text: str) -> dict[str, tuple[str, float]]:
        """``{casefolded surface: (class label, confidence)}`` for this chunk, or ``{}``.

        Off unless ``GRAPHKNOWS_DEFINITION_TYPING`` is set. The schema is built
        once per handler from the loaded ontology: it is one encoder input, so
        the label set is capped rather than handed all 1437 classes.
        """
        from graphknows.ingestion.extraction.entities import typing_model

        if self._ontology is None or not text.strip() or not typing_model.typing_enabled():
            return {}
        if self._typing_schema is None:
            self._typing_schema = typing_model.build_schema(self._ontology, _TYPING_CLASSES)
            # Name the absentees. The bundled CCO digest has no bare `Artifact`
            # class (only "... Artifact Function"), so the list silently built a
            # 13-label schema out of 14 and paintings had nowhere to be typed.
            # A richer ontology — one built by graphknows.symbolic.ontology.rdf — supplies it.
            if missing := [c for c in _TYPING_CLASSES if c not in self._typing_schema]:
                log.info(
                    "definition typing schema: %d classes (%d absent from this ontology: %s)",
                    len(self._typing_schema),
                    len(missing),
                    ", ".join(missing),
                )
            else:
                log.info("definition typing schema: %d classes", len(self._typing_schema))
        if not self._typing_schema:
            return {}
        return await asyncio.to_thread(
            typing_model.type_lines, text.split("\n"), self._typing_schema
        )

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
        """``label``, its transitive rdfs:subClassOf parents, and its direct skos:broader parents.

        All casefolded. The walk never continues above a skos:broader link: SKOS does not
        license composing one, so a grandparent reached that way is not an
        ancestor. Widening this set makes ``_relation_labels`` offer the
        extractor properties the chunk's entity types never evoked.
        """
        seen: set[str] = set()
        frontier = [label.casefold()]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(p.casefold() for p in self._parents_by_class.get(current, ()))
        # Unioned after the walk, not inside it: a label reachable both ways
        # would otherwise have its expansion suppressed depending on pop order.
        return seen | {p.casefold() for c in seen for p in self._broader_by_class.get(c, ())}

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

    @staticmethod
    def _surviving_nesting(hits: list[Any], kept_offsets: list[int]) -> list[tuple[int, str, int]]:
        """Re-index frame nesting onto the instances that actually survived.

        Nesting is computed over trigger offsets, but the fillability and junk
        gates drop instances in between — so a pair naming a dropped frame is
        dropped with it rather than pointing at the wrong row.
        """
        if not hits or len(kept_offsets) < 2:
            return []
        from graphknows.symbolic.framenet.srl import nested_pairs

        index_of = {off: i for i, off in enumerate(kept_offsets) if off >= 0}
        out = []
        for parent, role, child in nested_pairs(hits):
            p, c = index_of.get(hits[parent].offset), index_of.get(hits[child].offset)
            if p is not None and c is not None and p != c:
                out.append((p, role, c))
        return out

    async def _prewarm_frame_srl(self, texts: list[str], infer: bool = True) -> list[list[Any]]:
        """Parse every chunk's sentences once, up front, and hand back what was parsed.

        Returns one hit list per text — all empty unless SRL is on, so callers
        index it positionally either way. Handing the hits back is what keeps the
        parse single-use: the per-chunk stage used to re-derive them from the
        cache this warmed, and the decode request needs them before it is made.
        """
        empty: list[list[Any]] = [[] for _ in texts]
        if not (infer and texts and self._srl_active()):
            return empty
        from graphknows.symbolic.framenet.srl import prewarm

        start = time.monotonic()
        parsed = await asyncio.to_thread(prewarm, texts)
        log.info(
            "frame SRL prewarm: %d sentences parsed for %d chunks in %.1fs",
            parsed,
            len(texts),
            time.monotonic() - start,
        )
        return [await self._srl_hits(text) for text in texts]

    async def _write_frame_tags(self, chunk_id: str, matches: list[tuple[str, float]]) -> None:
        """Write FRAME chunk tags for ``matches`` (name, confidence) pairs."""
        if not matches:
            return
        from graphknows.symbolic.framenet.framenet import _broader_up
        from graphknows.symbolic.framenet.index import frame_embedding, frame_summary

        # FrameNet's own hierarchy rides along, so the frames evoked here
        # land as a traversable taxonomy rather than isolated tags: a
        # query matching a general frame can then walk DOWN to the
        # specific one a chunk actually evoked. Cached in-process.
        up = _broader_up()
        await self._store.write_frames(
            [
                {
                    "frame_type": name,
                    "chunk_id": chunk_id,
                    "confidence": sim,
                    "summary": frame_summary(name),
                    "embedding": frame_embedding(name),
                    "parents": [
                        {
                            "id": p,
                            "summary": frame_summary(p),
                            "embedding": frame_embedding(p),
                        }
                        for p in sorted(up.get(name, ()))
                    ],
                }
                for name, sim in matches
            ],
        )

    def _srl_active(self) -> bool:
        """Whether the transformer frame parser runs for this handler.

        It is a local extraction model like relex and the verifier, so it runs
        only beside the local decoder: under the LLM decoder the frame roles
        come from the decoder's own frame section over LU-trigger candidates,
        and no local model loads (FR-008). Measured on conv-30 it cost 12 s of a
        32 s assisted-mode chunk. The signal is the one every consumer of the
        decoder seam already reads — the local decoder exposes ``gliner``, the
        LLM decoder does not (contracts/decoder.md §1).
        """
        if not (self._enable_frames and hasattr(self._extractor, "gliner")):
            return False
        from graphknows.symbolic.framenet.srl import srl_enabled

        return srl_enabled()

    async def _srl_hits(self, text: str) -> list[Any]:
        """Parse the chunk's frame triggers once, off the event loop. ``[]`` when off."""
        if not (text.strip() and self._srl_active()):
            return []
        from graphknows.symbolic.framenet.srl import parse

        try:
            return await asyncio.to_thread(parse, text)
        except Exception as exc:
            # On by default now, so it must degrade rather than fail: a parser
            # that cannot load (no model cache, no VRAM, a broken install) drops
            # this chunk back to the LU/embedding identifier instead of losing
            # the document. Warned, not swallowed — silent degradation is how a
            # pipeline runs for weeks with a dead layer nobody noticed.
            log.warning("frame SRL parse failed, falling back to LU frames: %s", exc)
            return []

    def _veto_semtype_conflicts(
        self, inst: Any, roles: dict[str, list[str]], entity_types: dict[str, str] | None
    ) -> dict[str, list[str]]:
        """Drop role fillers whose type conflicts with the FE's semantic type.

        Only called when ``settings.enable_fe_type_gate`` is on. A filler's
        type comes from the graph's OWN entity typing (``entity_types``, built
        from `filtered` at the call site) so this never re-runs a WordNet
        lookup. An unmapped filler yields "" and never vetoes.
        """
        from graphknows.symbolic.framenet import fe

        out = dict(roles)
        for role, fillers in roles.items():
            kept = []
            for filler in fillers:
                filler_type = (entity_types or {}).get(filler.casefold(), "")
                if fe.semtype_conflict(inst.frame, role, filler_type):
                    self._abstentions["frame_semtype_veto"] += 1
                else:
                    kept.append(filler)
            if kept:
                out[role] = kept
            else:
                del out[role]
        return out

    def _locally_filled(
        self,
        text: str,
        doc: Any,
        speaker: str,
        speakers: list[str],
        embedding: list[float] | None,
        srl_hits: list[Any] | None,
    ) -> list[Any]:
        """Fill this chunk's frame roles on the local span model: parsed triggers, else LU.

        Split out of :meth:`_frame_instances` only so consuming already-filled
        roles is one branch there rather than three; both identifiers, their
        arity and the model they share are unchanged.
        """
        from graphknows.ingestion.extraction.relations.frame_srl import (
            extract_frame_instances,
            extract_frame_instances_srl,
            lu_triggers,
        )

        gliner = self._extractor.gliner
        if srl_hits:
            # Parsed triggers instead of one embedding-ranked guess per chunk, so
            # a chunk can yield one instance per fact rather than at most one.
            return extract_frame_instances_srl(text, speaker, speakers, gliner, srl_hits)
        nlp = self._extractor.nlp
        return extract_frame_instances(
            text,
            speaker,
            speakers,
            gliner,
            lambda t: lu_triggers(t, nlp, embedding=embedding, doc=doc if t == text else None),
        )

    def _frame_instances(
        self,
        text: str,
        chunk_id: str,
        speaker: str,
        embedding: list[float] | None = None,
        srl_hits: list[Any] | None = None,
        entity_types: dict[str, str] | None = None,
        decoded_frames: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[tuple[int, str, int]]]:
        """Frame-guided role extraction for one chunk → write dicts (thread-run).

        Reuses the entity extractor's already-loaded spaCy + GLiNER2 (no second
        load) and the chunk's already-computed ``embedding`` for frame ranking (no
        re-embed). Speaker/speakers come from the chunk's turn-lines (same logic the
        SVO pass uses), falling back to the passed ``speaker``. Returns
        ``([{frame, match_sim, role_confidence, roles, lvc}], nesting)`` — one item
        per trigger (a verb group, a nominalisation, or a light-verb construction)
        on the default path, one per parsed trigger on the SRL one, per the
        role-fillability gate; the FRAME_INSTANCE id is content-hashed at write
        time. ``nesting`` is ``(parent, role, child)`` indexed into the returned
        list, empty unless the SRL path ran. Empty if the extractor doesn't expose
        reusable models. ``decoded_frames`` are role fills the decoder already
        returned for locally-offered candidates (``ExtractionResult.frames``);
        given them, no model runs here at all and every gate below is unchanged.
        ``entity_types`` (``{name.casefold(): type}``, built from
        this chunk's already-typed ``filtered`` entities) feeds the default-off
        semantic-type veto; omit it and every filler's type is unknown, so the
        veto never fires.
        """
        extractor = self._extractor
        # Roles are either handed over already filled or filled by the local span
        # model; a decoder carrying neither produces nothing here and must not
        # fall back to one (contracts/decoder.md §4).
        if (
            extractor is None
            or not hasattr(extractor, "nlp")
            or not (decoded_frames or hasattr(extractor, "gliner"))
        ):
            return [], []
        from graphknows.ingestion.extraction.entities.extractor import _line_speakers
        from graphknows.linguistics import analyse

        line_speakers = _line_speakers(text)
        speakers = sorted({s for _, s in line_speakers if s})
        primary = speaker or (speakers[0] if speakers else "")
        if primary and primary not in speakers:
            speakers = sorted({*speakers, primary})

        # Shared with lexical labelling and frame-candidate ranking — see
        # extractor.parse(). Previously each of the three parsed the same window
        # again.
        doc = extractor.parse(text)
        instances = (
            _decoded_instances(decoded_frames)
            if decoded_frames
            else self._locally_filled(text, doc, primary, speakers, embedding, srl_hits)
        )
        # Role fillers become ENTITY nodes, so they must clear the SAME junk gate
        # every other minting path uses. This was the one path that did not call
        # it: measured on LoCoMo conv-30, it minted 624 ENTITY nodes with no type,
        # no confidence and no MENTIONS edge to any chunk — 30% of the graph —
        # and fed them to the generator's fact-sheet ("Buyer=Hey Gina!",
        # "Speaker=Yeah", "Cognizer=Wow").
        # Default-off experiment (settings.enable_fe_type_gate): vetoes a role
        # filler whose graph entity type conflicts with the FE's FrameNet
        # semantic type. Read off the settings instance this handler was built
        # with, so an eval A/B flips it through ``Memory(settings=...)`` as well
        # as through the environment.
        from graphknows.symbolic.framenet import fe

        type_gate_on = self._settings.enable_fe_type_gate

        out: list[dict[str, Any]] = []
        kept_offsets: list[int] = []
        for inst in instances:
            roles = {
                role: self._dedupe_nested([f for f in fills if is_graph_entity_name(f)])
                for role, fills in inst.roles.items()
            }
            roles = {r: f for r, f in roles.items() if f}

            if type_gate_on:
                roles = self._veto_semtype_conflicts(inst, roles, entity_types)

            # The frame's own gate: an instance under two filled roles is not a
            # fact, it is a frame label with a stray span attached. Re-checked
            # here (not just once above) because the type veto above can drop
            # a role that was filled a moment ago.
            if len(roles) < 2:
                self._abstentions["frame_under_two_roles"] += 1
                continue
            # How this instance is asserted — negated, hypothesised, hedged, or
            # put in someone else's mouth. Reuses the `doc` parsed once above:
            # no model load, no second parse. A trigger_offset of -1 (the
            # non-SRL path, which guesses one frame per chunk with no span)
            # returns the defaults rather than analysing an arbitrary token.
            # Hoisted above the validity/coreset gates below so voice is
            # computed once and shared by both the write-props and the CNI
            # split, including for an instance those gates go on to drop.
            props = analyse(doc, inst.trigger_offset).as_props()

            # Unfilled-core reporting: CNI (Constructional Null Instantiation)
            # vs unknown. Counted against this role set (post fillability
            # floor, post type veto) regardless of what the validity/coreset
            # gates below decide — a role FrameNet requires but the extractor
            # never filled is a miss either way. Core-Unexpressed FEs are
            # excluded by definition — the frame cannot express them locally,
            # so their absence is not a miss. A PASSIVE clause licenses
            # omitting its unfilled core roles (the agent is omissible by the
            # construction itself), so that is not evidence the extractor
            # missed anything; anything else left unfilled is a genuine
            # unknown — the denominator the bare count never gave us.
            # ponytail: only passive voice is checked. Imperative subject-drop
            # and definite/indefinite null instantiation are not detected, and
            # typed DNI needs the valence patterns the parent spec leaves
            # stubbed — both would need more of FrameNet's construction data
            # than this module reads today.
            fes = fe.frame_elements(inst.frame)
            unfilled_core = sum(
                1 for name, elem in fes.items() if elem.coreness == "Core" and name not in roles
            )
            if unfilled_core:
                key = (
                    "unfilled_core_cni"
                    if props.get("voice") == "passive"
                    else "unfilled_core_unknown"
                )
                self._abstentions[key] += unfilled_core

            # FrameNet's own Requires/Excludes constraints: a role configuration
            # that violates them is not a fact FrameNet licenses, regardless of
            # how many roles got filled.
            reason = fe.violation(inst.frame, roles.keys())
            if reason:
                kind = reason.split(":", 1)[0]
                self._abstentions[f"frame_invalid_{kind}"] += 1
                continue
            # CoreSet fillability: the members of a core set are alternative
            # expressions of the SAME core participant, so an instance filling
            # two roles drawn from one set while leaving another set wholly
            # empty has one participant, not two — a bare count of filled
            # roles cannot see that, which is exactly the gap this closes.
            if not fe.core_sets_satisfied(inst.frame, roles.keys()):
                self._abstentions["frame_coreset_unsatisfied"] += 1
                continue

            out.append(
                {
                    "frame": inst.frame,
                    "match_sim": inst.match_sim,
                    "role_confidence": inst.role_confidence,
                    "roles": roles,
                    "lvc": bool(inst.lvc),
                    **props,
                }
            )
            kept_offsets.append(inst.trigger_offset)
        return out, self._surviving_nesting(srl_hits or [], kept_offsets)

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
        from graphknows.temporal import resolve_temporal

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
        """One chunk's extraction, with the single-call decoder's failures contained.

        The message path decodes through here rather than through
        :meth:`_decode_batch`, and a raise on this route used to abandon the turn
        *before* it was written — a failed decode cost the chunk itself, not just
        its labels. The local two-pass decoder still raises: its failures are
        this process's own, not a provider's, and nothing counts them.
        """
        if self._extractor is None:
            return ExtractionResult()
        from typing import cast

        before = self._decoder_abstentions()
        try:
            result = cast(
                ExtractionResult,
                self._extractor.extract(
                    clean_text, extra_entity_labels, extra_relation_labels, speaker
                ),
            )
        except Exception as exc:
            if not self._decodes_in_one_call:
                raise
            self._note_decode_failure(exc)
            result = ExtractionResult()
        self._merge_decoder_abstentions(before)
        return result
