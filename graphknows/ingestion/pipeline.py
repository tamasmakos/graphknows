"""The write path end to end: a source, its segments, and the symbols they label.

This is where the bottom-up half of the traversal happens (FR-002). Extraction
answers in mentions and facts; every mention carries the pack label the decoder
gave the surface, and that label is what resolves a concept — so an entity is
written with an ``INSTANCE_OF`` edge to the concept it instantiates, and its
segment with an ``EVOKES`` edge to every concept its mentions reached. Top-down
activation reads exactly those two edges back.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime

from graphknows.ingestion.extraction.protocol import Extractor, PackGuidance
from graphknows.models.fact import Fact, Mention
from graphknows.models.report import Counters, IngestReport
from graphknows.models.segment import Segment
from graphknows.models.source import Source
from graphknows.models.symbols import ConceptRef
from graphknows.storage.arcadedb._base import ArcadeStoreBase
from graphknows.storage.arcadedb.graph_store import name_norm
from graphknows.storage.arcadedb.writers import (
    EntityWrite,
    EntityWriter,
    Evidence,
    FactWriter,
    SegmentWriter,
    SourceWriter,
    SymbolWriter,
)
from graphknows.storage.embedder import embed

Embedder = Callable[[Sequence[str]], list[list[float]]]
"""Texts in, one float vector per text out — the representation layer's encoder."""


def _embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """The configured embedder, as plain float lists the writers can inline."""
    return [[float(x) for x in row] for row in embed(list(texts))]


def _reader(pack: PackGuidance, fallback: Extractor) -> Extractor:
    """The extractor that reads *pack*'s domain: its own when it ships one.

    A pack that reads its domain off the syntax is authoritative for it
    (FR-030), and the core learns that by asking rather than by importing —
    ``extractor`` is an optional member of a pack, not of the guidance the core
    declares. The configured decoder stays the fallback for packs shipping none.
    """
    own: Callable[[], Extractor] | None = getattr(pack, "extractor", None)
    return fallback if own is None else own()


def _concepts_by_label(packs: Sequence[PackGuidance]) -> dict[str, ConceptRef]:
    """The loaded concepts, keyed by casefolded label — the labelling table.

    First declaration wins, as in the predicate index: a later pack never
    silently steals the label an earlier one already answers to.
    """
    table: dict[str, ConceptRef] = {}
    for pack in packs:
        for concept in pack.concepts():
            table.setdefault(concept.label.casefold(), concept)
    return table


def _functional_predicate_ids(packs: Sequence[PackGuidance]) -> frozenset[str]:
    """Ids of the predicates that admit at most one current object per subject.

    The functional flag is a property of the predicate symbol (FR-011), so the
    currency rule is read off the loaded packs, never off a hand-kept list.
    """
    return frozenset(
        predicate.id for pack in packs for predicate in pack.predicates() if predicate.functional
    )


class IngestPipeline:
    """Writes one source as segments, mentions and facts, labelled with symbols.

    Every loaded pack reads every segment: packs share a namespace without
    interfering, so the pack a segment "belongs to" is never guessed (FR-004).
    """

    def __init__(
        self,
        store: ArcadeStoreBase,
        extractor: Extractor,
        packs: Sequence[PackGuidance],
        embed: Embedder = _embed_texts,
    ) -> None:
        self._sources = SourceWriter(store)
        self._segments = SegmentWriter(store)
        self._entities = EntityWriter(store)
        self._facts = FactWriter(store)
        self._symbols = SymbolWriter(store)
        self._embed = embed
        self._packs = tuple(packs)
        self._readers = tuple(_reader(pack, extractor) for pack in self._packs)
        self._concepts = _concepts_by_label(self._packs)
        self._functional = _functional_predicate_ids(self._packs)

    def __repr__(self) -> str:
        readers = ", ".join(reader.name for reader in self._readers)
        return f"{type(self).__name__}({len(self._packs)} packs: {readers})"

    async def ingest(self, source: Source, segments: Sequence[Segment]) -> IngestReport:
        """Write *source* and *segments* with everything the extractors read in them.

        Content already stored is dropped here, before any extractor runs
        (FR-007): re-ingesting the same bytes costs one query, not a model pass.
        """
        if await self._sources.exists(source.content_hash):
            return IngestReport(
                source_id=source.id,
                counters=Counters(sources_deduplicated=1),
            )
        await self._sources.write(source)
        entities: set[str] = set()
        facts_written = 0
        previous: str | None = None
        vectors = self._embed([segment.text for segment in segments])
        for segment, vector in zip(segments, vectors, strict=True):
            await self._segments.write(segment, vector)
            if previous is not None:
                await self._segments.link_next(previous, segment.id)
            previous = segment.id
            for pack, reader in zip(self._packs, self._readers, strict=True):
                mentions, facts = reader.extract(segment, pack)
                await self._write_mentions(mentions)
                await self._label(segment, mentions)
                await self._write_facts(segment, facts)
                entities.update(mention.entity_id for mention in mentions)
                facts_written += len(facts)
        return IngestReport(
            source_id=source.id,
            segments_written=len(segments),
            facts_written=facts_written,
            entities_touched=len(entities),
        )

    async def write_symbols(self) -> None:
        """Write every loaded pack's concepts and predicates — the symbolic index.

        Idempotent, run once when a namespace opens for writing: labelling
        MATCHes these vertices, so a concept never written is a label bottom-up
        can give and top-down can never activate. A concept is indexed by the
        embedding of its definition — the vector is the symbol's "DNA".
        """
        for pack in self._packs:
            concepts = list(pack.concepts())
            vectors = self._embed([concept.definition for concept in concepts])
            for concept, vector in zip(concepts, vectors, strict=True):
                await self._symbols.write_concept(concept, vector)
            for predicate in pack.predicates():
                await self._symbols.write_predicate(predicate)

    async def _write_mentions(self, mentions: Sequence[Mention]) -> None:
        """Write each mention's entity and the evidence edge that cites its surface.

        The entity carries the embedding of its surface — the ANN candidate
        generator's key (FR-020) — and one count under the label it was read
        with, which the writer folds into the stored histogram.
        """
        vectors = self._embed([mention.surface for mention in mentions])
        for mention, vector in zip(mentions, vectors, strict=True):
            await self._entities.write(
                EntityWrite(
                    id=mention.entity_id,
                    name=mention.surface,
                    name_norm=name_norm(mention.surface),
                    embedding=vector,
                    type_histogram={mention.label: 1} if mention.label else {},
                )
            )
            await self._entities.write_mention(mention)

    async def _label(self, segment: Segment, mentions: Sequence[Mention]) -> None:
        """Label entities and their segment with the concepts the mentions name.

        A segment evokes a concept once, at the confidence of the strongest
        mention that reached it — the edge is a symbol's activation, not a
        tally of the surfaces behind it.
        """
        evoked: dict[str, float] = {}
        for mention in mentions:
            concept = self._concepts.get(mention.label.casefold())
            if concept is None:
                continue
            await self._symbols.write_instance_of(mention.entity_id, concept, mention.confidence)
            evoked[concept.uri] = max(evoked.get(concept.uri, 0.0), mention.confidence)
        for uri, confidence in evoked.items():
            await self._symbols.write_evokes(segment.id, uri, confidence)

    async def _write_facts(self, segment: Segment, facts: Sequence[Fact]) -> None:
        """Write each fact against the segment that evidences it (FR-008)."""
        evidence = [Evidence(segment_id=segment.id, span=(0, len(segment.text.encode())))]
        for fact in facts:
            await self._facts.write(fact, evidence)
            await self._settle_currency(fact, segment.observed_at)

    async def _settle_currency(self, fact: Fact, anchor: datetime | None) -> None:
        """Rule the newly written *fact* against the current one it competes with.

        Only functional predicates compete (FR-012): of two objects the newer
        one stays current and SUPERSEDES the older, which is retained and marked
        not current. Two assertions anchored at the same instant — or at no
        instant at all — cannot be ordered, so neither retires the other and
        they are linked by CONTRADICTS instead.

        The anchor is the fact's own ``valid_from`` when it has one, else the
        observation time of the segment that evidences it.
        """
        if fact.predicate not in self._functional:
            return
        incumbent = await self._facts.incumbent(fact)
        if incumbent is None:
            return
        incumbent_id, incumbent_at = incumbent
        asserted_at = fact.validity.valid_from or anchor
        if asserted_at is None or incumbent_at is None or asserted_at == incumbent_at:
            await self._facts.contradict(fact.id, incumbent_id)
        elif asserted_at > incumbent_at:
            await self._facts.supersede(fact.id, incumbent_id)
        else:
            await self._facts.supersede(incumbent_id, fact.id)


__all__ = ["IngestPipeline"]
