"""The write path end to end: a source, its segments, and the symbols they label.

This is where the bottom-up half of the traversal happens (FR-002). Extraction
answers in mentions and facts; every mention carries the pack label the decoder
gave the surface, and that label is what resolves a concept — so an entity is
written with an ``INSTANCE_OF`` edge to the concept it instantiates, and its
segment with an ``EVOKES`` edge to every concept its mentions reached. Top-down
activation reads exactly those two edges back.
"""

from __future__ import annotations

from collections.abc import Sequence

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
    ) -> None:
        self._sources = SourceWriter(store)
        self._segments = SegmentWriter(store)
        self._entities = EntityWriter(store)
        self._facts = FactWriter(store)
        self._symbols = SymbolWriter(store)
        self._extractor = extractor
        self._packs = tuple(packs)
        self._concepts = _concepts_by_label(self._packs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._extractor.name!r}, {len(self._packs)} packs)"

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
        for segment in segments:
            await self._segments.write(segment)
            for pack in self._packs:
                mentions, facts = self._extractor.extract(segment, pack)
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

    async def _write_mentions(self, mentions: Sequence[Mention]) -> None:
        """Write each mention's entity and the evidence edge that cites its surface."""
        for mention in mentions:
            await self._entities.write(
                EntityWrite(
                    id=mention.entity_id,
                    name=mention.surface,
                    name_norm=name_norm(mention.surface),
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


__all__ = ["IngestPipeline"]
