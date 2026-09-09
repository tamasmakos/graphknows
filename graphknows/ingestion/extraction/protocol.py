"""The extraction seam: what an extractor is, and the two paths behind it.

An extractor reads one segment with one pack's vocabulary and answers in the
neutral currency of the graph — mentions and facts — so the pipeline never
learns which of the two paths produced them (FR-043).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from hashlib import sha256
from typing import Any, Protocol, runtime_checkable

from graphknows.models.fact import Fact, Mention
from graphknows.models.segment import Segment
from graphknows.models.symbols import ConceptRef, PredicateRef
from graphknows.symbolic.predicates import PredicateIndex


@runtime_checkable
class PackGuidance(Protocol):
    """The whole of what the write path asks a pack for.

    Declared here rather than imported: ``graphknows.packs`` sits above the core
    and the core must not know it (``core-knows-no-domain``). Every ``DomainPack``
    satisfies this structurally.
    """

    @property
    def name(self) -> str:
        """Pack name; the key its predicate index is memoised under."""

    def concepts(self) -> Iterable[ConceptRef]:
        """The concepts a segment or entity from this pack may be labelled with."""

    def predicates(self) -> Iterable[PredicateRef]:
        """The predicates a fact from this pack may name."""

    def entity_labels(self) -> tuple[str, ...]:
        """The label set offered to the decoder — it *replaces*, never unions."""

    def prompt_addendum(self) -> str:
        """Domain guidance appended to the decoder's instructions; "" for none."""

    def hygiene(self) -> Callable[[str], bool]:
        """Admits a surface as an entity name. The core keeps no default (FR-022)."""

    def thresholds(self) -> Mapping[str, float]:
        """Confidence floors by name: ``mention`` and ``fact``, absent meaning none."""


@runtime_checkable
class Extractor(Protocol):
    """Turns one segment into the mentions and facts it evidences.

    Attributes:
        name: Extractor identity, recorded on everything it emits.
        version: Extractor version; a change re-extracts a segment.
    """

    name: str
    version: str

    def extract(self, segment: Segment, pack: PackGuidance) -> tuple[list[Mention], list[Fact]]:
        """The mentions and facts *segment* evidences under *pack*."""
        ...


def _entity_id(name: str) -> str:
    """Identity of the entity a surface refers to.

    Normalised like the graph's own merge key (mirrors ``graph_store._name_norm``),
    so two surfaces that differ only in case or spacing name one entity.
    """
    return sha256(" ".join(name.split()).casefold().encode()).hexdigest()[:16]


def _byte_span(text: str, surface: str) -> tuple[int, int] | None:
    """Byte offsets of *surface* within *text*, or ``None`` when it is absent.

    A surface the segment does not contain cannot be cited, and evidence is what
    makes a mention worth keeping (FR-008).
    """
    found = text.find(surface)
    if found < 0:
        return None
    start = len(text[:found].encode())
    return start, start + len(surface.encode())


class _DecoderExtractor:
    """Puts a chunk decoder behind :class:`Extractor`.

    Both decoders answer with an ``ExtractionResult`` of entity and relation
    dicts; the translation into mentions and facts is the same for both, so it
    lives here and the two subclasses only choose a decoder and a name.
    """

    name = ""
    version = "1"

    def __init__(self, decoder: Any) -> None:
        self._decoder = decoder
        # A pack's predicate index is built once per pack, not once per
        # segment: `predicates()` is a lazy iterable by contract.
        self._indexes: dict[str, PredicateIndex] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._decoder!r})"

    def extract(self, segment: Segment, pack: PackGuidance) -> tuple[list[Mention], list[Fact]]:
        """Label *segment* under the pack's guidance: labels, hygiene and floors (FR-022)."""
        result = self._decode(segment, pack)
        admits, floors = pack.hygiene(), pack.thresholds()
        mentions = [
            mention
            for entity in result.entities
            if (mention := self._mention(entity, segment)) is not None
            and admits(mention.surface)
            and mention.confidence >= floors.get("mention", 0.0)
        ]
        cited = {mention.entity_id for mention in mentions}
        facts = [
            fact
            for relation in result.relations
            if (fact := self._fact(relation, segment, pack)) is not None
            and {fact.subject, fact.object} <= cited
            and fact.confidence >= floors.get("fact", 0.0)
        ]
        return mentions, facts

    def _decode(self, segment: Segment, pack: PackGuidance) -> Any:
        """*segment* decoded against the pack's label set, which replaces the core's.

        The pack's prompt addendum is not passed here: this is the local path's
        call shape, and a span model takes no instructions.
        """
        return self._decoder.extract(segment.text, pack.entity_labels(), speaker=segment.role)

    def _mention(self, entity: dict[str, Any], segment: Segment) -> Mention | None:
        """One entity dict as a mention, or ``None`` when it cannot be cited."""
        surface = str(entity.get("name") or "")
        span = _byte_span(segment.text, surface) if surface else None
        if span is None:
            return None
        return Mention(
            segment_id=segment.id,
            entity_id=_entity_id(surface),
            surface=surface,
            span=span,
            label=str(entity.get("type") or ""),
            confidence=float(entity.get("score", 1.0) or 0.0),
            extractor=self.name,
        )

    def _fact(self, relation: dict[str, Any], segment: Segment, pack: PackGuidance) -> Fact | None:
        """One relation dict as a fact, or ``None`` when the pack has no predicate for it."""
        predicate = self._predicate_id(pack, str(relation.get("relation") or ""))
        if predicate is None:
            return None
        subject, object_ = _entity_id(str(relation["head"])), _entity_id(str(relation["tail"]))
        return Fact(
            id=sha256(f"{segment.id}|{subject}|{predicate}|{object_}".encode()).hexdigest()[:16],
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=float(relation.get("score", 1.0) or 0.0),
            extractor=self.name,
            extractor_version=self.version,
        )

    def _predicate_id(self, pack: PackGuidance, label: str) -> str | None:
        """The pack's predicate id for *label*; a relation it does not name is dropped."""
        index = self._indexes.get(pack.name)
        if index is None:
            index = self._indexes[pack.name] = PredicateIndex(pack.predicates())
        predicate = index.resolve(label)
        return predicate.id if predicate is not None else None


class LocalExtractor(_DecoderExtractor):
    """The local path: GLiNER2 entities and their jointly extracted relations."""

    name = "local"

    def __init__(self, decoder: Any | None = None) -> None:
        from graphknows.ingestion.extraction.entities import local_decoder

        super().__init__(decoder if decoder is not None else local_decoder())


class LLMExtractor(_DecoderExtractor):
    """The assisted path: one gated structured provider call per segment."""

    name = "llm"

    def __init__(self, decoder: Any | None = None) -> None:
        if decoder is None:
            from graphknows.ingestion.extraction.llm.decoder import LLMDecoder

            decoder = LLMDecoder()
        super().__init__(decoder)

    def _decode(self, segment: Segment, pack: PackGuidance) -> Any:
        """As the local path, plus the pack's guidance on the decoder's instructions."""
        return self._decoder.extract(
            segment.text,
            pack.entity_labels(),
            speaker=segment.role,
            prompt_addendum=pack.prompt_addendum(),
        )


__all__ = ["Extractor", "LLMExtractor", "LocalExtractor", "PackGuidance"]
