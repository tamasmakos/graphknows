"""The extraction seam: both paths answer in mentions and facts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.extraction.protocol import (
    Extractor,
    LLMExtractor,
    LocalExtractor,
    PackGuidance,
)
from graphknows.models import ConceptRef, PredicateRef
from graphknows.models.segment import Segment, SegmentKind
from graphknows.packs import DomainPack


@dataclass(frozen=True)
class _StubPack:
    """A pack naming one predicate and one entity label."""

    name: str = "demo"

    def concepts(self) -> Iterable[ConceptRef]:
        return iter(())

    def predicates(self) -> Iterable[PredicateRef]:
        yield PredicateRef(
            id="demo:works_at",
            label="works at",
            definition="Employment of a person by an organisation.",
            canonical="works at",
            pack=self.name,
        )

    def entity_labels(self) -> tuple[str, ...]:
        return ("person", "organisation")

    def prompt_addendum(self) -> str:
        return ""

    def hygiene(self) -> Callable[[str], bool]:
        return lambda surface: True

    def thresholds(self) -> Mapping[str, float]:
        return {}

    def veto(self, a: str, b: str) -> bool:
        return False


class _StubDecoder:
    """A decoder answering with a fixed result, recording what it was asked."""

    def __init__(self, result: ExtractionResult) -> None:
        self.result = result
        self.calls: list[tuple[Any, ...]] = []

    def extract(
        self,
        text: str,
        extra_entity_labels: Any = None,
        extra_relation_labels: Any = None,
        speaker: str = "",
        prompt_addendum: str = "",
    ) -> ExtractionResult:
        self.calls.append((text, tuple(extra_entity_labels or ()), speaker))
        return self.result


def _segment(text: str = "Ada works at Analytical.") -> Segment:
    return Segment(source_id="src", text=text, kind=SegmentKind.prose, byte_range=(0, 24))


def _result() -> ExtractionResult:
    return ExtractionResult(
        entities=[
            {"name": "Ada", "type": "PERSON", "score": 0.9},
            {"name": "Analytical", "type": "ORG", "score": 0.8},
        ],
        relations=[{"head": "Ada", "relation": "works at", "tail": "Analytical", "score": 0.7}],
    )


def test_a_domain_pack_is_guidance_the_core_can_read() -> None:
    pack = _StubPack()
    assert isinstance(pack, DomainPack)
    assert isinstance(pack, PackGuidance)


def test_both_paths_satisfy_the_protocol() -> None:
    assert isinstance(LocalExtractor(_StubDecoder(ExtractionResult())), Extractor)
    assert isinstance(LLMExtractor(_StubDecoder(ExtractionResult())), Extractor)
    assert (
        LocalExtractor(_StubDecoder(ExtractionResult())).name
        != LLMExtractor(_StubDecoder(ExtractionResult())).name
    )


def test_a_segment_becomes_mentions_and_facts() -> None:
    segment, pack = _segment(), _StubPack()
    extractor = LocalExtractor(_StubDecoder(_result()))

    mentions, facts = extractor.extract(segment, pack)

    assert [m.surface for m in mentions] == ["Ada", "Analytical"]
    assert all(m.segment_id == segment.id and m.extractor == "local" for m in mentions)
    assert [segment.text[start:end] for start, end in (m.span for m in mentions)] == [
        "Ada",
        "Analytical",
    ]
    assert len(facts) == 1
    assert facts[0].predicate == "demo:works_at"
    assert (facts[0].subject, facts[0].object) == (mentions[0].entity_id, mentions[1].entity_id)
    assert facts[0].extractor_version == extractor.version


def test_the_pack_supplies_the_label_set() -> None:
    decoder = _StubDecoder(ExtractionResult())

    LLMExtractor(decoder).extract(_segment(), _StubPack())

    assert decoder.calls == [("Ada works at Analytical.", ("person", "organisation"), "")]


def test_a_predicate_the_pack_does_not_name_is_dropped() -> None:
    result = ExtractionResult(
        entities=[{"name": "Ada", "score": 0.9}, {"name": "Analytical", "score": 0.8}],
        relations=[{"head": "Ada", "relation": "founded", "tail": "Analytical", "score": 0.7}],
    )

    _, facts = LLMExtractor(_StubDecoder(result)).extract(_segment(), _StubPack())

    assert facts == []


def test_an_uncitable_surface_yields_neither_mention_nor_fact() -> None:
    result = ExtractionResult(
        entities=[{"name": "Ada", "score": 0.9}, {"name": "Babbage", "score": 0.8}],
        relations=[{"head": "Ada", "relation": "works at", "tail": "Babbage", "score": 0.7}],
    )

    mentions, facts = LocalExtractor(_StubDecoder(result)).extract(_segment(), _StubPack())

    assert [m.surface for m in mentions] == ["Ada"]
    assert facts == []
