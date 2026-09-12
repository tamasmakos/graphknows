"""Extraction guidance comes from the pack, and the core keeps none of its own (FR-022)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.extraction.llm import decoder as llm_decoder
from graphknows.ingestion.extraction.llm.schema import DecodeRequest
from graphknows.ingestion.extraction.protocol import LLMExtractor, LocalExtractor, PackGuidance
from graphknows.models import PredicateRef
from graphknows.models.segment import Segment, SegmentKind


@dataclass(frozen=True)
class _GuidingPack:
    """A pack that guides every knob an extractor asks it for."""

    name: str = "demo"
    addendum: str = "Decisions are what this domain cares about."
    rejected: frozenset[str] = frozenset({"Analytical"})
    floors: Mapping[str, float] = field(default_factory=lambda: {"mention": 0.5, "fact": 0.6})

    def concepts(self) -> Iterable[Any]:
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
        return self.addendum

    def hygiene(self) -> Callable[[str], bool]:
        return lambda surface: surface not in self.rejected

    def thresholds(self) -> Mapping[str, float]:
        return self.floors

    def veto(self, a: str, b: str) -> bool:
        return False


class _StubDecoder:
    """A decoder answering with a fixed result, recording how it was asked."""

    def __init__(self, result: ExtractionResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def extract(
        self,
        text: str,
        extra_entity_labels: Any = None,
        extra_relation_labels: Any = None,
        speaker: str = "",
        prompt_addendum: str = "",
    ) -> ExtractionResult:
        self.calls.append(
            {
                "text": text,
                "labels": tuple(extra_entity_labels or ()),
                "prompt_addendum": prompt_addendum,
            }
        )
        return self.result


def _segment(text: str = "Ada works at Analytical.") -> Segment:
    return Segment(source_id="src", text=text, kind=SegmentKind.prose, byte_range=(0, 24))


def _result(entity_score: float = 0.9, relation_score: float = 0.7) -> ExtractionResult:
    return ExtractionResult(
        entities=[
            {"name": "Ada", "score": entity_score},
            {"name": "Analytical", "score": 0.8},
        ],
        relations=[
            {"head": "Ada", "relation": "works at", "tail": "Analytical", "score": relation_score}
        ],
    )


def test_the_guiding_pack_is_guidance_the_core_can_read() -> None:
    assert isinstance(_GuidingPack(), PackGuidance)


def test_the_pack_label_set_replaces_and_the_core_offers_none() -> None:
    decoder = _StubDecoder(ExtractionResult())

    LocalExtractor(decoder).extract(_segment(), _GuidingPack())

    assert decoder.calls[0]["labels"] == ("person", "organisation")
    # The label set the LLM decoder puts on offer is the pack's, deduplicated —
    # no core inventory is unioned in behind the pack's back.
    assert llm_decoder._entity_labels(["person", "person", "organisation"]) == (
        "person",
        "organisation",
    )
    assert llm_decoder._entity_labels(None) == ()


def test_the_prompt_addendum_reaches_the_assisted_path_only() -> None:
    llm, local = _StubDecoder(ExtractionResult()), _StubDecoder(ExtractionResult())
    pack = _GuidingPack()

    LLMExtractor(llm).extract(_segment(), pack)
    LocalExtractor(local).extract(_segment(), pack)

    assert llm.calls[0]["prompt_addendum"] == pack.addendum
    assert local.calls[0]["prompt_addendum"] == ""


def test_the_addendum_is_appended_to_the_decoder_instructions() -> None:
    request = DecodeRequest(
        text="Ada works at Analytical.",
        entity_labels=("person",),
        relation_spec={},
        frame_candidates=(),
        prompt_addendum="Decisions are what this domain cares about.",
    )

    instructions = llm_decoder._signature_for(request).instructions

    # Appended, never substituted: the core's own instructions still stand.
    assert instructions.endswith("\n\nDecisions are what this domain cares about.")
    assert instructions.startswith("Label one chunk against the closed vocabularies")


def test_pack_hygiene_drops_a_surface_and_the_facts_that_cite_it() -> None:
    mentions, facts = LocalExtractor(_StubDecoder(_result())).extract(_segment(), _GuidingPack())

    assert [mention.surface for mention in mentions] == ["Ada"]
    assert facts == []


def test_pack_thresholds_floor_mentions_and_facts() -> None:
    pack = _GuidingPack(rejected=frozenset())
    below_mention = _StubDecoder(_result(entity_score=0.4))
    below_fact = _StubDecoder(_result(relation_score=0.5))

    mentions, _ = LocalExtractor(below_mention).extract(_segment(), pack)
    kept, facts = LocalExtractor(below_fact).extract(_segment(), pack)

    assert [mention.surface for mention in mentions] == ["Analytical"]
    assert [mention.surface for mention in kept] == ["Ada", "Analytical"]
    assert facts == []
