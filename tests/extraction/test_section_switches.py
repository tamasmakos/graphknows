"""FR-016: a section switched off is not asked for and produces nothing.

Scenario 3.3 is the shape all three cases are built to: the relation section off
must leave entities and frame instances standing and produce no relations at all
— and in particular no GLiNER-produced substitute, which on this decoder is
structurally impossible rather than merely unused (contracts/decoder.md §1).

The stub provider answers with all three sections whatever it was asked for, so a
section that comes back empty came back empty because the switch trimmed it, not
because the reply was short. Each case also reads what actually went out: an
ablation whose section is still in the prompt measures nothing.
"""

from __future__ import annotations

import json
from typing import Any

import dspy

from graphknows.ingestion.extraction.llm.decoder import LLMDecoder
from graphknows.ingestion.extraction.llm.schema import FrameCandidate
from graphknows.settings import GraphKnowsSettings

TEXT = "Melanie works for Acme."
ENTITY_LABELS = ("person", "organization")
RELATION_SPEC = {"worksFor": "the subject is employed by the object"}
EMPLOYED = FrameCandidate(
    frame="Being_employed",
    trigger="works",
    trigger_offset=8,
    core_elements=(("Employee", "the one employed"), ("Employer", "the one employing")),
)

FULL_REPLY = json.dumps(
    {
        "entities": [
            {"surface": "Melanie", "label": "person"},
            {"surface": "Acme", "label": "organization"},
        ],
        "relations": [
            {
                "head": "Melanie",
                "predicate": "worksFor",
                "tail": "Acme",
                "evidence": "Melanie works for Acme",
                "confidence": 0.9,
            }
        ],
        "frames": [
            {
                "frame": "Being_employed",
                "trigger": "works",
                "roles": {"Employee": ["Melanie"], "Employer": ["Acme"]},
            }
        ],
    }
)


class _StubProvider(dspy.LM):  # type: ignore[misc]
    """Over-answers every request — all three sections — and records what it got."""

    def __init__(self) -> None:
        super().__init__(model="openrouter/stub/sections-v1", api_key="stub", cache=False)
        self.requests: list[Any] = []

    def __call__(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> list[str]:
        self.requests.append(messages if messages is not None else prompt)
        return [FULL_REPLY]

    @property
    def sent(self) -> str:
        """Everything the provider was told, as one blob to search."""
        return json.dumps(self.requests)


def _decode(provider: _StubProvider, **switches: bool) -> tuple[LLMDecoder, Any]:
    """One chunk offered all three sections, decoded under *switches*."""
    decoder = LLMDecoder(GraphKnowsSettings(**switches), lm=provider)
    result = decoder.extract(TEXT, ENTITY_LABELS, RELATION_SPEC, frame_candidates=(EMPLOYED,))
    return decoder, result


def test_relations_off_keeps_entities_and_frames_and_yields_no_relation() -> None:
    provider = _StubProvider()

    decoder, result = _decode(provider, GRAPHKNOWS_DECODE_RELATIONS=False)

    assert sorted(entity["name"] for entity in result.entities) == ["Acme", "Melanie"]
    assert [frame["frame"] for frame in result.frames] == ["Being_employed"]
    assert result.relations == []
    # Nothing local filled the hole: the two attributes a consumer probes for a
    # local relation pass do not exist on this decoder, so a GLiNER relation is
    # unreachable rather than merely absent.
    assert not hasattr(decoder, "gliner")
    assert not hasattr(decoder, "relation_verifier")
    assert "relation_spec" not in provider.sent
    assert "worksFor" not in provider.sent


def test_frames_off_keeps_entities_and_relations_and_yields_no_frame() -> None:
    provider = _StubProvider()

    _, result = _decode(provider, GRAPHKNOWS_DECODE_FRAMES=False)

    assert sorted(entity["name"] for entity in result.entities) == ["Acme", "Melanie"]
    assert [relation["relation"] for relation in result.relations] == ["worksFor"]
    assert result.frames == []
    assert "frame_candidates" not in provider.sent
    assert "Being_employed" not in provider.sent


def test_entities_off_leaves_relations_and_roles_nothing_to_anchor_on() -> None:
    provider = _StubProvider()

    _, result = _decode(provider, GRAPHKNOWS_DECODE_ENTITIES=False)

    assert result.entities == []
    # A relation is built from entity surfaces and a role filler is anchored as
    # one, so the entity switch takes both with it — and the chunk costs nothing.
    assert result.relations == []
    assert result.frames == []
    assert "entity_labels" not in provider.sent
    assert provider.requests == []
