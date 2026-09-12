"""The decode request shape, and the strict response validated item by item.

A returned item that violates the schema is that item's problem
(``malformed_item``); only a payload that is not a container at all is the
chunk's problem (contracts/decoder.md §2-§3, spec §Edge Cases "well-formed JSON
but violates the schema").
"""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from graphknows.ingestion.extraction.llm.schema import (
    DecodedEntity,
    DecodedFrameInstance,
    DecodedItems,
    DecodedRelation,
    DecodeRequest,
    DecodeResponse,
    FrameCandidate,
)

ENTITY = {"surface": "Melanie", "label": "PERSON"}
RELATION = {
    "head": "Melanie",
    "predicate": "worksFor",
    "tail": "Acme",
    "evidence": "Melanie works for Acme",
    "confidence": 0.9,
}
FRAME = {"frame": "Being_employed", "trigger": "works", "roles": {"Employee": ["Melanie"]}}


def test_request_carries_the_closed_vocabularies_and_is_frozen() -> None:
    candidate = FrameCandidate(
        frame="Being_employed",
        trigger="works",
        trigger_offset=8,
        core_elements=(("Employee", "the one employed"),),
    )
    request = DecodeRequest(
        text="Melanie works for Acme",
        entity_labels=("PERSON", "ORG"),
        relation_spec={"worksFor": "the subject is employed by the object"},
        frame_candidates=(candidate,),
    )

    assert request.speaker == ""  # unresolved speaker is the default, not an error
    assert [f.name for f in dataclasses.fields(request)] == [
        "text",
        "entity_labels",
        "relation_spec",
        "frame_candidates",
        "speaker",
        "prompt_addendum",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.text = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.frame = "other"  # type: ignore[misc]


def test_item_models_accept_the_contract_shape() -> None:
    assert DecodedEntity(**ENTITY).label == "PERSON"
    assert DecodedRelation(**RELATION).confidence == 0.9
    assert DecodedFrameInstance(**FRAME).roles == {"Employee": ["Melanie"]}


@pytest.mark.parametrize(
    ("model", "item"),
    [
        (DecodedEntity, ENTITY),
        (DecodedRelation, RELATION),
        (DecodedFrameInstance, FRAME),
    ],
)
def test_unknown_key_is_forbidden_on_every_item_model(model, item) -> None:
    with pytest.raises(ValidationError):
        model(**item, note="invented by the model")


@pytest.mark.parametrize("confidence", [None, "high", -0.1, 1.1])
def test_confidence_is_required_and_bounded(confidence) -> None:
    with pytest.raises(ValidationError):
        DecodedRelation(**{**RELATION, "confidence": confidence})
    with pytest.raises(ValidationError):
        DecodedRelation(**{k: v for k, v in RELATION.items() if k != "confidence"})


def test_a_malformed_item_leaves_its_section_intact() -> None:
    response = DecodeResponse.model_validate(
        {
            "entities": [ENTITY, {"surface": "Acme"}, {**ENTITY, "surface": "Acme", "gloss": "x"}],
            "relations": [{**RELATION, "confidence": "very"}, RELATION],
            "frames": [FRAME],
        }
    )

    decoded = response.decoded()

    assert [e.surface for e in decoded.entities] == ["Melanie"]
    assert [r.predicate for r in decoded.relations] == ["worksFor"]
    assert [f.frame for f in decoded.frames] == ["Being_employed"]
    assert decoded.malformed_item == 3  # missing field, unknown key, malformed confidence


def test_absent_sections_decode_to_nothing_rather_than_failing() -> None:
    assert DecodeResponse().decoded() == DecodedItems()


def test_only_an_unparseable_container_is_a_decoder_failure() -> None:
    with pytest.raises(ValidationError):
        DecodeResponse.model_validate({"entities": "Melanie is a PERSON"})
