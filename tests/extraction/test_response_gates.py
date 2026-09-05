"""The four response gates, and the determinism the anchor law rests on.

Anchor, vocabulary, confidence and the local cleaning, run over a decoded
response with no provider anywhere near it (contracts/decoder.md §3,
data-model.md §4).
"""

from __future__ import annotations

from graphknows.ingestion.extraction.llm.anchor import ResponseGates
from graphknows.ingestion.extraction.llm.schema import (
    DecodedEntity,
    DecodedFrameInstance,
    DecodedItems,
    DecodedRelation,
    DecodeRequest,
    FrameCandidate,
)

TEXT = "Melanie works for Acme. Priya works for Acme too."

EMPLOYED = FrameCandidate(
    frame="Being_employed",
    trigger="works",
    trigger_offset=8,
    core_elements=(("Employee", "the one employed"), ("Employer", "the one employing")),
)

REQUEST = DecodeRequest(
    text=TEXT,
    entity_labels=("person", "organization"),
    relation_spec={"worksFor": "the subject is employed by the object"},
    frame_candidates=(EMPLOYED,),
)


def _gate(items: DecodedItems, confidence_min: float = 0.5, request: DecodeRequest = REQUEST):
    return ResponseGates(request, confidence_min).apply(items)


def _entities(*pairs: tuple[str, str]) -> tuple[DecodedEntity, ...]:
    return tuple(DecodedEntity(surface=s, label=label) for s, label in pairs)


def _relation(**overrides) -> DecodedRelation:
    return DecodedRelation(
        **{
            "head": "Melanie",
            "predicate": "worksFor",
            "tail": "Acme",
            "evidence": "Melanie works for Acme",
            "confidence": 0.9,
            **overrides,
        }
    )


# --- Gate 1: anchoring -------------------------------------------------------


def test_a_surface_absent_from_the_chunk_is_dropped_and_counted() -> None:
    gated = _gate(DecodedItems(entities=_entities(("Globex", "organization"))))

    assert gated.result.entities == []
    assert gated.abstentions["unanchorable_surface"] == 1


def test_an_unanchorable_role_filler_is_dropped_and_counted() -> None:
    instance = DecodedFrameInstance(
        frame="Being_employed",
        trigger="works",
        roles={"Employee": ["Melanie"], "Employer": ["Globex"]},
    )

    gated = _gate(DecodedItems(frames=(instance,)))

    assert gated.result.frames[0]["roles"] == {"Employee": ["Melanie"]}
    assert gated.abstentions["unanchorable_surface"] == 1


def test_a_surface_is_never_approximately_matched() -> None:
    # "Melany" is one character off a real span; an approximate anchor would
    # manufacture the grounding the anchor law forbids (SC-007).
    gated = _gate(DecodedItems(entities=_entities(("Melany", "person"))))

    assert gated.result.entities == []
    assert gated.abstentions["unanchorable_surface"] == 1


# --- Gate 2: closed vocabulary ----------------------------------------------


def test_an_entity_label_outside_the_offered_set_is_dropped_not_mapped() -> None:
    gated = _gate(DecodedItems(entities=_entities(("Melanie", "people"))))

    assert gated.result.entities == []
    assert gated.abstentions["out_of_vocabulary"] == 1


def test_a_label_matches_casefolded_exactly_and_keeps_the_offered_spelling() -> None:
    gated = _gate(DecodedItems(entities=_entities(("Melanie", "PERSON"))))

    assert [(e["name"], e["type"]) for e in gated.result.entities] == [("Melanie", "PERSON")]
    assert gated.abstentions["out_of_vocabulary"] == 0


def test_a_predicate_outside_the_relation_spec_is_dropped_and_counted() -> None:
    gated = _gate(DecodedItems(relations=(_relation(predicate="works_for"),)))

    assert gated.result.relations == []
    assert gated.abstentions["out_of_vocabulary"] == 1


def test_a_frame_the_local_pass_never_offered_is_dropped_and_counted() -> None:
    instance = DecodedFrameInstance(frame="Giving", trigger="works", roles={"Donor": ["Melanie"]})

    gated = _gate(DecodedItems(frames=(instance,)))

    assert gated.result.frames == []
    assert gated.abstentions["out_of_vocabulary"] == 1


def test_a_role_outside_the_candidates_core_elements_is_dropped_and_counted() -> None:
    instance = DecodedFrameInstance(
        frame="Being_employed",
        trigger="works",
        roles={"Employee": ["Melanie"], "Manner": ["Acme"]},
    )

    gated = _gate(DecodedItems(frames=(instance,)))

    assert gated.result.frames[0]["roles"] == {"Employee": ["Melanie"]}
    assert gated.abstentions["out_of_vocabulary"] == 1


# --- Gate 3: the confidence floor -------------------------------------------


def test_a_relation_below_the_floor_is_dropped_and_counted() -> None:
    gated = _gate(DecodedItems(relations=(_relation(confidence=0.3),)), confidence_min=0.5)

    assert gated.result.relations == []
    assert gated.abstentions["low_confidence_relation"] == 1


def test_a_zero_threshold_disables_the_gate() -> None:
    # FR-014: this is how the fourth ablation run is performed — no other switch.
    gated = _gate(DecodedItems(relations=(_relation(confidence=0.0),)), confidence_min=0.0)

    assert [r["relation"] for r in gated.result.relations] == ["worksFor"]
    assert gated.abstentions["low_confidence_relation"] == 0


def test_a_kept_relation_carries_the_confidence_the_verifier_score_carried() -> None:
    gated = _gate(DecodedItems(relations=(_relation(),)))

    (relation,) = gated.result.relations
    assert relation["source"] == "llm"
    assert relation["score"] == 0.9
    assert relation["verifier_score"] == 0.9
    assert relation["evidence"] == "Melanie works for Acme"


# --- Gate 4: the schema's own drops reach the same report --------------------


def test_malformed_items_are_carried_through_under_their_own_gate() -> None:
    gated = _gate(DecodedItems(malformed_item=2))

    assert gated.abstentions["malformed_item"] == 2
    assert gated.result.entities == []


# --- Determinism -------------------------------------------------------------


def test_a_repeated_surface_takes_the_next_unclaimed_occurrence() -> None:
    # "Acme" occurs twice; a third filler has nothing left to claim.
    instance = DecodedFrameInstance(
        frame="Being_employed",
        trigger="works",
        roles={"Employer": ["Acme", "Acme", "Acme"]},
    )

    gated = _gate(DecodedItems(frames=(instance,)))

    assert gated.result.frames[0]["roles"] == {"Employer": ["Acme", "Acme"]}
    assert gated.abstentions["unanchorable_surface"] == 1


def test_the_same_response_always_yields_the_same_result() -> None:
    items = DecodedItems(
        entities=_entities(
            ("Melanie", "person"), ("Acme", "organization"), ("Acme", "ORGANIZATION")
        ),
        relations=(_relation(), _relation(head="Priya")),
        frames=(
            DecodedFrameInstance(
                frame="Being_employed",
                trigger="works",
                roles={"Employee": ["Melanie", "Priya"], "Employer": ["Acme"]},
            ),
        ),
    )
    gates = ResponseGates(REQUEST, 0.5)

    first, second = gates.apply(items), gates.apply(items)

    assert first == second
    assert _gate(items) == first  # and a fresh instance agrees


def test_an_entity_and_a_role_filler_do_not_compete_for_one_span() -> None:
    items = DecodedItems(
        entities=_entities(("Melanie", "person")),
        frames=(
            DecodedFrameInstance(
                frame="Being_employed", trigger="works", roles={"Employee": ["Melanie"]}
            ),
        ),
    )

    gated = _gate(items)

    assert [e["name"] for e in gated.result.entities] == ["Melanie"]
    assert gated.result.frames[0]["roles"] == {"Employee": ["Melanie"]}
    assert gated.abstentions["unanchorable_surface"] == 0


# --- The local path's cleaning, so both modes mint the same merge keys -------


CLEAN_TEXT = "Melanie: I met a banker named Priya."
CLEAN_REQUEST = DecodeRequest(
    text=CLEAN_TEXT,
    entity_labels=("person", "occupation"),
    relation_spec={"worksFor": "the subject is employed by the object"},
    frame_candidates=(),
    speaker="Melanie",
)


def test_surfaces_are_cleaned_exactly_as_the_local_decoder_cleans_them() -> None:
    gated = _gate(
        DecodedItems(entities=_entities(("a banker", "occupation"), ("Melanie: I met", "person"))),
        request=CLEAN_REQUEST,
    )

    # _strip_determiner drops the article; _reduce_speaker_span folds a span that
    # ran across the turn-line colon back onto the speaker.
    assert sorted(e["name"] for e in gated.result.entities) == ["Melanie", "banker"]


def test_a_triple_the_relation_guard_rejects_never_reaches_the_result() -> None:
    items = DecodedItems(
        entities=_entities(("Priya", "person")),
        relations=(_relation(head="Priya", tail="Priya"),),
    )

    gated = _gate(items, request=CLEAN_REQUEST)

    # A self-relation is a structural reject, not one of the four response gates.
    assert gated.result.relations == []
    assert gated.abstentions == {}
