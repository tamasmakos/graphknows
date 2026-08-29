"""Nominalisation and light-verb triggers on the non-SRL path (issue #144).

Today ``lu_candidate_frames`` builds its one candidate set from the clause main
verb(s) only, consulting noun/adjective LUs solely in an ``if not cand:``
fallback that never runs once a verb fires -- so "the destruction of the
bridge ... happened" only ever surfaces happen.v's frames, never Destroying.
A light verb ("make", "take", ...) makes things worse: it contributes its own
irrelevant LU frames instead of being read as a detector pointing at its noun
object ("made a decision" -> Deciding, not Building/Cooking_creation).

``extract_frame_instances`` then collapses everything to at most one instance
per chunk, so even if the noun's frame were found, it would still be dropped
whenever a verb's instance also passed the gate.

These tests pin the fixed contract: ``lu_triggers`` returns one trigger per
verb group, one per nominalisation noun, one per light-verb noun object
(flagged ``lvc=True``); ``extract_frame_instances`` takes such triggers and
lets each contribute its own instance.
"""

from __future__ import annotations

import pytest

from graphknows.ingestion.extraction.relations.frame_srl import (
    FrameTrigger,
    extract_frame_instances,
    lu_triggers,
)

pytestmark = pytest.mark.unit


class _Tok:
    """Duck-typed spaCy token: .text, .lemma_, .pos_, .dep_, .children."""

    def __init__(
        self,
        text: str,
        lemma: str,
        pos: str,
        dep: str,
        children: list[_Tok] | None = None,
    ) -> None:
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.dep_ = dep
        self.children = children or []


def _doc_a() -> list[_Tok]:
    """ "The destruction of the bridge by the storm happened last year."""
    the1 = _Tok("The", "the", "DET", "det")
    bridge = _Tok("bridge", "bridge", "NOUN", "pobj")
    of = _Tok("of", "of", "ADP", "prep", children=[bridge])
    storm = _Tok("storm", "storm", "NOUN", "pobj")
    by = _Tok("by", "by", "ADP", "prep", children=[storm])
    destruction = _Tok("destruction", "destruction", "NOUN", "nsubj", children=[the1, of, by])
    happened = _Tok("happened", "happen", "VERB", "ROOT", children=[destruction])
    return [the1, destruction, of, bridge, by, storm, happened]


def _doc_b() -> list[_Tok]:
    """ "Melanie made a decision about Berlin."""
    melanie = _Tok("Melanie", "Melanie", "PROPN", "nsubj")
    a = _Tok("a", "a", "DET", "det")
    berlin = _Tok("Berlin", "Berlin", "PROPN", "pobj")
    about = _Tok("about", "about", "ADP", "prep", children=[berlin])
    decision = _Tok("decision", "decision", "NOUN", "dobj", children=[a, about])
    made = _Tok("made", "make", "VERB", "ROOT", children=[melanie, decision])
    return [melanie, made, a, decision, about, berlin]


@pytest.fixture(autouse=True)
def _no_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every candidate scores 0.0 -- no embedder is ever invoked. Tests assert on
    # candidate-name SETS, never on sim ordering.
    monkeypatch.setattr(
        "graphknows.symbolic.framenet.index.match_frames_text",
        lambda text, top_k=200, min_sim=0.0: [],
    )


def _names(trigger: FrameTrigger) -> set[str]:
    return {name for name, _sim in trigger.candidates}


def test_a_nominalisation_triggers_even_though_a_verb_fired() -> None:
    triggers = lu_triggers(
        "The destruction of the bridge by the storm happened last year.", None, doc=_doc_a()
    )

    all_names = [_names(t) for t in triggers]
    assert any("Destroying" in names for names in all_names)
    assert any("Event" in names for names in all_names)


def test_a_light_verb_triggers_on_its_noun_object_and_is_flagged() -> None:
    triggers = lu_triggers("Melanie made a decision about Berlin.", None, doc=_doc_b())

    lvc_triggers = [t for t in triggers if t.lvc]
    assert len(lvc_triggers) == 1
    assert _names(lvc_triggers[0]) == {"Deciding"}

    union_names: set[str] = set()
    for t in triggers:
        union_names |= _names(t)
    assert "Building" not in union_names
    assert "Cooking_creation" not in union_names


class _Gliner:
    """Fills two core roles, confidently -- enough to clear the fillability gate."""

    def inference(self, **kwargs: object) -> list[list[dict[str, object]]]:
        labels = kwargs["labels"]
        assert isinstance(labels, list)
        return [
            [
                {"text": "the storm", "label": str(labels[0]), "score": 0.9},
                {"text": "the bridge", "label": str(labels[1]), "score": 0.8},
            ]
        ]


def test_each_trigger_becomes_its_own_frame_instance() -> None:
    text = "The destruction of the bridge by the storm happened last year."
    doc_a = _doc_a()

    instances = extract_frame_instances(
        text,
        "Melanie",
        ["Melanie"],
        _Gliner(),
        triggers=lambda t: lu_triggers(t, None, doc=doc_a),
    )

    assert len(instances) >= 2
    assert "Destroying" in {i.frame for i in instances}


def test_the_light_verb_instance_carries_the_lvc_flag() -> None:
    text = "Melanie made a decision about Berlin."
    doc_b = _doc_b()

    instances = extract_frame_instances(
        text,
        "Melanie",
        ["Melanie"],
        _Gliner(),
        triggers=lambda t: lu_triggers(t, None, doc=doc_b),
    )

    deciding = [i for i in instances if i.frame == "Deciding"]
    assert len(deciding) == 1
    assert deciding[0].lvc is True
    assert all(i.lvc is False for i in instances if i.frame != "Deciding")
