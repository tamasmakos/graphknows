"""Sense-anchored WordNet expansion replaces first-sense lookup (issue #141).

``skos.lexicalize`` stamps a concept's altLabels with the lemmas of its label's
FIRST WordNet synset, at scheme-BUILD time — before any mention exists to
disambiguate against. ``lexical.evoked`` then fires a one-token concept
whenever any of those synonyms appears in a chunk, in whatever sense the
speaker meant. The fix moves expansion to match time: a present lemma may only
license expansion through a synset chosen by embedding the trigger-starred
mention line against the candidate synsets' definitions, above a similarity
threshold AND a top1-top2 margin, abstaining otherwise. These three tests pin
the seam (``skos.lexicalize`` + ``skos.lexical_index`` + ``labels.entity_labels``
/ ``labels.relation_labels``) without touching any private helper, so the file
compiles and runs both before and after the fix.
"""

from __future__ import annotations

import pytest
import spacy

from graphknows.settings import GraphKnowsSettings
from graphknows.symbolic.ontology import entity_labels, relation_labels
from graphknows.symbolic.ontology.scheme import load_scheme
from graphknows.symbolic.ontology.skos import Concept, lexical_index, lexicalize


@pytest.fixture(scope="module")
def nlp():
    return spacy.load(GraphKnowsSettings().spacy_model)


@pytest.fixture(scope="module")
def production_scheme() -> dict:
    settings = GraphKnowsSettings()
    return load_scheme(settings.ontology_source, settings.overlay_source)


def test_wrong_sense_does_not_license_expansion(nlp) -> None:
    """RED TODAY: context-free expansion fires on the wrong sense of "let".

    "Allows" is deliberately NOT BFO-typed, so the existing upper-ontology
    blocklist in ``lexicalize`` cannot mask the mechanism under test. Today
    ``lexicalize`` gives "Allows" the altLabel "let" (allow -> permit.v.01 ->
    {permit, allow, let, countenance}), so the property is offered for "Gina
    let her flat to a student" — the LEASING sense of "let", not the permission
    sense. Nobody asserted a permission relation in that sentence; the property
    fires purely because the word appears, in any sense at all. This test must
    fail until expansion is anchored to the mention's actual sense.

    No ``domain``/``range`` is declared, on purpose: ``relation_labels``'
    structural half only offers a property whose declared domain AND range are
    both satisfied by ``entity_types``, so a domain/range pair matching
    ``{"person", "product"}`` would let "Allows" through structurally with zero
    lexical evidence, passing (or failing) regardless of anything this test
    means to pin. Leaving both empty routes "Allows" through the LEXICAL half
    only — exactly the mechanism under test.
    """
    concept = lexicalize(
        Concept(
            uri="ex:Allows",
            pref_label="Allows",
            kind="property",
            definition="grants someone permission to use a thing",
        )
    )
    scheme = {concept.pref_label: concept}
    index = lexical_index(scheme)
    doc = nlp("Gina let her flat to a student for the summer.")

    assert "Allows" not in relation_labels(doc, {"person", "product"}, scheme, index)


def test_unambiguous_sense_still_expands(nlp) -> None:
    """GREEN BEFORE AND AFTER: a correct in-context sense still expands.

    Guards against a fix that simply deletes expansion rather than
    context-anchoring it. "infirmary" has exactly one noun synset
    (hospital.n.01, lemmas {hospital, infirmary}), so the mention is
    unambiguous and expansion must still be licensed both before and after
    the fix.
    """
    concept = lexicalize(
        Concept(
            uri="ex:Hospital",
            pref_label="Hospital",
            kind="class",
            definition="a health facility where patients receive treatment",
        )
    )
    scheme = {concept.pref_label: concept}
    index = lexical_index(scheme)
    doc = nlp("She was rushed to the infirmary last night.")

    assert "Hospital" in entity_labels(doc, scheme, index)


def test_the_word_let_mints_no_upper_ontology_plumbing(nlp, production_scheme: dict) -> None:
    """THE TICKET PIN: "let" mints no upper-ontology plumbing on the real scheme.

    conv-30 minted "Gina permits fashion" from the word "let" meaning
    "inform" — 64 of 286 REL edges (22%) came from this class of failure. On
    the bundled production scheme, a casual "I'll let you know" must not
    offer any of the upper-ontology properties this expansion mechanism was
    firing spuriously.
    """
    index = lexical_index(production_scheme)
    doc = nlp("Gina: Sure, I'll let you know when I hear back.")

    offered = set(relation_labels(doc, {"person", "organization"}, production_scheme, index))
    assert {"permits", "requires", "realizes"}.isdisjoint(offered)
