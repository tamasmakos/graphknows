"""Lexical label selection, the overlapping window, and per-turn deixis.

Three behaviours that are silent when broken, which is why each is pinned:

* the SKOS scheme reaches a concept through an authored altLabel, so the
  overlay's vocabulary is actually live rather than merely loaded;
* the window resolves deixis PER TURN, because a window has many speakers and
  one text — leaving that unresolved is what attributed Gina's job to Jon;
* the attribution guard refuses a subject the asserting turn does not support.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphknows.ingestion.stm._attribution import admits, turn_spans
from graphknows.ingestion.stm._deixis import resolve_line, resolve_window
from graphknows.ontology import entity_labels, relation_labels
from graphknows.ontology.scheme import load_scheme
from graphknows.settings import GraphKnowsSettings


@pytest.fixture(scope="module")
def scheme() -> dict:
    settings = GraphKnowsSettings()
    return load_scheme(settings.ontology_source, settings.overlay_source)


class TestOverlay:
    def test_the_overlay_is_loaded_over_cco(self, scheme: dict) -> None:
        """CCO has no binary predicate for employment; the overlay supplies one."""
        assert "works for" in scheme
        assert scheme["works for"].module == "overlay"
        assert scheme["works for"].domain == ("Person",)

    def test_authored_alt_labels_are_lemmatised(self, scheme: dict) -> None:
        """Matching is on spaCy lemmas, so "lost" must also be reachable as "lose".

        Authored verbatim, ``lost`` matched nothing — the text lemmatises to
        ``lose`` and the surface form is never what is compared. It looked like a
        model failure and was a string-normalisation bug.
        """
        alts = scheme["formerly worked for"].alt_labels
        assert "lost" in alts
        assert "lose" in alts

    def test_the_overlay_links_into_ccos_hierarchy(self, scheme: dict) -> None:
        """``broader`` into CCO is what makes overlay concepts generalise."""
        assert "Facility" in scheme["Dance Studio"].broader


class TestLexicalSelection:
    """Selection needs an index; these build a tiny one rather than a corpus."""

    @pytest.fixture
    def index(self, scheme: dict) -> dict:
        from graphknows.ontology import lexical_index

        return lexical_index(scheme)

    def test_a_spoken_relation_is_offered(self, scheme: dict, index: dict) -> None:
        """The whole point: the words in the turn choose the relation vocabulary."""
        import spacy

        nlp = spacy.load(GraphKnowsSettings().spacy_model)
        doc = nlp("Jon: I lost my job at Door Dash and I want to open a dance studio.")
        offered = relation_labels(doc, {"Person", "Organization"}, scheme, index)

        assert "formerly worked for" in offered or "works for" in offered
        assert len(offered) <= 10, "the budget is load-bearing — relex degrades as labels grow"

    def test_coarse_types_are_always_offered(self, scheme: dict, index: dict) -> None:
        """They are measured-best AND are CCO classes verbatim, so they cost nothing."""
        import spacy

        nlp = spacy.load(GraphKnowsSettings().spacy_model)
        labels = entity_labels(nlp("Nothing relevant here at all."), scheme, index)
        assert {"person", "organization", "facility"} <= set(labels)


class TestDeixis:
    def test_first_person_resolves_to_the_speaker(self) -> None:
        assert resolve_line("I lost my job", "Gina", "Jon") == "Gina lost Gina's job"

    def test_second_person_resolves_to_the_addressee(self) -> None:
        assert resolve_line("How is your studio?", "Gina", "Jon") == "How is Jon's studio?"

    def test_each_turn_resolves_to_its_own_speaker(self) -> None:
        """The bug this exists for: one window, several speakers, one text."""
        window = [
            {"speaker": "Jon", "text": "Lost my job as a banker."},
            {"speaker": "Gina", "text": "I also lost my job at Door Dash."},
        ]
        out = resolve_window(window)
        assert "Jon's job as a banker" in out
        assert "Gina also lost Gina's job at Door Dash" in out

    def test_three_way_conversation_leaves_second_person_alone(self) -> None:
        """With more than two participants the addressee is a guess, so don't."""
        window = [
            {"speaker": "A", "text": "your turn"},
            {"speaker": "B", "text": "ok"},
            {"speaker": "C", "text": "mine"},
        ]
        assert "your turn" in resolve_window(window)


class TestAttributionGuard:
    @pytest.fixture
    def window(self) -> tuple[str, list, set]:
        turns = [
            {"speaker": "Jon", "resolved": "Lost Jon's job as a banker yesterday."},
            {
                "speaker": "Gina",
                "resolved": "Sorry about Jon's job Jon, but Gina also lost Gina's job at Door Dash.",
            },
        ]
        text = "\n".join(f"{t['speaker']}: {t['resolved']}" for t in turns)
        return text, turn_spans(turns, text), {"Jon", "Gina"}

    def test_the_asserting_speaker_is_admitted(self, window: tuple) -> None:
        text, spans, speakers = window
        assert admits("Gina", "Door Dash", text, spans, speakers)

    def test_a_bystander_named_in_the_same_turn_is_refused(self, window: tuple) -> None:
        """Presence is not evidence — deixis writes the addressee into the line."""
        text, spans, speakers = window
        assert not admits("Jon", "Door Dash", text, spans, speakers)

    def test_a_cross_turn_relation_still_passes(self, window: tuple) -> None:
        """The window exists for these; the guard must not eat them."""
        text, spans, speakers = window
        assert admits("Jon", "a thing nobody named", text, spans, speakers)


def test_the_bundled_overlay_ships() -> None:
    assert Path(GraphKnowsSettings().overlay_source).is_file()


class TestContractions:
    """An apostrophe is a word boundary, which is how "I'm" became "Jon'm".

    Not cosmetic: "Jon'm" is a second PERSON node, so the speaker is split in two
    and half their relations hang off the wrong one. Measured on a smoke ingest,
    all seven of Jon's relations attached to "Jon'm".
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("I'm starting a studio", "Gina is starting a studio"),
            ("I've been dancing", "Gina has been dancing"),
            ("I'll open a store", "Gina will open a store"),
            ("I'd like that", "Gina would like that"),
            ("We're opening today", "Gina is opening today"),
        ],
    )
    def test_first_person_contractions_expand(self, text: str, expected: str) -> None:
        assert resolve_line(text, "Gina", "Jon") == expected

    def test_second_person_contractions_expand(self) -> None:
        assert resolve_line("You're the best", "Gina", "Jon") == "Jon is the best"

    def test_no_apostrophe_fragment_survives(self) -> None:
        """The failure mode itself: never emit a name glued to a contraction tail."""
        for text in ("I'm here", "I've gone", "we're set", "you'll see"):
            assert "'m" not in resolve_line(text, "Gina", "Jon")
            assert "'ve" not in resolve_line(text, "Gina", "Jon")
