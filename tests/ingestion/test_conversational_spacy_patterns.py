"""Unit tests for conversational first-person spacy patterns (J, K, L).

These tests validate the pattern-vocabulary constants and the triple-builder
helper without requiring a live spaCy model or heavy ML dependencies.
The spaCy-model-dependent functions (_pattern_J, _pattern_K, _pattern_L) are
imported and their presence verified; runtime execution is skipped when spaCy
is not installed.
"""

from __future__ import annotations

import importlib
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_spacy_module():
    """Load graphknows.extraction.relations.spacy without triggering the
    graphknows.extraction package __init__ (which eagerly imports heavy ML deps)."""
    import importlib.util
    from pathlib import Path

    mod_path = (
        Path(__file__).parent.parent.parent
        / "graphknows"
        / "extraction"
        / "relations"
        / "spacy.py"
    )
    # Pre-load graphknows.extraction.utils via the same trick to avoid __init__
    utils_path = (
        Path(__file__).parent.parent.parent
        / "graphknows"
        / "extraction"
        / "utils.py"
    )
    utils_spec = importlib.util.spec_from_file_location(
        "graphknows.extraction.utils", str(utils_path)
    )
    utils_mod = importlib.util.module_from_spec(utils_spec)  # type: ignore[arg-type]
    sys.modules["graphknows.extraction.utils"] = utils_mod
    utils_spec.loader.exec_module(utils_mod)  # type: ignore[union-attr]

    spec = importlib.util.spec_from_file_location(
        "graphknows.extraction.relations.spacy", str(mod_path)
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def spacy_mod():
    return _load_spacy_module()


# ---------------------------------------------------------------------------
# Vocabulary constant tests (no spaCy model required)
# ---------------------------------------------------------------------------


class TestPreferenceVocabulary:
    def test_like_verbs_present(self, spacy_mod):
        verbs = spacy_mod._LIKE_VERBS
        assert "love" in verbs
        assert "like" in verbs
        assert "enjoy" in verbs
        assert "prefer" in verbs

    def test_dislike_verbs_present(self, spacy_mod):
        verbs = spacy_mod._DISLIKE_VERBS
        assert "hate" in verbs
        assert "dislike" in verbs

    def test_like_and_dislike_disjoint(self, spacy_mod):
        overlap = spacy_mod._LIKE_VERBS & spacy_mod._DISLIKE_VERBS
        assert not overlap, f"Like and dislike verb sets overlap: {overlap}"


class TestPossessionVocabulary:
    def test_have_verbs_present(self, spacy_mod):
        verbs = spacy_mod._HAVE_VERBS
        assert "have" in verbs
        assert "own" in verbs
        assert "possess" in verbs


class TestRelationshipVocabulary:
    def test_kinship_nouns_present(self, spacy_mod):
        nouns = spacy_mod._KINSHIP_NOUNS
        for word in ("mother", "father", "sister", "brother", "wife", "husband"):
            assert word in nouns, f"Expected kinship noun {word!r} missing"

    def test_friend_nouns_present(self, spacy_mod):
        nouns = spacy_mod._FRIEND_NOUNS
        assert "friend" in nouns

    def test_partner_nouns_present(self, spacy_mod):
        nouns = spacy_mod._PARTNER_NOUNS
        for word in ("boyfriend", "girlfriend", "partner"):
            assert word in nouns, f"Expected partner noun {word!r} missing"

    def test_kinship_friend_partner_disjoint(self, spacy_mod):
        k = spacy_mod._KINSHIP_NOUNS
        f = spacy_mod._FRIEND_NOUNS
        p = spacy_mod._PARTNER_NOUNS
        assert not (k & f), f"Kinship and friend sets overlap: {k & f}"
        # wife/husband appear only in kinship, not partner
        assert "wife" not in p
        assert "husband" not in p


# ---------------------------------------------------------------------------
# Pattern function presence tests
# ---------------------------------------------------------------------------


class TestPatternFunctionsExist:
    def test_pattern_J_exists(self, spacy_mod):
        assert callable(spacy_mod._pattern_J)

    def test_pattern_K_exists(self, spacy_mod):
        assert callable(spacy_mod._pattern_K)

    def test_pattern_L_exists(self, spacy_mod):
        assert callable(spacy_mod._pattern_L)

    def test_all_patterns_registered(self, spacy_mod):
        """Verify J/K/L are in the extractor's dispatch loop (inspect source)."""
        import inspect
        src = inspect.getsource(spacy_mod.extract_intentional_relations)
        assert "_pattern_J" in src
        assert "_pattern_K" in src
        assert "_pattern_L" in src


# ---------------------------------------------------------------------------
# Runtime execution tests (require spaCy model; skipped when unavailable)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nlp(spacy_mod):
    """Return a spaCy pipeline or skip the test."""
    spacy = pytest.importorskip("spacy", reason="spaCy not installed")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("spaCy model en_core_web_sm not installed")
    return nlp


def _run_pattern(spacy_mod, nlp, pattern_fn, text: str) -> list[dict]:
    doc = nlp(text)
    spacy_ent_map = spacy_mod._build_spacy_ent_map(doc)
    results = []
    for sent in doc.sents:
        results.extend(pattern_fn(sent, doc, [], spacy_ent_map))
    return results


class TestPatternJPreference:
    def test_i_love_x(self, spacy_mod, nlp):
        triples = _run_pattern(spacy_mod, nlp, spacy_mod._pattern_J, "I love Italian food.")
        assert any(t["relation"] == "likes" for t in triples), f"Got: {triples}"

    def test_i_hate_x(self, spacy_mod, nlp):
        triples = _run_pattern(spacy_mod, nlp, spacy_mod._pattern_J, "I hate loud music.")
        assert any(t["relation"] == "dislikes" for t in triples), f"Got: {triples}"

    def test_i_like_x(self, spacy_mod, nlp):
        triples = _run_pattern(spacy_mod, nlp, spacy_mod._pattern_J, "I like hiking.")
        assert any(t["relation"] == "likes" for t in triples), f"Got: {triples}"

    def test_negated_like_becomes_dislike(self, spacy_mod, nlp):
        triples = _run_pattern(
            spacy_mod, nlp, spacy_mod._pattern_J, "I don't like spicy food."
        )
        assert any(t["relation"] == "dislikes" for t in triples), f"Got: {triples}"

    def test_source_is_i(self, spacy_mod, nlp):
        triples = _run_pattern(spacy_mod, nlp, spacy_mod._pattern_J, "I love coffee.")
        for t in triples:
            if t["relation"] == "likes":
                assert t["source"] == "I"
                assert t["source_label"] == "PERSON"
                break


class TestPatternKPossession:
    def test_my_noun(self, spacy_mod, nlp):
        triples = _run_pattern(spacy_mod, nlp, spacy_mod._pattern_K, "My car broke down.")
        assert any(t["relation"] == "owns" for t in triples), f"Got: {triples}"

    def test_i_have_x(self, spacy_mod, nlp):
        triples = _run_pattern(spacy_mod, nlp, spacy_mod._pattern_K, "I have a dog.")
        assert any(t["relation"] == "owns" for t in triples), f"Got: {triples}"

    def test_kinship_noun_not_in_K(self, spacy_mod, nlp):
        # "my sister" should NOT emit an "owns" triple; that is pattern L's job
        triples = _run_pattern(spacy_mod, nlp, spacy_mod._pattern_K, "My sister is great.")
        owns_triples = [t for t in triples if t["relation"] == "owns"]
        assert not owns_triples, f"Pattern K should not emit 'owns' for kinship: {owns_triples}"


class TestPatternLRelationship:
    def test_my_sister(self, spacy_mod, nlp):
        triples = _run_pattern(spacy_mod, nlp, spacy_mod._pattern_L, "My sister lives nearby.")
        assert any(t["relation"] == "family_of" for t in triples), f"Got: {triples}"

    def test_my_friend(self, spacy_mod, nlp):
        triples = _run_pattern(spacy_mod, nlp, spacy_mod._pattern_L, "My friend Tom is funny.")
        assert any(t["relation"] == "friend_of" for t in triples), f"Got: {triples}"

    def test_source_is_i(self, spacy_mod, nlp):
        triples = _run_pattern(spacy_mod, nlp, spacy_mod._pattern_L, "My brother works in Paris.")
        for t in triples:
            if t["relation"] == "family_of":
                assert t["source"] == "I"
                assert t["source_label"] == "PERSON"
                break
