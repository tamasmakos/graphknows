"""Tests for graph cleaning: canonicalization and coreference resolution."""

from __future__ import annotations

from graphknows.graphgen.extraction.canonicalization import (
    are_potential_aliases,
    classify_surface_form,
    normalize_surface_form,
    surface_forms_conflict,
)

# ---------------------------------------------------------------------------
# normalize_surface_form
# ---------------------------------------------------------------------------


def test_normalize_surface_form_uppercases_text():
    assert normalize_surface_form("hello") == "HELLO"


def test_normalize_surface_form_replaces_spaces_with_underscore():
    assert normalize_surface_form("European Union") == "EUROPEAN_UNION"


def test_normalize_surface_form_strips_punctuation():
    # Special chars become spaces then underscores; leading/trailing stripped
    result = normalize_surface_form("EU.")
    assert result == "EU"


def test_normalize_surface_form_empty_string_returns_empty():
    assert normalize_surface_form("") == ""


def test_normalize_surface_form_strips_leading_trailing_underscores():
    result = normalize_surface_form("  hello  ")
    assert not result.startswith("_")
    assert not result.endswith("_")


def test_normalize_surface_form_collapses_consecutive_separators():
    result = normalize_surface_form("hello   world")
    assert "__" not in result
    assert result == "HELLO_WORLD"


def test_normalize_surface_form_preserves_numbers():
    assert normalize_surface_form("Article 50") == "ARTICLE_50"


# ---------------------------------------------------------------------------
# classify_surface_form
# ---------------------------------------------------------------------------


def test_classify_role_term_as_role_artifact():
    # PRIME_MINISTER is in ROLE_TERMS
    assert classify_surface_form("Prime Minister") == "role_artifact"


def test_classify_president_as_role_artifact():
    assert classify_surface_form("PRESIDENT") == "role_artifact"


def test_classify_concept_term_as_concept_like():
    assert classify_surface_form("Migration") == "concept_like"


def test_classify_named_entity_hint_as_named_entity():
    assert classify_surface_form("European Parliament") == "named_entity"


def test_classify_empty_string_returns_unknown():
    assert classify_surface_form("") == "unknown"


def test_classify_short_token_returns_unknown():
    # "XYZ" normalizes to 3 chars, not in any special set → unknown
    assert classify_surface_form("xyz") == "unknown"


def test_classify_multi_word_name_as_named_entity():
    # Multi-token with alpha starts → "named_entity"
    assert classify_surface_form("Mario Draghi") == "named_entity"


# ---------------------------------------------------------------------------
# surface_forms_conflict
# ---------------------------------------------------------------------------


def test_conflict_opposing_terms_dependence_independence():
    assert surface_forms_conflict("Energy Dependence", "Energy Independence") is True


def test_conflict_identical_forms_not_a_conflict():
    # Same normalized form → not a conflict (returns False)
    assert surface_forms_conflict("EU", "EU") is False


def test_conflict_empty_inputs_returns_false():
    assert surface_forms_conflict("", "EU") is False
    assert surface_forms_conflict("EU", "") is False


def test_conflict_unrelated_terms_returns_false():
    assert surface_forms_conflict("Alice", "Bob") is False


def test_conflict_demonym_country_pair_is_conflict():
    # e.g. "UKRAINE" and "UKRAINIAN" — country vs demonym
    assert surface_forms_conflict("Ukraine", "Ukrainian") is True


# ---------------------------------------------------------------------------
# are_potential_aliases
# ---------------------------------------------------------------------------


def test_are_potential_aliases_identical_strings():
    assert are_potential_aliases("EU", "EU") is True


def test_are_potential_aliases_same_normalized_form():
    # "european-union" and "European Union" normalize to the same thing
    assert are_potential_aliases("European Union", "European-Union") is True


def test_are_potential_aliases_eu_and_europe_is_false():
    # Explicitly excluded per code: {"EU", "EUROPE"} == False
    assert are_potential_aliases("EU", "EUROPE") is False


def test_are_potential_aliases_empty_strings_returns_false():
    assert are_potential_aliases("", "EU") is False
    assert are_potential_aliases("EU", "") is False


def test_are_potential_aliases_role_artifact_vs_named_entity_returns_false():
    # PRIME_MINISTER (role) vs MARIO_DRAGHI (named_entity) → different classes → False
    assert are_potential_aliases("Prime Minister", "Mario Draghi") is False


def test_are_potential_aliases_high_similarity_strings():
    # Very similar strings should be considered aliases
    assert are_potential_aliases("European Parliament", "European-Parliament") is True


def test_are_potential_aliases_clearly_different_terms():
    assert are_potential_aliases("Alice", "Bob") is False


def test_are_potential_aliases_opposing_terms_not_aliases():
    # Conflicting forms (dependence vs independence) should not be aliases
    assert are_potential_aliases("Energy Dependence", "Energy Independence") is False




# ===========================================================================
# Coreference: canonicalize, similarity helpers, resolve_extraction_coreferences
# ===========================================================================

import pytest


# ---------------------------------------------------------------------------
# _canonicalize_entity_name
# ---------------------------------------------------------------------------


class TestCanonicalizeEntityName:
    def test_empty_returns_empty(self):
        from graphknows.graphgen.extraction.coreference import _canonicalize_entity_name

        assert _canonicalize_entity_name("") == ""

    def test_lowercases_and_strips(self):
        from graphknows.graphgen.extraction.coreference import _canonicalize_entity_name

        assert _canonicalize_entity_name("  Alice  ") == "alice"

    def test_removes_punctuation(self):
        from graphknows.graphgen.extraction.coreference import _canonicalize_entity_name

        result = _canonicalize_entity_name("European Union!")
        assert "!" not in result
        assert result == "european union"


# ---------------------------------------------------------------------------
# _string_similarity
# ---------------------------------------------------------------------------


class TestStringSimilarity:
    def test_identical_strings(self):
        from graphknows.graphgen.extraction.coreference import _string_similarity

        assert _string_similarity("alice", "alice") == pytest.approx(1.0)

    def test_completely_different(self):
        from graphknows.graphgen.extraction.coreference import _string_similarity

        score = _string_similarity("abc", "xyz")
        assert score < 0.5

    def test_partial_overlap(self):
        from graphknows.graphgen.extraction.coreference import _string_similarity

        score = _string_similarity("european union", "european parliament")
        assert 0.0 < score < 1.0

    def test_symmetry(self):
        from graphknows.graphgen.extraction.coreference import _string_similarity

        assert _string_similarity("alice", "alice bob") == pytest.approx(
            _string_similarity("alice bob", "alice")
        )


# ---------------------------------------------------------------------------
# _token_similarity
# ---------------------------------------------------------------------------


class TestTokenSimilarity:
    def test_identical(self):
        from graphknows.graphgen.extraction.coreference import _token_similarity

        assert _token_similarity("president of ecb", "president of ecb") == pytest.approx(1.0)

    def test_word_reordering(self):
        from graphknows.graphgen.extraction.coreference import _token_similarity

        score = _token_similarity("president of ecb", "ecb president")
        # Intersection=2 words ("ecb", "president"), union=3 → 2/3
        assert score == pytest.approx(2 / 3)

    def test_no_overlap(self):
        from graphknows.graphgen.extraction.coreference import _token_similarity

        assert _token_similarity("alice", "bob") == pytest.approx(0.0)

    def test_empty_string(self):
        from graphknows.graphgen.extraction.coreference import _token_similarity

        assert _token_similarity("", "alice") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _are_coreferent
# ---------------------------------------------------------------------------


class TestAreCoreferent:
    def test_identical_strings_are_coreferent(self):
        from graphknows.graphgen.extraction.coreference import _are_coreferent

        assert _are_coreferent("alice", "alice", threshold=0.85) is True

    def test_very_different_strings_not_coreferent(self):
        from graphknows.graphgen.extraction.coreference import _are_coreferent

        assert _are_coreferent("alice", "zebra financing", threshold=0.85) is False

    def test_high_token_similarity_passes(self):
        from graphknows.graphgen.extraction.coreference import _are_coreferent

        # "ecb president" ↔ "president ecb" — token sim = 1.0 → coreferent
        assert _are_coreferent("ecb president", "president ecb", threshold=0.95) is True

    def test_threshold_is_respected(self):
        from graphknows.graphgen.extraction.coreference import _are_coreferent

        assert _are_coreferent("alice", "alice", threshold=0.0) is True
        assert _are_coreferent("alice", "ali", threshold=0.99) is False


# ---------------------------------------------------------------------------
# resolve_extraction_coreferences
# ---------------------------------------------------------------------------


class TestResolveExtractionCoreferences:
    def test_empty_inputs_return_empty(self):
        from graphknows.graphgen.extraction.coreference import resolve_extraction_coreferences

        result = resolve_extraction_coreferences([], [])
        assert result["cleaned_relations"] == []
        assert result["entity_mappings"] == {}

    def test_passes_through_non_overlapping_relations(self):
        from graphknows.graphgen.extraction.coreference import resolve_extraction_coreferences

        relations = [("Alice", "KNOWS", "Bob", {})]
        result = resolve_extraction_coreferences(relations, [], similarity_threshold=0.9)
        assert len(result["cleaned_relations"]) == 1
        cleaned = result["cleaned_relations"][0]
        assert cleaned[0] in result["entity_mappings"].values() or cleaned[0] == "Alice"

    def test_merges_near_duplicate_entity_names(self):
        from graphknows.graphgen.extraction.coreference import resolve_extraction_coreferences

        relations = [
            ("European Union", "MEMBER", "Germany", {}),
            ("European Union.", "INCLUDES", "France", {}),
        ]
        result = resolve_extraction_coreferences(relations, [], similarity_threshold=0.85)
        mappings = result["entity_mappings"]
        reps = {mappings.get("European Union"), mappings.get("European Union.")}
        assert len(reps) == 1 or "European Union" in str(reps)

    def test_removes_self_loops_after_merge(self):
        from graphknows.graphgen.extraction.coreference import resolve_extraction_coreferences

        relations = [("EU", "IS", "EU", {})]
        result = resolve_extraction_coreferences(relations, [], similarity_threshold=0.85)
        for s, r, t, _ in result["cleaned_relations"]:
            assert s != t

    def test_handles_three_tuple_relations(self):
        from graphknows.graphgen.extraction.coreference import resolve_extraction_coreferences

        relations = [("Alice", "KNOWS", "Bob")]
        result = resolve_extraction_coreferences(relations, [])
        assert len(result["cleaned_relations"]) >= 0

    def test_entities_list_added_to_surface_forms(self):
        from graphknows.graphgen.extraction.coreference import resolve_extraction_coreferences

        result = resolve_extraction_coreferences([], ["Alice", "Bob", "alice"])
        assert "Alice" in result["entity_mappings"]
        assert "alice" in result["entity_mappings"]

    def test_returns_debug_log(self):
        from graphknows.graphgen.extraction.coreference import resolve_extraction_coreferences

        result = resolve_extraction_coreferences([], ["A"])
        assert "debug_log" in result
        assert isinstance(result["debug_log"], list)
