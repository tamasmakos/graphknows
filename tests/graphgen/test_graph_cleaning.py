"""Tests for graph cleaning: canonicalization helpers."""

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


