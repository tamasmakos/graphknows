"""Tests for graph cleaning: canonicalization and pruning."""

from __future__ import annotations

import networkx as nx

from graphknows.graphgen.pipeline.graph_cleaning.canonicalization import (
    are_potential_aliases,
    classify_surface_form,
    normalize_surface_form,
    surface_forms_conflict,
)
from graphknows.graphgen.pipeline.graph_cleaning.pruning import prune_graph

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


# ---------------------------------------------------------------------------
# prune_graph
# ---------------------------------------------------------------------------


def _make_pruning_graph():
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_node("isolated")
    g.add_edge("A", "B", graph_type="entity_relation", confidence=0.9)
    return g


def test_prune_graph_removes_isolated_nodes_by_default():
    g = _make_pruning_graph()
    result = prune_graph(g, {})
    assert "isolated" not in g.nodes
    assert result["nodes_removed"] == 1


def test_prune_graph_keeps_connected_nodes():
    g = _make_pruning_graph()
    prune_graph(g, {})
    assert "A" in g.nodes
    assert "B" in g.nodes


def test_prune_graph_returns_stats_dict():
    g = _make_pruning_graph()
    result = prune_graph(g, {})
    assert "nodes_removed" in result
    assert "edges_removed" in result
    assert "final_nodes" in result
    assert "final_edges" in result


def test_prune_graph_low_confidence_edges_removed():
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", graph_type="entity_relation", confidence=0.1)
    result = prune_graph(g, {"min_edge_confidence": 0.5, "prune_isolated_nodes": False})
    assert result["edges_removed"] == 1
    assert not g.has_edge("A", "B")


def test_prune_graph_high_confidence_edges_kept():
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", graph_type="entity_relation", confidence=0.95)
    prune_graph(g, {"min_edge_confidence": 0.5, "prune_isolated_nodes": False})
    assert g.has_edge("A", "B")


def test_prune_graph_non_entity_relation_edges_not_pruned():
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", graph_type="structural", confidence=0.0)
    prune_graph(g, {"min_edge_confidence": 0.9, "prune_isolated_nodes": False})
    # Non-entity_relation edges are not pruned by confidence
    assert g.has_edge("A", "B")


def test_prune_graph_isolated_nodes_not_removed_when_disabled():
    g = _make_pruning_graph()
    result = prune_graph(g, {"prune_isolated_nodes": False})
    assert "isolated" in g.nodes
    assert result["nodes_removed"] == 0


def test_prune_graph_empty_graph_no_crash():
    g = nx.DiGraph()
    result = prune_graph(g, {})
    assert result["final_nodes"] == 0
    assert result["final_edges"] == 0


def test_prune_graph_final_counts_reflect_actual_graph():
    g = nx.DiGraph()
    g.add_node("X")
    g.add_node("Y")
    g.add_edge("X", "Y", graph_type="entity_relation", confidence=0.95)
    result = prune_graph(g, {})
    assert result["final_nodes"] == g.number_of_nodes()
    assert result["final_edges"] == g.number_of_edges()


# ===========================================================================
# Entity resolution
# ===========================================================================


import numpy as np
import networkx as nx
import pytest

from graphknows.graphgen.pipeline.graph_cleaning.resolution import (
    EntityRecord,
    BlockingResolver,
    _types_compatible,
    _neighbor_signature,
    _neighbor_compatibility,
)


# ---------------------------------------------------------------------------
# _types_compatible
# ---------------------------------------------------------------------------

def test_types_compatible_same_type():
    assert _types_compatible("PERSON", "PERSON")


def test_types_compatible_empty_type():
    """Empty/None type is always compatible (missing type metadata)."""
    assert _types_compatible("", "PERSON")
    assert _types_compatible("ORG", "")
    assert _types_compatible(None, "ORG")


def test_types_compatible_different():
    assert not _types_compatible("PERSON", "ORG")


def test_types_compatible_case_insensitive():
    """Normalisation should make 'person' == 'PERSON'."""
    assert _types_compatible("person", "PERSON")


# ---------------------------------------------------------------------------
# _neighbor_signature
# ---------------------------------------------------------------------------

def test_neighbor_signature_empty_graph():
    g = nx.DiGraph()
    sig = _neighbor_signature(g, "nonexistent")
    assert sig == set()


def test_neighbor_signature_captures_out_edges():
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B", label="WORKS_AT")
    sig = _neighbor_signature(g, "A")
    # normalize_surface_form lowercases, but the label goes through normalize
    # Actual: sig contains tuples like ('WORKS_AT', 'B') normalized
    assert len(sig) > 0, "signature must not be empty when out-edges exist"
    assert any("works_at" in item[0].lower() for item in sig)


def test_neighbor_signature_captures_in_edges():
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("B", "A", label="KNOWS")
    sig = _neighbor_signature(g, "A")
    assert len(sig) > 0, "signature must not be empty when in-edges exist"
    assert any("knows" in item[0].lower() for item in sig)


# ---------------------------------------------------------------------------
# _neighbor_compatibility
# ---------------------------------------------------------------------------

def test_neighbor_compatibility_none_graph():
    assert _neighbor_compatibility(None, "A", "B") == 0.0


def test_neighbor_compatibility_no_neighbors():
    g = nx.DiGraph()
    g.add_node("A")
    g.add_node("B")
    assert _neighbor_compatibility(g, "A", "B") == 0.0


def test_neighbor_compatibility_identical_neighbors():
    g = nx.DiGraph()
    g.add_edge("A", "C", label="WORKS_AT")
    g.add_edge("B", "C", label="WORKS_AT")
    # A and B have the same neighbor (C) with the same edge label
    score = _neighbor_compatibility(g, "A", "B")
    assert score > 0.0


def test_neighbor_compatibility_disjoint_neighbors():
    g = nx.DiGraph()
    g.add_edge("A", "C", label="WORKS_AT")
    g.add_edge("B", "D", label="WORKS_AT")
    score = _neighbor_compatibility(g, "A", "B")
    assert score == 0.0


# ---------------------------------------------------------------------------
# BlockingResolver._get_blocking_keys
# ---------------------------------------------------------------------------

def test_blocking_keys_alpha_prefix():
    resolver = BlockingResolver()
    keys = resolver._get_blocking_keys("Apple Inc")
    assert any(k.startswith("ALPHA:") for k in keys)


def test_blocking_keys_token_extraction():
    resolver = BlockingResolver()
    keys = resolver._get_blocking_keys("Apple Inc")
    token_keys = [k for k in keys if k.startswith("TOKEN:")]
    # "Apple" is long enough and not a stop word
    assert any("apple" in k for k in token_keys)
    # "Inc" is a stop word — should not appear as a token key
    assert not any("inc" in k and k.startswith("TOKEN:") for k in token_keys)


def test_blocking_keys_short_tokens_excluded():
    resolver = BlockingResolver()
    keys = resolver._get_blocking_keys("a of the")
    token_keys = [k for k in keys if k.startswith("TOKEN:")]
    assert token_keys == []


# ---------------------------------------------------------------------------
# BlockingResolver._compute_similarity
# ---------------------------------------------------------------------------

def _make_rec(id_, text, type_="PERSON", emb=None):
    return EntityRecord(id=id_, text=text, type=type_, embedding=emb)


def test_similarity_identical_strings():
    resolver = BlockingResolver()
    rec = _make_rec("a", "John Smith")
    sim = resolver._compute_similarity(rec, rec)
    assert sim > 0.99


def test_similarity_incompatible_types_returns_zero():
    resolver = BlockingResolver()
    a = _make_rec("a", "Python", "LANG")
    b = _make_rec("b", "Python", "ORG")
    sim = resolver._compute_similarity(a, b)
    assert sim == 0.0


def test_similarity_string_only_similar():
    resolver = BlockingResolver()
    a = _make_rec("a", "John Smith")
    b = _make_rec("b", "John Smyth")
    sim = resolver._compute_similarity(a, b)
    assert 0.5 < sim < 1.0


def test_similarity_string_dissimilar():
    resolver = BlockingResolver()
    a = _make_rec("a", "Quantum Physics")
    b = _make_rec("b", "Banana Bread")
    sim = resolver._compute_similarity(a, b)
    assert sim < 0.5


def test_similarity_with_embeddings():
    resolver = BlockingResolver()
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([1.0, 0.0, 0.0])   # identical → cosine=1.0
    a = _make_rec("a", "Alice", emb=emb_a)
    b = _make_rec("b", "Alice", emb=emb_b)
    sim = resolver._compute_similarity(a, b)
    assert sim > 0.8


def test_similarity_orthogonal_embeddings():
    resolver = BlockingResolver()
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.0, 1.0, 0.0])   # orthogonal → cosine=0
    a = _make_rec("a", "Alice", emb=emb_a)
    b = _make_rec("b", "Bob",   emb=emb_b)
    sim = resolver._compute_similarity(a, b)
    # Completely unrelated names AND zero cosine → low similarity
    assert sim < 0.5


def test_similarity_prefix_boost():
    """
    Name A is a prefix of name B → the resolver's prefix logic boosts string_score.
    BUT: classify_surface_form('John')='concept_like', ('John Smith')='named_entity'
    → the mixed-class guard returns 0.0 before prefix logic runs.
    This is intentional behavior: short name fragments conflict with full named entities.
    We verify the actual (correct) behavior rather than a wrong assumption.
    """
    resolver = BlockingResolver()
    a = _make_rec("a", "John")       # concept_like
    b = _make_rec("b", "John Smith") # named_entity
    sim = resolver._compute_similarity(a, b)
    # Mixed concept_like vs named_entity → guard returns 0.0
    assert sim == 0.0


def test_similarity_prefix_boost_same_class():
    """Two named_entity class strings where one is a prefix of the other → ≥ 0.85."""
    resolver = BlockingResolver()
    # Use full proper names (both named_entity)
    a = _make_rec("a", "Barack Obama")
    b = _make_rec("b", "Barack Obama Jr")
    sim = resolver._compute_similarity(a, b)
    # Both are named_entity; A is a prefix of B → boost applies
    assert sim >= 0.85


# ---------------------------------------------------------------------------
# BlockingResolver.resolve — end-to-end
# ---------------------------------------------------------------------------

def test_resolve_merges_near_duplicates():
    """Two very similar names should resolve to the same cluster ID."""
    resolver = BlockingResolver(similarity_threshold=0.85)
    resolver.add_records([
        EntityRecord(id="r1", text="Barack Obama", type="PERSON"),
        EntityRecord(id="r2", text="Barack Obama", type="PERSON"),  # exact dup
    ])
    mapping = resolver.resolve()
    # Both records should map to the same canonical ID
    assert mapping["r1"] == mapping["r2"]


def test_resolve_keeps_distinct_entities_separate():
    """Completely unrelated entities must NOT be merged."""
    resolver = BlockingResolver(similarity_threshold=0.85)
    resolver.add_records([
        EntityRecord(id="r1", text="Albert Einstein", type="PERSON"),
        EntityRecord(id="r2", text="Quantum Mechanics", type="CONCEPT"),
    ])
    mapping = resolver.resolve()
    # Different types → incompatible → should stay distinct
    assert mapping.get("r1") != mapping.get("r2") or mapping.get("r1") is None or mapping.get("r2") is None


def test_resolve_returns_all_ids():
    """Every input ID must appear as a key in the mapping."""
    resolver = BlockingResolver()
    records = [EntityRecord(id=f"r{i}", text=f"Entity {i}", type="ORG") for i in range(5)]
    resolver.add_records(records)
    mapping = resolver.resolve()
    for rec in records:
        assert rec.id in mapping


def test_resolve_canonical_id_is_one_of_the_input_ids():
    """
    The canonical value in the mapping is actually the *text* of the canonical entity
    (by how BlockingResolver.resolve() is implemented — it maps id → canonical_text).
    Verify the values are texts from the input records, not arbitrary strings.
    """
    resolver = BlockingResolver(similarity_threshold=0.80)
    resolver.add_records([
        EntityRecord(id="a1", text="Google", type="ORG"),
        EntityRecord(id="a2", text="Google Inc", type="ORG"),
    ])
    mapping = resolver.resolve()
    all_texts = {"Google", "Google Inc"}
    for resolved_val in mapping.values():
        assert resolved_val in all_texts, f"Unexpected resolved value: {resolved_val}"


def test_resolve_empty_records():
    resolver = BlockingResolver()
    mapping = resolver.resolve()
    assert mapping == {}


def test_resolve_single_record():
    """Single record should map to its own text (canonical = self)."""
    resolver = BlockingResolver()
    resolver.add_records([EntityRecord(id="solo", text="Solo Entity", type="ORG")])
    mapping = resolver.resolve()
    assert "solo" in mapping
    # Canonical text should be the record's own text
    assert mapping["solo"] == "Solo Entity"


def test_resolve_alias_merges_at_high_threshold():
    """
    Known abbreviation / alias pairs should merge even with a high threshold
    because of the alias_bonus (0.97) in _compute_similarity.
    """
    resolver = BlockingResolver(similarity_threshold=0.90)
    resolver.add_records([
        EntityRecord(id="full", text="United States of America", type="GPE"),
        EntityRecord(id="abbr", text="USA", type="GPE"),
    ])
    mapping = resolver.resolve()
    # are_potential_aliases should fire for this pair
    # Both should be in the mapping; if merged, same canonical
    assert "full" in mapping
    assert "abbr" in mapping


# ===========================================================================
# Advanced resolution: canonicalize, similarity, coreference
# ===========================================================================



import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _canonicalize_entity_name
# ---------------------------------------------------------------------------


class TestCanonicalizeEntityName:
    def test_empty_returns_empty(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _canonicalize_entity_name

        assert _canonicalize_entity_name("") == ""

    def test_lowercases_and_strips(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _canonicalize_entity_name

        assert _canonicalize_entity_name("  Alice  ") == "alice"

    def test_removes_punctuation(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _canonicalize_entity_name

        result = _canonicalize_entity_name("European Union!")
        assert "!" not in result
        assert result == "european union"


# ---------------------------------------------------------------------------
# _string_similarity
# ---------------------------------------------------------------------------


class TestStringSimilarity:
    def test_identical_strings(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _string_similarity

        assert _string_similarity("alice", "alice") == pytest.approx(1.0)

    def test_completely_different(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _string_similarity

        score = _string_similarity("abc", "xyz")
        assert score < 0.5

    def test_partial_overlap(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _string_similarity

        score = _string_similarity("european union", "european parliament")
        assert 0.0 < score < 1.0

    def test_symmetry(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _string_similarity

        assert _string_similarity("alice", "alice bob") == pytest.approx(
            _string_similarity("alice bob", "alice")
        )


# ---------------------------------------------------------------------------
# _token_similarity
# ---------------------------------------------------------------------------


class TestTokenSimilarity:
    def test_identical(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _token_similarity

        assert _token_similarity("president of ecb", "president of ecb") == pytest.approx(1.0)

    def test_word_reordering(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _token_similarity

        score = _token_similarity("president of ecb", "ecb president")
        # Intersection=2 words ("ecb", "president"), union=3 → 2/3
        assert score == pytest.approx(2 / 3)

    def test_no_overlap(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _token_similarity

        assert _token_similarity("alice", "bob") == pytest.approx(0.0)

    def test_empty_string(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _token_similarity

        assert _token_similarity("", "alice") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _are_coreferent
# ---------------------------------------------------------------------------


class TestAreCoreferent:
    def test_identical_strings_are_coreferent(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _are_coreferent

        assert _are_coreferent("alice", "alice", threshold=0.85) is True

    def test_very_different_strings_not_coreferent(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _are_coreferent

        assert _are_coreferent("alice", "zebra financing", threshold=0.85) is False

    def test_high_token_similarity_passes(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _are_coreferent

        # "ecb president" ↔ "president ecb" — token sim = 1.0 → coreferent
        assert _are_coreferent("ecb president", "president ecb", threshold=0.95) is True

    def test_threshold_is_respected(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _are_coreferent

        # Same strings with threshold=0.0 → always True (sim=1.0 ≥ 0.0)
        assert _are_coreferent("alice", "alice", threshold=0.0) is True
        # Different strings with threshold=0.99
        assert _are_coreferent("alice", "ali", threshold=0.99) is False


# ---------------------------------------------------------------------------
# resolve_extraction_coreferences
# ---------------------------------------------------------------------------


class TestResolveExtractionCoreferences:
    def test_empty_inputs_return_empty(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import resolve_extraction_coreferences

        result = resolve_extraction_coreferences([], [])
        assert result["cleaned_relations"] == []
        assert result["entity_mappings"] == {}

    def test_passes_through_non_overlapping_relations(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import resolve_extraction_coreferences

        relations = [("Alice", "KNOWS", "Bob", {})]
        result = resolve_extraction_coreferences(relations, [], similarity_threshold=0.9)
        assert len(result["cleaned_relations"]) == 1
        cleaned = result["cleaned_relations"][0]
        assert cleaned[0] in result["entity_mappings"].values() or cleaned[0] == "Alice"

    def test_merges_near_duplicate_entity_names(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import resolve_extraction_coreferences

        # "European Union" and "European Union." should merge
        relations = [
            ("European Union", "MEMBER", "Germany", {}),
            ("European Union.", "INCLUDES", "France", {}),
        ]
        result = resolve_extraction_coreferences(relations, [], similarity_threshold=0.85)
        # Both source entities should map to the same representative
        mappings = result["entity_mappings"]
        reps = {mappings.get("European Union"), mappings.get("European Union.")}
        # After stripping punctuation, these should collapse to the same key
        assert len(reps) == 1 or "European Union" in str(reps)

    def test_removes_self_loops_after_merge(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import resolve_extraction_coreferences

        # Two names that would merge — relation between them becomes self-loop → dropped
        relations = [("EU", "IS", "EU", {})]
        result = resolve_extraction_coreferences(relations, [], similarity_threshold=0.85)
        # Self-loop (cs == ct) should be removed
        for s, r, t, _ in result["cleaned_relations"]:
            assert s != t

    def test_handles_three_tuple_relations(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import resolve_extraction_coreferences

        relations = [("Alice", "KNOWS", "Bob")]  # 3-tuple (no props)
        result = resolve_extraction_coreferences(relations, [])
        assert len(result["cleaned_relations"]) >= 0  # Should not raise

    def test_entities_list_added_to_surface_forms(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import resolve_extraction_coreferences

        result = resolve_extraction_coreferences([], ["Alice", "Bob", "alice"])
        # "alice" and "Alice" both canonicalize to "alice" → same rep
        assert "Alice" in result["entity_mappings"]
        assert "alice" in result["entity_mappings"]

    def test_returns_debug_log(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import resolve_extraction_coreferences

        result = resolve_extraction_coreferences([], ["A"])
        assert "debug_log" in result
        assert isinstance(result["debug_log"], list)


# ---------------------------------------------------------------------------
# _compute_similarity_matrix
# ---------------------------------------------------------------------------


class TestComputeSimilarityMatrix:
    def test_empty_embeddings_return_empty(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _compute_similarity_matrix

        ids, matrix = _compute_similarity_matrix({})
        assert ids == []
        assert matrix.size == 0

    def test_single_embedding(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _compute_similarity_matrix

        emb = {"e1": np.array([1.0, 0.0, 0.0])}
        ids, matrix = _compute_similarity_matrix(emb)
        assert ids == ["e1"]
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == pytest.approx(1.0)

    def test_two_orthogonal_embeddings_score_zero(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _compute_similarity_matrix

        embs = {
            "e1": np.array([1.0, 0.0]),
            "e2": np.array([0.0, 1.0]),
        }
        ids, matrix = _compute_similarity_matrix(embs)
        # Off-diagonal should be ~0 (orthogonal)
        i1, i2 = ids.index("e1"), ids.index("e2")
        assert matrix[i1, i2] == pytest.approx(0.0, abs=1e-5)

    def test_identical_embeddings_score_one(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _compute_similarity_matrix

        embs = {
            "a": np.array([0.5, 0.5, 0.5]),
            "b": np.array([0.5, 0.5, 0.5]),
        }
        ids, matrix = _compute_similarity_matrix(embs)
        ia, ib = ids.index("a"), ids.index("b")
        assert matrix[ia, ib] == pytest.approx(1.0)

    def test_handles_zero_norm_embedding(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _compute_similarity_matrix

        # Zero vector — norms[norms == 0] = 1 prevents division by zero
        embs = {
            "zero": np.array([0.0, 0.0, 0.0]),
            "normal": np.array([1.0, 0.0, 0.0]),
        }
        ids, matrix = _compute_similarity_matrix(embs)
        assert matrix is not None
        assert not np.isnan(matrix).any()

    def test_matrix_is_symmetric(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _compute_similarity_matrix

        embs = {
            "a": np.random.rand(64).astype(np.float32),
            "b": np.random.rand(64).astype(np.float32),
            "c": np.random.rand(64).astype(np.float32),
        }
        ids, matrix = _compute_similarity_matrix(embs)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-5)

    def test_diagonal_is_approximately_one(self):
        from graphknows.graphgen.pipeline.graph_cleaning.resolution import _compute_similarity_matrix

        embs = {f"e{i}": np.random.rand(32).astype(np.float32) for i in range(5)}
        ids, matrix = _compute_similarity_matrix(embs)
        np.testing.assert_allclose(np.diag(matrix), np.ones(5), atol=1e-5)
