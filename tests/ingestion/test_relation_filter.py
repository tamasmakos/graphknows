"""Unit tests for the ontology-driven relation filter (post-extraction curation)."""

from __future__ import annotations

from graphknows.extraction.ontology.relation_filter import (
    RelationOntologyFilter,
    build_relation_filter,
)


def test_allows_ontology_label_case_and_punctuation_insensitive():
    f = RelationOntologyFilter(["memberOf", "author"])
    assert f.is_allowed("member_of")
    assert f.is_allowed("MEMBEROF")
    assert f.is_allowed("Author")


def test_drops_unknown_labels():
    f = RelationOntologyFilter(["author"])
    assert not f.is_allowed("EXEMPLIFIES")
    assert not f.is_allowed("AM")


def test_empty_allowlist_is_passthrough():
    f = RelationOntologyFilter([])
    assert not f.active
    assert f.is_allowed("anything")
    rels = [{"relation": "AM"}, {"relation": "PM"}]
    assert f.filter(rels, "relation") == rels


def test_filter_keeps_only_allowed():
    f = RelationOntologyFilter(["author", "memberOf"])
    rels = [
        {"head": "a", "relation": "author", "tail": "b"},
        {"head": "a", "relation": "EXEMPLIFIES", "tail": "b"},
        {"head": "a", "relation": "member of", "tail": "b"},
    ]
    kept = f.filter(rels, "relation")
    assert [r["relation"] for r in kept] == ["author", "member of"]


def test_build_default_filter_loads_schema_org():
    f = build_relation_filter()
    # schema.org defines these; junk verb labels are not present
    assert f.active
    assert f.is_allowed("author")
    assert not f.is_allowed("EXEMPLIFIES")
