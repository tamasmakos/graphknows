"""Guard tests for the taxonomy-driven relation schema (schema-guided extraction).

Locks the step-2 decisions: the deterministic relation model extracts only the
high-precision content predicates (the noisy interpersonal and stative families
are excluded from extraction while remaining in the canonical vocabulary).
"""

from __future__ import annotations

from graphknows.ingestion.models.taxonomy import (
    TaxonomyConfig,
    build_relation_schema,
    get_default_taxonomy,
)

_EXCLUDED_FROM_EXTRACTION = {
    "friend_of",
    "family_of",
    "partner_of",  # interpersonal: co-presence confusion
    "lives_in",
    "works_as",
    "member_of",  # stative: wrong object-type over-fire
}


def test_default_taxonomy_has_relation_descriptions():
    t = get_default_taxonomy()
    assert t.relation_descriptions, "conversational taxonomy must define relation_descriptions"


def test_schema_extracts_only_content_predicates():
    schema = build_relation_schema(get_default_taxonomy())
    for pred in _EXCLUDED_FROM_EXTRACTION:
        assert pred not in schema, f"{pred!r} must be excluded from the extraction schema"
    # The high-precision content predicates ARE extracted.
    for pred in ("attended", "happened_on", "owns", "likes"):
        assert pred in schema and schema[pred].strip()


def test_excluded_predicates_stay_in_canonical_vocabulary():
    """Dropped-from-extraction predicates remain available to LLM-assisted paths."""
    t = get_default_taxonomy()
    for pred in _EXCLUDED_FROM_EXTRACTION:
        assert pred in t.relation_types


def test_build_relation_schema_falls_back_without_descriptions():
    """A taxonomy with no relation_descriptions uses every relation_type."""
    t = TaxonomyConfig(name="x", relation_types=["likes", "owns"])
    schema = build_relation_schema(t)
    assert set(schema) == {"likes", "owns"}


def test_extractor_defaults_to_open_vocabulary():
    """The live extractor has NO taxonomy relation schema — mined labels only.

    An explicitly passed schema (escape hatch) is still honored.
    """
    from unittest.mock import MagicMock

    from graphknows.ingestion.extraction.entities.extractor import GLiNER2EntityExtractor

    extractor = GLiNER2EntityExtractor(schema_miner=MagicMock())
    assert extractor._relation_schema == {}

    explicit = build_relation_schema(get_default_taxonomy())
    extractor = GLiNER2EntityExtractor(schema_miner=MagicMock(), relation_schema=explicit)
    assert extractor._relation_schema == explicit
