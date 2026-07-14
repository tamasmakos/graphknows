"""Unit tests for the shared ArcadeDB label sanitizer."""

import pytest

from graphknows.storage.arcadedb._labels import _safe_rel_label


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Basic normalization
        ("located_in", "LOCATED_IN"),
        ("located-in", "LOCATED_IN"),
        ("Located In", "LOCATED_IN"),
        # Strip invalid characters
        ("rel@type!", "RELTYPE"),
        ("foo.bar", "FOOBAR"),
        # Vertex-name collision → REL_ prefix
        ("ENTITY", "REL_ENTITY"),
        ("entity", "REL_ENTITY"),
        ("CHUNK", "REL_CHUNK"),
        ("TOPIC", "REL_TOPIC"),
        ("SESSION", "REL_SESSION"),
        # Leading digit → invalid
        ("3_things", ""),
        ("123", ""),
        # Empty after stripping → invalid
        ("@#!", ""),
        ("", ""),
        # Valid passthrough
        ("FOUNDED_BY", "FOUNDED_BY"),
        ("CO_OCCURS", "CO_OCCURS"),
        ("temporal", "REL_TEMPORAL"),
        ("frame", "REL_FRAME"),
        # SQL reserved keyword → REL_ prefix (open-vocab predicates: match, like…)
        ("MATCH", "REL_MATCH"),
        ("match", "REL_MATCH"),
        ("like", "REL_LIKE"),
        ("SET", "REL_SET"),
        # A compound label containing a reserved word is not itself reserved
        ("MATCH_WITH", "MATCH_WITH"),
        ("WORK_AS", "WORK_AS"),
    ],
)
def test_safe_rel_label(raw: str, expected: str) -> None:
    assert _safe_rel_label(raw) == expected
