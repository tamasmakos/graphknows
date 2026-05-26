"""Tests for domain.schema canonical constants and label conventions."""

import re
from pathlib import Path

import pytest


def test_canonical_labels_are_uppercase():
    """Ensure all canonical labels follow UPPERCASE convention."""
    from graphknows.domain.schema import (
        LABEL_CHUNK,
        LABEL_DIALOG,
        LABEL_ENTITY,
        LABEL_FILE,
        LABEL_SESSION,
        LABEL_TOPIC,
        LABEL_TURN,
    )

    labels = [
        LABEL_SESSION,
        LABEL_FILE,
        LABEL_CHUNK,
        LABEL_ENTITY,
        LABEL_DIALOG,
        LABEL_TURN,
        LABEL_TOPIC,
    ]

    for label in labels:
        assert label.isupper(), f"Label {label} must be uppercase"
        assert label == label.upper()


def test_canonical_relationships_are_uppercase():
    """Ensure all canonical relationship types follow UPPERCASE convention."""
    from graphknows.domain.schema import (
        REL_BELONGS_TO,
        REL_CO_OCCURS,
        REL_HAS_CHUNK,
        REL_HAS_DIALOG,
        REL_HAS_ENTITY,
        REL_HAS_FILE,
        REL_HAS_TURN,
        REL_RELATES_TO,
    )

    rels = [
        REL_HAS_FILE,
        REL_HAS_CHUNK,
        REL_HAS_ENTITY,
        REL_HAS_DIALOG,
        REL_HAS_TURN,
        REL_BELONGS_TO,
        REL_CO_OCCURS,
        REL_RELATES_TO,
    ]

    for rel in rels:
        assert rel.isupper(), f"Relationship {rel} must be uppercase"
        assert rel == rel.upper()


@pytest.mark.slow
def test_no_mixed_case_labels_in_codebase():
    """Scan codebase for forbidden mixed-case label usage.

    This test enforces the UPPERCASE-only convention for node labels.
    If this test fails, it means a Cypher query is using the wrong casing.

    Common violations:
      :Entity (should be :ENTITY)
      :Topic  (should be :TOPIC)
      :Chunk  (should be :CHUNK)
    """
    # Forbidden patterns (PascalCase labels)
    forbidden_patterns = [
        r'\([\w]*:Entity\b',
        r'\([\w]*:Topic\b',
        r'\([\w]*:Chunk\b',
        r'\([\w]*:Dialog\b',
        r'\([\w]*:Session\b',
        r'\([\w]*:File\b',
        r'\([\w]*:Turn\b',
    ]

    violations = []
    base = Path(__file__).parent.parent.parent / 'graphknows'

    for py_file in base.rglob('*.py'):
        # Skip test files, migrations, and legacy patches
        if 'test_' in py_file.name or '/migrations/' in str(py_file) or '/patch/phases/legacy.py' in str(py_file):
            continue

        try:
            with open(py_file) as f:
                for line_no, line in enumerate(f, 1):
                    # Skip comments and docstrings
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                        continue

                    for pattern in forbidden_patterns:
                        if re.search(pattern, line):
                            violations.append(
                                f"{py_file.relative_to(base.parent)}:{line_no} - {line.strip()[:80]}"
                            )
        except Exception:
            # Skip files that can't be read
            continue

    if violations:
        msg = (
            f"Found {len(violations)} mixed-case label violations.\n\n"
            "All Cypher labels must be UPPERCASE to match schema constraints.\n\n"
            "First 10 violations:\n" + "\n".join(violations[:10])
        )
        pytest.fail(msg)


def test_schema_ddl_uses_canonical_labels():
    """Verify that schema.py constraint definitions use the canonical constants."""
    from graphknows.schema import CONSTRAINTS, INDEXES, MEMGRAPH_INDEXES

    # All DDL should contain only uppercase labels
    all_ddl = "\n".join(CONSTRAINTS + INDEXES + MEMGRAPH_INDEXES)

    # Should contain uppercase labels
    assert ':SESSION' in all_ddl
    assert ':ENTITY' in all_ddl
    assert ':TOPIC' in all_ddl

    # Should NOT contain mixed-case labels
    assert ':Entity' not in all_ddl, "schema.py still has :Entity (should be :ENTITY)"
    assert ':Topic' not in all_ddl, "schema.py still has :Topic (should be :TOPIC)"
    assert ':Chunk' not in all_ddl, "schema.py still has :Chunk (should be :CHUNK)"
