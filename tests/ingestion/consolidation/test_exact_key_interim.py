"""Identity is exact-key only until the identity story lands (FR-048).

Two surfaces name one entity when their normalised names are equal, and never
otherwise: no merge is committed and no candidate is proposed, so the interim
core is precision-first by construction. The merge planes stay *declared* by the
golden schema — this checks that nothing in the core writes them, and that the
cutover report says so out loud.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import graphknows
from evaluation.scenarios import EXACT_KEY_ONLY
from graphknows.ingestion.extraction.protocol import _entity_id

pytestmark = pytest.mark.unit

CORE = Path(graphknows.__file__).resolve().parent
SCHEMA = CORE / "storage" / "arcadedb" / "_schema.py"


@pytest.mark.parametrize("edge", ["MERGED_INTO", "SAME_AS"])
def test_no_merge_edge_is_written(edge: str) -> None:
    """Only the schema may name the merge planes; a writer naming them writes them."""
    offenders = [
        str(module.relative_to(CORE))
        for module in CORE.rglob("*.py")
        if module != SCHEMA and edge in module.read_text(encoding="utf-8")
    ]
    assert offenders == [], offenders


def test_one_key_per_normalised_name() -> None:
    assert _entity_id("M5 MacBook  Air") == _entity_id("m5 macbook air")


def test_distinct_names_stay_distinct() -> None:
    assert _entity_id("Acme Corp") != _entity_id("Acme Corporation")


def test_cutover_report_states_exact_key_identity() -> None:
    assert "exact-key" in EXACT_KEY_ONLY
    assert "MERGED_INTO" in EXACT_KEY_ONLY and "SAME_AS" in EXACT_KEY_ONLY
