"""Verifier coverage is countable per tier.

``verifier_coverage_rows`` aggregates edges/scored/mean per source, so a
producer that skips the verifier gate shows up as a number instead of a
surprise. "Never scored" reports ``None``, not 0.0, keeping it distinguishable
from "scored zero" exactly as ``write_relation`` already promises.

The tests that pinned the DSPy sidecar's gating went with the sidecar: under
the replacement decoder the LLM tier carries its own per-relation confidence in
the verifier's place, and there is no second relation loop left to gate.
"""

from __future__ import annotations

import pytest


def test_verifier_coverage_rows_aggregates_per_tier():
    """Coverage per source: how many edges, how many judged, and how strongly."""
    from graphknows.storage.arcadedb.graph_store import verifier_coverage_rows

    coverage = verifier_coverage_rows(
        [
            {"source": "gliner", "verifier_score": 0.6},
            {"source": "gliner", "verifier_score": 0.8},
            {"source": "gliner", "verifier_score": None},
            {"source": "llm", "verifier_score": None},
            {"source": "llm", "verifier_score": None},
            {"source": "", "verifier_score": 0.5},
        ]
    )

    assert coverage["gliner"] == {"edges": 3, "scored": 2, "mean": pytest.approx(0.7)}
    assert coverage["llm"]["edges"] == 2
    assert coverage["llm"]["scored"] == 0
    # "never scored" must stay distinguishable from "scored zero".
    assert coverage["llm"]["mean"] is None
    assert coverage["unknown"] == {"edges": 1, "scored": 1, "mean": pytest.approx(0.5)}
