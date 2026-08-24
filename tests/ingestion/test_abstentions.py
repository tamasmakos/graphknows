"""Every gate that declines to write is counted (#139 / B15).

The defect this closes: `_write_relations` built a full report — blocked_domain,
blocked_range, unsupported, pruned_endpoint, conformance, hallucination — and
its caller discarded the return value. The numbers were computed and thrown
away, reaching a per-chunk log line and nothing else. Same shape as the
verifier_score that was computed and never read.

An abstention nobody can count is indistinguishable from an extractor that
found nothing, which is the difference between "the gate is working" and "the
gate ate the graph".
"""

from __future__ import annotations

import pytest

from graphknows.models.ingest import IngestResult

pytestmark = pytest.mark.unit


def test_the_result_carries_a_tally() -> None:
    r = IngestResult(session_id="s", file_id="f", title="", abstentions={"blocked_range": 3})
    assert r.abstentions == {"blocked_range": 3}


def test_the_tally_defaults_to_empty_not_absent() -> None:
    """Empty means 'no gate fired'. That is a different claim from 'no gate ran',
    and a caller must be able to distinguish them without a KeyError."""
    assert IngestResult(session_id="s", file_id="f", title="").abstentions == {}


def test_only_gates_that_fired_are_reported() -> None:
    """A tally full of zeros hides the ones that matter."""
    from collections import defaultdict

    tally: dict[str, int] = defaultdict(int)
    tally["blocked_range"] += 2
    tally["unsupported"] += 0
    reported = {k: v for k, v in tally.items() if v}
    assert reported == {"blocked_range": 2}


def test_the_flush_contract_exposes_it() -> None:
    """flush_memory's return dict is the only place an operator sees these."""
    import inspect

    from graphknows.memory import Memory

    src = inspect.getsource(Memory.flush_memory)
    assert '"abstentions": abstentions' in src, "flush stopped reporting the tally"
