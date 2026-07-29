"""The recall ceiling must not silently truncate the pool.

`_MAX_RECALL_TOP_K` capped every recall at 100 hits. It was introduced as a
runaway-request bound, but it is indistinguishable from "that is all the memory
holds" at the call site, and it corrupts measurement: a recall-vs-depth sweep on
conv-30 plateaued at 0.827 and read as a STRUCTURAL defect -- gold chunks
unreachable at any depth. With the cap lifted the same sweep reaches 1.000 at
k=369, i.e. every gold chunk IS retrievable and the deficit is purely ranking:

    k        recall   zero-recall
    25        0.579        31/81
    100       0.839        10/81
    200       0.933         3/81
    369       1.000         0/81

A bound that changes the conclusion of a measurement is not a safety rail.
"""

from __future__ import annotations

import pytest

from graphknows import memory as gk_memory

pytestmark = pytest.mark.unit


def test_no_hard_ceiling_constant() -> None:
    """The module must not carry a cap that silently truncates a caller's request."""
    assert not hasattr(gk_memory, "_MAX_RECALL_TOP_K"), (
        "a recall cap re-introduces the measurement artifact it caused before"
    )
