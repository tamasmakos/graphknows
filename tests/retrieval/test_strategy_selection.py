"""Retrieval strategy: there is exactly one, and it is deterministic.

The agentic (ReAct) retriever was a SECOND full retrieval implementation, live
only under `llm_assisted`. That mode measured worse on the benchmark it exists
for -- 0.750 against llm_free's 0.862, with 3x the abstention rate -- so the
codebase carried ~600 lines of alternative retrieval whose only configuration
scored below the path it replaced.

Deleted with `graphknows/retrieval/query/decompose.py`, its only caller.
"""

from __future__ import annotations

import pytest

from graphknows.settings import GraphKnowsSettings, MemoryMode, RetrievalStrategy

pytestmark = pytest.mark.unit


def test_only_one_strategy_exists() -> None:
    assert [s.value for s in RetrievalStrategy] == ["det"]


@pytest.mark.parametrize("mode", [MemoryMode.llm_free, MemoryMode.llm_assisted])
def test_every_mode_resolves_to_deterministic(mode: MemoryMode) -> None:
    s = GraphKnowsSettings(GRAPHKNOWS_MODE=mode)
    assert s.retrieval_strategy is RetrievalStrategy.det


def test_agentic_module_is_gone() -> None:
    import importlib

    for name in ("graphknows.retrieval.agentic", "graphknows.retrieval.query.decompose"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
