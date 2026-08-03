"""Regression guard: the retrieval strategy fork is gone and stays gone.

The agentic (ReAct) retriever was a SECOND full retrieval implementation, live
only under `llm_assisted`. That mode measured worse on the benchmark it exists
for -- 0.750 against llm_free's 0.862, with 3x the abstention rate -- so it was
deleted along with `graphknows/retrieval/query/decompose.py`, its only caller.

The SELECTOR outlived it: a one-member enum, a `strategy` parameter threaded
through `build_retriever`, and a validation expression whose only effect was to
raise on a value nothing could produce. These tests pin the shape that is left
-- one retriever, selected by nothing -- so the fork cannot grow back by
accident.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

import graphknows.settings
from graphknows.retrieval import build_retriever

pytestmark = pytest.mark.unit


def test_build_retriever_selects_between_nothing() -> None:
    """No `strategy` parameter: there is nothing to select between.

    ``extra_channels`` is not a selector — it appends caller-supplied channels
    to the settings-gated defaults, which is how an application reads its own
    graph structure (the ``collect`` half of ``Channel``) without editing the
    registry. The list is still pinned, so an actual fork cannot reappear
    quietly; it just has to be added here deliberately.
    """
    params = inspect.signature(build_retriever).parameters
    assert list(params) == ["settings", "store", "extra_channels"]
    assert params["store"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["extra_channels"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not any("strateg" in p.lower() or "mode" in p.lower() for p in params)


def test_settings_exposes_no_retrieval_strategy() -> None:
    """The one-member enum is gone from the settings module."""
    assert not hasattr(graphknows.settings, "RetrievalStrategy")


def test_agentic_module_is_gone() -> None:
    for name in ("graphknows.retrieval.agentic", "graphknows.retrieval.query.decompose"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
