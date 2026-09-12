"""Identical spaCy configs must share one loaded pipeline across all callers.

Protects the fix for issue #239: ``load_spacy_model`` is the single entry
point every spaCy load in the package goes through, and it must memoize per
``(name, options)`` instead of loading a fresh pipeline on every call — the
call sites were each growing their own private cache because they could not
see each other's loaded model.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

import graphknows.nlp as nlp_module

pytestmark = pytest.mark.unit


class _FakePipeline:
    """A stand-in spaCy pipeline: callable."""

    def __call__(self, text: str) -> list[Any]:
        return []


@pytest.fixture(autouse=True)
def _clear_model_cache() -> Any:
    nlp_module._MODELS.clear()
    yield
    nlp_module._MODELS.clear()


def _install_fake_load(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    import spacy

    calls: list[tuple[str, dict[str, Any]]] = []

    def fake(name: str, **options: Any) -> _FakePipeline:
        calls.append((name, options))
        return _FakePipeline()

    monkeypatch.setattr(spacy, "load", fake)
    return calls


def test_identical_config_shares_one_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_load(monkeypatch)

    from graphknows.nlp import load_spacy_model

    first = load_spacy_model("gk_fake_model")
    second = load_spacy_model("gk_fake_model")

    assert first is second, "same (name, options) must return the same pipeline object"
    assert len(calls) == 1, f"expected 1 spacy.load call, spacy.load was called {len(calls)} times"


def test_distinct_options_get_distinct_pipelines(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_load(monkeypatch)

    from graphknows.nlp import load_spacy_model

    plain = load_spacy_model("gk_fake_model")
    disabled_first = load_spacy_model("gk_fake_model", disable=["ner"])
    disabled_second = load_spacy_model("gk_fake_model", disable=["ner"])

    assert disabled_first is not plain, "a different option set must not reuse the plain pipeline"
    assert disabled_first is disabled_second, (
        "repeating the same (name, options) must return the same pipeline object"
    )

    disabled_calls = [options for _, options in calls if options.get("disable") == ["ner"]]
    assert len(disabled_calls) == 1, (
        f"expected exactly 1 spacy.load call with disable=['ner'], got {len(disabled_calls)}"
    )


def test_call_sites_share_one_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one configuration left in use must resolve to one resident pipeline.

    The ``disable=['ner']`` configuration went with the frame plane's parser;
    every remaining call site asks for the plain pipeline.
    """
    calls = _install_fake_load(monkeypatch)
    monkeypatch.setenv("GRAPHKNOWS_SPACY_MODEL", "gk_fake_model")

    from graphknows.ingestion.extraction.entities.extractor import _SpacySchemaMiner
    from graphknows.ingestion.extraction.entities.hygiene import _parser
    from graphknows.symbolic.ontology.skos import _lemmatise

    hygiene_pipeline = _parser()
    miner_pipeline = _SpacySchemaMiner(model_name="gk_fake_model").nlp
    _lemmatise({"dog"})

    no_option_calls = [c for c in calls if c[1] == {}]

    assert len(calls) == 1, (
        f"expected 1 resident spaCy pipeline, but {len(calls)} were loaded: {calls}"
    )
    assert len(no_option_calls) == 1, (
        f"expected 1 spacy.load call with no options, got {len(no_option_calls)}: {calls}"
    )
    assert hygiene_pipeline is miner_pipeline, (
        "hygiene._parser() and _SpacySchemaMiner.nlp must share the same plain pipeline instance"
    )


def test_concurrent_loads_build_one_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Threads racing a cold cache (e.g. ``asyncio.to_thread`` workers) must not
    each load their own copy of a hundreds-of-MB model (FR-039)."""
    import spacy

    calls: list[tuple[str, dict[str, Any]]] = []

    def slow_fake(name: str, **options: Any) -> _FakePipeline:
        # Widens the race window so two threads both observing a cache miss,
        # without the lock, would both reach here before either writes back.
        time.sleep(0.05)
        calls.append((name, options))
        return _FakePipeline()

    monkeypatch.setattr(spacy, "load", slow_fake)

    from graphknows.nlp import load_spacy_model

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    results: list[Any] = [None] * n_threads

    def worker(i: int) -> None:
        barrier.wait()  # release every thread at once, all racing a cold cache
        results[i] = load_spacy_model("gk_fake_model")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1, (
        f"expected exactly 1 spacy.load call across {n_threads} racing threads, "
        f"got {len(calls)}: {calls}"
    )
    assert all(r is results[0] for r in results), (
        "every thread must receive the same pipeline instance"
    )
