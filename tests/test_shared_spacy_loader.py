"""Identical spaCy configs must share one loaded pipeline across all callers.

Protects the fix for issue #239: ``load_spacy_model`` is the single entry
point every spaCy load in the package goes through, and it must memoize per
``(name, options)`` instead of loading a fresh pipeline on every call — five
call sites were each growing their own private cache because they could not
see each other's loaded model.
"""

from __future__ import annotations

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


def test_five_call_sites_share_at_most_two_pipelines(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two configurations in use (full pipeline, and disable=['ner']) must
    resolve to exactly two resident pipelines, not five."""
    calls = _install_fake_load(monkeypatch)
    monkeypatch.setenv("GRAPHKNOWS_SPACY_MODEL", "gk_fake_model")

    import graphknows.worth as worth
    from graphknows.ingestion.consolidation.entity_resolution import _parser
    from graphknows.ingestion.extraction.entities.extractor import _SpacySchemaMiner
    from graphknows.symbolic.framenet.framenet import _spacy
    from graphknows.symbolic.ontology.skos import _lemmatise

    worth_pipeline = worth._nlp()
    resolution_pipeline = _parser()
    miner_pipeline = _SpacySchemaMiner(model_name="gk_fake_model").nlp
    _lemmatise({"dog"})
    framenet_pipeline = _spacy()

    no_option_calls = [c for c in calls if c[1] == {}]
    disable_ner_calls = [c for c in calls if c[1] == {"disable": ["ner"]}]

    assert len(calls) == 2, (
        f"expected 2 resident spaCy pipelines (one plain, one disable=['ner']), "
        f"but {len(calls)} pipelines were loaded: {calls}"
    )
    assert len(no_option_calls) == 1, (
        f"expected 1 spacy.load call with no options, got {len(no_option_calls)}: {calls}"
    )
    assert len(disable_ner_calls) == 1, (
        f"expected 1 spacy.load call with disable=['ner'], got {len(disable_ner_calls)}: {calls}"
    )

    assert worth_pipeline is resolution_pipeline is miner_pipeline, (
        "worth._nlp(), entity_resolution._parser() and _SpacySchemaMiner.nlp "
        "must share the same plain pipeline instance"
    )
    assert framenet_pipeline is not worth_pipeline, (
        "framenet._spacy() uses disable=['ner'] and must not share the plain pipeline"
    )
