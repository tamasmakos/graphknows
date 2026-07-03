"""Guard tests for neural coreference resolution.

The fast tests lock the scope decision and the soft-fail contract without loading
the model; a single opt-in integration test exercises the real fastcoref path
when the cached model is available.
"""

from __future__ import annotations

import pytest

from graphknows.extraction.relations import coref


def test_scope_is_singular_animate_only():
    """Plural/first/second person are deliberately out of scope (corpus finding:
    plural 'they' clusters to inanimate antecedents and corrupts text)."""
    assert "she" in coref._THIRD_PERSON
    assert "he" in coref._THIRD_PERSON
    for excluded in ("they", "them", "their", "we", "us", "i", "you"):
        assert excluded not in coref._THIRD_PERSON


def test_is_name_accepts_names_rejects_pronouns():
    assert coref._is_name("Melanie")
    assert coref._is_name("Matt Patterson")
    assert not coref._is_name("she")
    assert not coref._is_name("")
    assert not coref._is_name("the dog")  # lowercase -> not a name


def test_resolve_is_identity_without_model(monkeypatch):
    """Soft-fail: when the model is unavailable, text is returned unchanged so
    ingestion never breaks on a missing/broken coref dependency."""
    monkeypatch.setattr(coref, "_get_model", lambda: None)
    text = "Caroline went to Rome. She loved it."
    assert coref.resolve_coref(text) == text
    assert coref.resolve_coref("") == ""


def test_resolve_rewrites_singular_leaves_plural():
    """Integration: real model rewrites 'he/she' to the named antecedent and
    leaves 'they' untouched. Skips if the cached model can't load in this env."""
    if coref._get_model() is None:
        pytest.skip("coref model unavailable in this environment")
    out = coref.resolve_coref("Matt Patterson is talented. He wrote a song. They cheered.")
    assert "Matt Patterson wrote" in out or "Matt Patterson" in out.split(". ")[1]
    assert "They cheered" in out  # plural pronoun left as-is
