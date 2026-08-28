"""``senses._wordnet`` — the corpus guard moved here from ``skos`` (issue #141).

A missing corpus warns and degrades — it never downloads. #137 shipped
`nltk.download("wordnet")` here because the corpus was absent from the image.
The tests went green, so the criterion was met, and a package strangers
pip-install reached the internet mid-ingest. It could not even work:
nltk.download ignores NLTK_DATA without an explicit download_dir
(scripts/bake_models.py:99), so the fetch landed where nothing reads.
Provisioning is scripts/bake_models.py's job.
"""

from __future__ import annotations

import logging

import pytest

from graphknows.symbolic.ontology import senses


class TestMissingCorpusIsLoudAndOffline:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        senses._wordnet.cache_clear()
        yield
        senses._wordnet.cache_clear()

    def _absent_corpus(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        from nltk.corpus import wordnet as wn

        monkeypatch.setattr(
            wn, "synsets", lambda *a, **k: (_ for _ in ()).throw(LookupError("wordnet"))
        )
        calls = {"downloads": 0}

        def _record(*_a, **_k):
            calls["downloads"] += 1
            return True

        import nltk

        monkeypatch.setattr(nltk, "download", _record)
        return calls

    def test_it_never_downloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._absent_corpus(monkeypatch)
        assert senses._wordnet() is None
        assert calls["downloads"] == 0, "library code reached the network for a missing corpus"

    def test_it_says_what_is_missing_and_how_to_provision_it(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._absent_corpus(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=senses.__name__):
            assert senses._wordnet() is None
        assert caplog.records, "a missing corpus degraded matching silently"
        text = caplog.text
        assert "bake_models" in text, "the warning does not name the provisioning step"
        assert "sense-anchored expansion is disabled" in text, (
            "the warning does not name what the missing corpus disables"
        )
