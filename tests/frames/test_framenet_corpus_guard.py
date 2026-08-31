"""A missing FrameNet corpus is an ENVIRONMENT defect, not something library
code routes around.

``graphknows.symbolic.framenet.framenet._framenet()`` is the single accessor
every FrameNet caller in the package routes through. A package installed from
PyPI must not reach the network mid-ingest to fetch a corpus, and
``nltk.download()`` without ``download_dir=`` ignores ``NLTK_DATA`` anyway
(see scripts/bake_models.py). On a corpus miss, ``_framenet()`` must raise a
loud, specific error pointing at the provisioning step (``make bake``)
instead of downloading or returning an unusable handle.
"""

from __future__ import annotations

import pytest

from graphknows.symbolic.framenet import framenet

pytestmark = pytest.mark.unit


class _MissingCorpusStub:
    def frames(self) -> None:
        raise LookupError("Resource framenet_v17 not found.")


def test_missing_corpus_raises_loudly_instead_of_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nltk.corpus.framenet", _MissingCorpusStub())

    with pytest.raises(RuntimeError) as exc_info:
        framenet._framenet()

    message = str(exc_info.value)
    assert "framenet_v17" in message
    assert "make bake" in message
