"""A model's declared ``suffix`` must reach the encoder.

Regression for the ontology class-matching inversion: ``zembed-1`` declares
``suffix: "<|im_end|>\\n"`` in ``config_sentence_transformers.json`` and pools the
**last token** (``pooling_mode_lasttoken``). sentence-transformers applies
``prompts`` (prefixes) but has no suffix support at all, so the sentinel was
never appended and the pooled vector was taken at an arbitrary content token.

The observable damage: for "He finally bought that vintage motorcycle he had
been saving up for", the ontology class ``Possession`` ranked 11/11 while
``Emotion`` ranked 1st. Appending the declared suffix moved ``Possession`` to
2/11 and ``Place`` from 7/11 to 1/11 on the sibling case.

Tested here rather than against the real model because zembed-1 is a 4B opt-in
download; the bug pattern is "text reaches the encoder without its declared
sentinel", which is exactly what these assert.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from graphknows.exceptions import ConfigurationError
from graphknows.storage import embedder


@pytest.fixture(autouse=True)
def _clear_caches():
    embedder._model_suffix.cache_clear()
    embedder._local_model.cache_clear()
    yield
    embedder._model_suffix.cache_clear()
    embedder._local_model.cache_clear()


def _write_st_config(directory: Path, payload: dict) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config_sentence_transformers.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return str(directory)


class TestModelSuffix:
    def test_declared_suffix_is_read(self, tmp_path: Path) -> None:
        path = _write_st_config(tmp_path / "m", {"suffix": "<|im_end|>\n"})
        assert embedder._model_suffix(path) == "<|im_end|>\n"

    def test_absent_suffix_is_empty(self, tmp_path: Path) -> None:
        """bge-small declares no suffix — the common case must stay a no-op."""
        path = _write_st_config(tmp_path / "m", {"prompts": {"query": "q: "}})
        assert embedder._model_suffix(path) == ""

    def test_unknown_model_is_empty_not_an_error(self) -> None:
        """A miss must not raise: embedding is on the ingest hot path."""
        assert embedder._model_suffix("definitely/not-a-real-model-xyz") == ""

    def test_malformed_config_is_empty_not_an_error(self, tmp_path: Path) -> None:
        d = tmp_path / "m"
        d.mkdir()
        (d / "config_sentence_transformers.json").write_text("{not json", encoding="utf-8")
        assert embedder._model_suffix(str(d)) == ""


class TestSuffixReachesTheEncoder:
    """The bug pattern: what the model actually receives."""

    @staticmethod
    def _fake_model(seen: list[list[str]]):
        class _M:
            def encode(self, texts, **_kw):
                seen.append(list(texts))
                return np.zeros((len(texts), 4), dtype=np.float32)

        return _M()

    def test_declared_suffix_is_appended_to_every_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_st_config(tmp_path / "m", {"suffix": "<|im_end|>\n"})
        seen: list[list[str]] = []
        monkeypatch.setattr(embedder, "_local_model", lambda _n: self._fake_model(seen))

        embedder._embed_local(["alpha", "beta"], path)

        assert seen == [["alpha<|im_end|>\n", "beta<|im_end|>\n"]]

    def test_no_suffix_leaves_text_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_st_config(tmp_path / "m", {})
        seen: list[list[str]] = []
        monkeypatch.setattr(embedder, "_local_model", lambda _n: self._fake_model(seen))

        embedder._embed_local(["alpha"], path)

        assert seen == [["alpha"]]


class TestDeclaredDimension:
    """A module stack that exposes no dimension must name the model.

    Both sentence-transformers accessors are typed ``int | None``, so the old
    ``int(get_dim())`` raised ``TypeError: int() argument must be...`` naming
    neither the model nor the cause — and every vector written afterwards would
    have been sized against a dimension nobody could account for.
    """

    @staticmethod
    def _patch_sentence_transformers(monkeypatch: pytest.MonkeyPatch, dim: int | None) -> None:
        import sentence_transformers

        class _M:
            def __init__(self, *_args: object, **_kwargs: object) -> None: ...

            def get_sentence_embedding_dimension(self) -> int | None:
                return dim

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _M)

    def test_absent_dimension_names_the_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_sentence_transformers(monkeypatch, None)

        with pytest.raises(ConfigurationError, match="silent/model"):
            embedder._local_model("silent/model")

    def test_declared_dimension_becomes_embed_dim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(embedder, "EMBED_DIM", embedder.EMBED_DIM)  # restored on teardown
        self._patch_sentence_transformers(monkeypatch, 384)

        embedder._local_model("sized/model")

        assert embedder.EMBED_DIM == 384
