"""Tests for graphknows.storage.embedder — local + remote embed paths."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from pydantic import SecretStr


def _settings(*, embed_api_base="", embed_model="BAAI/bge-small-en-v1.5", embed_api_key=""):
    return SimpleNamespace(
        embed_api_base=embed_api_base,
        embed_model=embed_model,
        embed_api_key=SecretStr(embed_api_key),
    )


class TestEmbedLocal:
    def test_empty_list_returns_zero_array(self):
        from graphknows.storage.embedder import EMBED_DIM, embed

        result = embed([])
        assert result.shape == (0, EMBED_DIM)

    def test_local_embed_returns_correct_shape(self):
        from graphknows.storage import embedder

        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        with (
            patch.object(embedder, "_settings", return_value=_settings()),
            patch.object(embedder, "_local_model", return_value=fake_model),
        ):
            result = embedder.embed(["hello world"])

        assert result.shape == (1, 3)
        assert result.dtype == np.float32
        fake_model.encode.assert_called_once()


class TestEmbedRemote:
    def test_remote_embed_hits_configured_endpoint(self):
        from graphknows.storage import embedder

        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["url"] = url
            resp = MagicMock()
            resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
            resp.raise_for_status = MagicMock()
            return resp

        with (
            patch.object(
                embedder,
                "_settings",
                return_value=_settings(
                    embed_api_base="https://api.example.com/v1", embed_api_key="k"
                ),
            ),
            patch("httpx.post", side_effect=fake_post),
        ):
            result = embedder.embed(["hello"])

        assert captured["url"] == "https://api.example.com/v1/embeddings"
        assert result.shape == (1, 2)

    def test_remote_truncates_long_text(self):
        from graphknows.storage import embedder

        sent = []

        def fake_post(url, headers, json, timeout):
            sent.extend(json["input"])
            resp = MagicMock()
            resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1]}]}
            resp.raise_for_status = MagicMock()
            return resp

        with (
            patch.object(
                embedder, "_settings", return_value=_settings(embed_api_base="https://x/v1")
            ),
            patch("httpx.post", side_effect=fake_post),
        ):
            embedder.embed(["x" * 50_000])

        assert len(sent[0]) == embedder._MAX_CHARS_PER_TEXT

    def test_remote_raises_when_data_missing(self):
        import pytest

        from graphknows.storage import embedder

        resp = MagicMock()
        resp.json.return_value = {"error": "quota exceeded"}
        resp.status_code = 429
        resp.text = "quota exceeded"
        resp.raise_for_status = MagicMock()

        with (
            patch.object(
                embedder, "_settings", return_value=_settings(embed_api_base="https://x/v1")
            ),
            patch("httpx.post", return_value=resp),
            pytest.raises(RuntimeError, match="missing 'data' key"),
        ):
            embedder.embed(["hello"])


class TestEmbedOne:
    def test_returns_list_of_floats(self):
        import pytest

        from graphknows.storage import embedder

        embedder._embed_one_cached.cache_clear()
        with patch.object(
            embedder, "embed", return_value=np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        ):
            result = embedder.embed_one("hello-unique")

        assert isinstance(result, list)
        assert result == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
