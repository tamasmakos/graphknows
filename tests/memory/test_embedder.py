"""Tests for graphknows.memory.stores.embedder — embed, embed_one, get_embedder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEmbed:
    def test_empty_list_returns_zero_array(self):
        from graphknows.memory.stores.embedder import EMBED_DIM, embed

        result = embed([])
        assert result.shape == (0, EMBED_DIM)

    def test_raises_without_api_key(self, monkeypatch):
        from graphknows.memory.stores.embedder import embed

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPEN_ROUTER_API", raising=False)
        monkeypatch.setattr("graphknows.memory.stores.embedder._repo_env_key", lambda: "")
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            embed(["hello"])

    def test_successful_embed_returns_correct_shape(self, monkeypatch):
        from graphknows.memory.stores.embedder import EMBED_DIM, embed

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        fake_response = {"data": [{"index": 0, "embedding": [0.1] * EMBED_DIM}]}

        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = fake_response
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = embed(["hello world"])

        assert result.shape == (1, EMBED_DIM)
        assert result.dtype == np.float32

    def test_truncates_long_text(self, monkeypatch):
        from graphknows.memory.stores.embedder import _MAX_CHARS_PER_TEXT, EMBED_DIM, embed

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        sent_inputs = []

        def fake_post(url, headers, json, timeout):
            sent_inputs.extend(json["input"])
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1] * EMBED_DIM}]}
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("httpx.post", side_effect=fake_post):
            embed(["x" * 50_000])

        assert len(sent_inputs[0]) == _MAX_CHARS_PER_TEXT

    def test_raises_when_data_missing_from_response(self, monkeypatch):
        from graphknows.memory.stores.embedder import embed

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"error": "quota exceeded"}
            mock_resp.status_code = 429
            mock_resp.text = "quota exceeded"
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            with pytest.raises(RuntimeError, match="missing 'data' key"):
                embed(["hello"])


class TestEmbedOne:
    def test_returns_list_of_floats(self, monkeypatch):
        from graphknows.memory.stores.embedder import EMBED_DIM, embed_one

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        with patch("graphknows.memory.stores.embedder.embed") as mock_embed:
            mock_embed.return_value = np.array([[0.1] * EMBED_DIM], dtype=np.float32)
            result = embed_one("hello")

        assert isinstance(result, list)
        assert len(result) == EMBED_DIM


class TestGetEmbedder:
    def test_shim_encode_calls_embed(self, monkeypatch):
        from graphknows.memory.stores.embedder import EMBED_DIM, get_embedder

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        with patch("graphknows.memory.stores.embedder.embed") as mock_embed:
            mock_embed.return_value = np.array([[0.2] * EMBED_DIM], dtype=np.float32)
            shim = get_embedder()
            result = shim.encode(["test text"])

        mock_embed.assert_called_once_with(["test text"], "openai/text-embedding-3-small")
        assert result.shape == (1, EMBED_DIM)

    def test_shim_wraps_string_in_list(self, monkeypatch):
        from graphknows.memory.stores.embedder import EMBED_DIM, get_embedder

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

        with patch("graphknows.memory.stores.embedder.embed") as mock_embed:
            mock_embed.return_value = np.array([[0.1] * EMBED_DIM], dtype=np.float32)
            shim = get_embedder()
            shim.encode("single string")

        call_args = mock_embed.call_args[0][0]
        assert call_args == ["single string"]
