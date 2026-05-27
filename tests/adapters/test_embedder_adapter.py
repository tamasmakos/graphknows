"""Tests for EmbedderAdapter — uses mocked embed functions."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from graphknows.adapters.embedder_adapter import EmbedderAdapter
from graphknows.adapters.ports.embedder import IEmbedder


def test_embedder_adapter_implements_protocol() -> None:
    """EmbedderAdapter must satisfy the IEmbedder protocol at runtime."""
    adapter = EmbedderAdapter()
    assert isinstance(adapter, IEmbedder)


def test_embed_delegates_to_embed_fn() -> None:
    """embed() should call the underlying embed function and return list[list[float]]."""
    fake_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    with patch("graphknows.adapters.embedder_adapter.EmbedderAdapter.embed") as mock_embed:
        mock_embed.return_value = fake_matrix.tolist()
        adapter = EmbedderAdapter()
        result = adapter.embed(["hello", "world"])
        mock_embed.assert_called_once_with(["hello", "world"])
        assert len(result) == 2


def test_embed_one_delegates_to_embed_one_fn() -> None:
    """embed_one() should call the underlying embed_one function and return list[float]."""
    fake_vec = [0.1, 0.2, 0.3]

    with patch("graphknows.adapters.embedder_adapter.EmbedderAdapter.embed_one") as mock_embed_one:
        mock_embed_one.return_value = fake_vec
        adapter = EmbedderAdapter()
        result = adapter.embed_one("hello")
        mock_embed_one.assert_called_once_with("hello")
        assert result == fake_vec


def test_embed_calls_retrieval_stores_embedder() -> None:
    """embed() should import and call graphknows.retrieval.stores.embedder.embed."""
    fake_matrix = np.array([[0.1, 0.2]], dtype=np.float32)

    with patch(
        "graphknows.retrieval.stores.embedder.embed", return_value=fake_matrix
    ) as mock_embed:
        adapter = EmbedderAdapter(model_name="openai/text-embedding-3-small")
        result = adapter.embed(["test"])
        mock_embed.assert_called_once_with(["test"], "openai/text-embedding-3-small")
        assert isinstance(result, list)
        assert len(result) == 1


def test_embed_one_calls_retrieval_stores_embedder() -> None:
    """embed_one() should import and call graphknows.retrieval.stores.embedder.embed_one."""
    fake_vec = [0.1, 0.2, 0.3]

    with patch(
        "graphknows.retrieval.stores.embedder.embed_one", return_value=fake_vec
    ) as mock_embed_one:
        adapter = EmbedderAdapter()
        result = adapter.embed_one("test text")
        mock_embed_one.assert_called_once_with("test text", "openai/text-embedding-3-small")
        assert result == fake_vec
