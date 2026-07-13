"""Tests for _startup_warmup in stdio_server."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_startup_warmup_calls_embed_one() -> None:
    """Warmup should invoke embed_one once."""
    from graphknows.server.mcp.stdio_server import _startup_warmup

    with patch("graphknows.storage.embedder.embed_one") as mock_embed:
        mock_embed.return_value = [0.1, 0.2]
        await _startup_warmup()
    mock_embed.assert_called_once()


@pytest.mark.asyncio
async def test_startup_warmup_non_fatal() -> None:
    """Exceptions in warmup must not propagate."""
    from graphknows.server.mcp.stdio_server import _startup_warmup

    with patch(
        "graphknows.storage.embedder.embed_one",
        side_effect=RuntimeError("API down"),
    ):
        await _startup_warmup()  # must not raise
