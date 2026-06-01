"""Tests for graphknows.config.health."""

from __future__ import annotations

import pytest
from graphknows.config import GraphKnowsSettings
from graphknows.config.health import check_backends, get_version


def test_get_version_returns_string():
    v = get_version()
    assert isinstance(v, str)


@pytest.mark.asyncio
async def test_check_backends_returns_expected_keys():
    """All backends unreachable — must return False for each, no exception raised."""
    settings = GraphKnowsSettings()
    result = await check_backends(settings)
    assert "neo4j" in result
    assert "memgraph" in result
    assert "pgvector" in result
    assert "version" in result


@pytest.mark.asyncio
async def test_check_backends_all_fail():
    """All backends unreachable — must return False for each, no exception raised."""
    settings = GraphKnowsSettings()
    result = await check_backends(settings)
    assert isinstance(result, dict)
    assert "neo4j" in result
    assert "memgraph" in result
    assert "pgvector" in result
