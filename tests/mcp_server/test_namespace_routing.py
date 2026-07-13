"""The MCP _state cache keeps one runtime per namespace and evicts past its cap."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import pytest_asyncio

from graphknows.server.mcp import _state


class FakeMemory:
    """Stand-in for Memory that records its namespace and close() calls."""

    def __init__(self, settings: object, namespace: str = "") -> None:
        self.namespace = namespace
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest_asyncio.fixture(autouse=True)
async def _clean_state() -> None:
    await _state.close()
    yield
    await _state.close()


@pytest.mark.asyncio
async def test_distinct_runtime_per_namespace() -> None:
    with patch("graphknows.memory.Memory", FakeMemory):
        a1 = await _state.get_runtime("acme")
        a2 = await _state.get_runtime("acme")
        b = await _state.get_runtime("beta")
    assert a1 is a2  # cached per namespace
    assert a1 is not b
    assert a1.namespace == "acme" and b.namespace == "beta"


@pytest.mark.asyncio
async def test_explicit_namespace_overrides_settings_default() -> None:
    fake_settings = type("S", (), {"namespace": "server_default"})()
    with patch.object(_state, "get_settings", return_value=fake_settings), \
         patch("graphknows.memory.Memory", FakeMemory):
        default = await _state.get_runtime("")
        explicit = await _state.get_runtime("other")
    assert default.namespace == "server_default"
    assert explicit.namespace == "other"


@pytest.mark.asyncio
async def test_lru_eviction_closes_oldest() -> None:
    with patch.object(_state, "_MAX_OPEN_NAMESPACES", 2), \
         patch("graphknows.memory.Memory", FakeMemory):
        a = await _state.get_runtime("a")
        await _state.get_runtime("b")
        await _state.get_runtime("a")  # touch 'a' so 'b' is now least-recently-used
        await _state.get_runtime("c")  # exceeds cap of 2 → evicts LRU ('b')
    assert a.closed is False  # 'a' was recently used, survives
    # 'b' was evicted and closed; a fresh get rebuilds it
    assert "b" not in _state._runtimes
