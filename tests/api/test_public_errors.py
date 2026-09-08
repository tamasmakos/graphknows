"""Every failure escaping a documented public entry point is a GraphKnowsError.

The launch-remediation audit found three raw exception types escaping the
public methods of ``Memory`` and the MCP tool surface (FR-024): a raw
``httpx`` transport failure, a pydantic ``ValidationError`` (bad scope), and a
bare ``ValueError`` (bad mode). Each test below drives the *public* entry
point — not the internal raiser — and asserts ``pytest.raises(GraphKnowsError)``
(SC-016).
"""

from __future__ import annotations

import pytest

from graphknows.exceptions import ConfigurationError, GraphKnowsError, StoreError
from graphknows.memory import Memory
from graphknows.models.scope import MemoryScope
from graphknows.settings import GraphKnowsSettings


async def test_store_transport_failure_out_of_memory_is_graphknows_error() -> None:
    """A transport failure reaching ArcadeDB surfaces as StoreError, not httpx.

    Port 1 refuses the connection immediately (no service listens there), so
    this stays hermetic and fast — no live ArcadeDB required.
    """
    settings = GraphKnowsSettings(arcadedb_url="http://127.0.0.1:1")
    memory = Memory(settings=settings)
    with pytest.raises(StoreError):
        await memory.stats()


def test_bad_scope_out_of_memory_scope_is_graphknows_error() -> None:
    """MemoryScope's "at least one" validator escapes as ConfigurationError."""
    with pytest.raises(ConfigurationError):
        MemoryScope()


def test_bad_mode_out_of_memory_init_is_graphknows_error() -> None:
    """An unrecognised mode string escapes ``Memory.__init__`` as ConfigurationError."""
    with pytest.raises(ConfigurationError):
        Memory(mode="not-a-real-mode")


@pytest.mark.parametrize("error_cls", [StoreError, ConfigurationError])
def test_each_escaping_type_is_a_graphknows_error(error_cls: type[Exception]) -> None:
    assert issubclass(error_cls, GraphKnowsError)
