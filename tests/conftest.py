"""Shared pytest configuration.

sys.path is managed via pytest.ini ``pythonpath`` entries.
All search-package imports use relative imports (not ``from search.*``),
so there is no two-class-object problem and no aliasing is needed here.

Module-stub isolation
---------------------
The legacy ``test_langgraph_agent_search_*.py`` files stub out ``langgraph`` and
``langchain_core`` modules in ``sys.modules`` so they can run without the real deps.
Those stubs must not leak into real-langgraph tests, and real modules must not leak
into the stub tests.

The ``_restore_stub_modules`` autouse fixture saves and restores the relevant
``sys.modules`` entries around every test.
"""

from __future__ import annotations

import sys

import pytest

_STUB_KEYS = (
    "langgraph.prebuilt",
    "langchain_core.tools",
    "langchain_openai",
    "langchain_core.messages",
    "services.agent.langgraph_agent",
    "agent.langgraph_agent",
)


@pytest.fixture(autouse=True)
def _reset_settings_singletons():
    """Reset module-level _settings singletons so env-patching tests stay isolated."""
    import sys
    yield
    for mod_name in ("graphknows.mcp.stdio_server", "graphknows.cli.memory"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_settings"):
            mod._settings = None


@pytest.fixture(autouse=True)
def _restore_stub_modules():
    """Save and restore stub-sensitive ``sys.modules`` entries around each test."""
    sentinel = object()
    saved = {k: sys.modules.get(k, sentinel) for k in _STUB_KEYS}
    yield
    for k, v in saved.items():
        if v is sentinel:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
