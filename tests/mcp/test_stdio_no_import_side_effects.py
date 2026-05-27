"""Verify no import-time dotenv side effects."""
import importlib
import sys

import pytest


def test_import_has_no_dotenv_side_effect(monkeypatch):
    pytest.importorskip("mcp.server.fastmcp")
    called = []
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: called.append(1))
    for mod in list(sys.modules):
        if "graphknows.mcp" in mod:
            del sys.modules[mod]
    importlib.import_module("graphknows.mcp.stdio_server")
    assert called == [], "load_dotenv must not be called at import time"
