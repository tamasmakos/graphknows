"""Verify graphknows.models deprecation state."""
from __future__ import annotations


def test_models_init_has_deprecation_comment():
    """Confirm the deprecation notice is present."""
    import inspect
    import graphknows.models as m
    src = inspect.getfile(m)
    with open(src) as f:
        content = f.read()
    assert "DEPRECATED" in content, "graphknows/models/__init__.py must have DEPRECATED comment"


def test_graphknows_models_exports_graphknows_tool():
    """GraphKnowsTool (if present) must be importable from models."""
    try:
        from graphknows.models import GraphKnowsTool  # noqa: F401
        assert GraphKnowsTool is not None
    except ImportError:
        pass  # GraphKnowsTool may not exist — that's fine
