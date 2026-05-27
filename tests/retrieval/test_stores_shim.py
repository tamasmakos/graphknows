"""Tests for retrieval/stores compat shim — verifies DeprecationWarning fires."""

from __future__ import annotations

import importlib
import sys
import warnings


def _reload_stores_shim() -> object:
    """Force re-import of the shim module in a clean state."""
    mod_name = "graphknows.retrieval.stores"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def test_neo4j_store_import_warns() -> None:
    mod = _reload_stores_shim()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = mod.Neo4jStore  # type: ignore[attr-defined]
    assert any("Neo4jStore" in str(x.message) for x in w), "Expected DeprecationWarning for Neo4jStore"
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_memgraph_store_import_warns() -> None:
    mod = _reload_stores_shim()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = mod.MemgraphStore  # type: ignore[attr-defined]
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_pgvector_store_import_warns() -> None:
    mod = _reload_stores_shim()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = mod.PgVectorStore  # type: ignore[attr-defined]
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_embed_one_warns() -> None:
    mod = _reload_stores_shim()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = mod.embed_one  # type: ignore[attr-defined]
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_unknown_attr_raises_attribute_error() -> None:
    mod = _reload_stores_shim()
    try:
        _ = mod.NonExistent  # type: ignore[attr-defined]
        raise AssertionError("Should have raised AttributeError")
    except AttributeError:
        pass
