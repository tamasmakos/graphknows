"""The bring-your-own-ontology tool: importable without rdflib, honest when it is missing.

``digest`` moved into the package precisely so users can digest their own RDF,
which means a base install (no ``ontology`` extra) must be able to *import* it —
only ``build`` may need rdflib, and it must say which extra to install rather
than raising a bare ImportError at the user.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from graphknows.exceptions import MissingExtraError
from graphknows.symbolic.ontology import digest


def test_module_imports_without_rdflib() -> None:
    """Importing the tool must not pull the RDF stack; only `build` may."""
    assert callable(digest.build)
    assert callable(digest.main)


def test_build_names_the_ontology_extra_when_rdflib_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A base install must be told `ontology`, not handed an ImportError."""
    real_import = builtins.__import__

    def _no_rdflib(name: str, *args: object, **kwargs: object) -> object:
        if name == "rdflib" or name.startswith("rdflib."):
            raise ImportError("No module named 'rdflib'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_rdflib)

    with pytest.raises(MissingExtraError) as exc:
        digest.build([Path("nonexistent.ttl")])
    assert "ontology" in str(exc.value)
