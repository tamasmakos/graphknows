"""FR-017 / SC-011: settings are built a bounded number of times per process.

``GraphKnowsSettings()`` costs ~140 ms (environment + ``.env`` parsing plus
validation). Before this, roughly fifteen call sites paid it ad hoc, some of
them once per message and once per chunk. They now route through
:func:`graphknows.settings.get_settings`, whose ``lru_cache`` makes the count
per process a constant rather than a function of the workload.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from graphknows.settings import GraphKnowsSettings, get_settings

_PACKAGE = Path(__file__).resolve().parents[2] / "graphknows"


def _direct_construction_sites() -> list[str]:
    """``file:line`` for every bare ``GraphKnowsSettings()`` call under the package.

    ``settings.py`` itself is where the accessor builds it, so it is excluded.
    """
    sites = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path.name == "settings.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # The class is often imported under an alias (``as _Settings``), so
        # resolve the local names it is bound to rather than matching one spelling.
        names = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name == "GraphKnowsSettings"
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in names:
                    sites.append(f"{path.relative_to(_PACKAGE.parent).as_posix()}:{node.lineno}")
    return sites


def test_settings_construction_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated reads build one instance, and no module builds its own."""
    calls = 0
    original = GraphKnowsSettings.__init__

    def counting_init(self, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        original(self, **kwargs)

    monkeypatch.setattr(GraphKnowsSettings, "__init__", counting_init)
    get_settings.cache_clear()

    first = get_settings()
    for _ in range(50):
        assert get_settings() is first
    assert calls == 1

    assert _direct_construction_sites() == []
