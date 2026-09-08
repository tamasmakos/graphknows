"""FR-017 / SC-011: settings are built a bounded number of times per process.

``GraphKnowsSettings()`` costs ~140 ms (environment + ``.env`` parsing plus
validation). Before this, roughly fifteen call sites paid it ad hoc, some of
them once per message and once per chunk. They now route through
:func:`graphknows.settings.get_settings`, whose ``lru_cache`` makes the count
per process a constant rather than a function of the workload.
"""

from __future__ import annotations

import ast
import os
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


def test_passed_settings_instance_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-013: the ingest handler reads the instance it was handed, not the environment.

    ``get_settings()`` alone would not close this: a handler that called it
    would still read the process environment, so ``Memory(settings=s)`` would
    be silently ignored on the write path.
    """
    import graphknows.ingestion.extraction.entities as entities
    import graphknows.storage as storage
    import graphknows.symbolic.ontology as ontology
    from graphknows.ingestion import build_stm_service

    monkeypatch.setattr(storage, "build_graph_store", lambda settings, ns: object())
    monkeypatch.setattr(entities, "build_decoder", lambda settings: object())
    monkeypatch.setattr(ontology, "load_ontology_index", lambda source: None)

    monkeypatch.setenv("GRAPHKNOWS_EXTRACT_WINDOW", "3")
    get_settings.cache_clear()
    assert get_settings().extract_window == 3

    passed = GraphKnowsSettings(extract_window=11)
    handler = build_stm_service(passed, "ns")._handler()

    assert handler._settings is passed
    assert handler._extract_window == 11


def test_dotenv_only_variable_takes_effect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-014: a knob set ONLY in ``.env`` reaches the code that reads it.

    pydantic-settings parses ``.env`` into the settings object and never writes
    ``os.environ``, so every knob still read through ``os.environ.get`` was inert
    for anyone who configured it the documented way. ``GRAPHKNOWS_RELEX_MODEL``
    is the documented case: it is in configuration.md and it did nothing.
    """
    from graphknows.ranking import frame_boost

    (tmp_path / ".env").write_text(
        "GRAPHKNOWS_RELEX_MODEL=acme/relex-tiny\nGRAPHKNOWS_FRAME_RERANK=false\n",
        encoding="utf-8",
    )
    for var in ("GRAPHKNOWS_RELEX_MODEL", "GRAPHKNOWS_FRAME_RERANK"):
        monkeypatch.delenv(var, raising=False)
    # ``env_file=".env"`` is resolved against the working directory.
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    assert os.environ.get("GRAPHKNOWS_RELEX_MODEL") is None
    assert get_settings().relex_model == "acme/relex-tiny"
    # And the consumer reads the field rather than the environment behind it.
    assert os.environ.get("GRAPHKNOWS_FRAME_RERANK") is None
    assert frame_boost.enabled() is False
