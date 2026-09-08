"""The shared live-ArcadeDB guard: skips by default, fails when CI demands a DB.

Needs no database — reachability is stubbed both ways. ``GRAPHKNOWS_REQUIRE_ARCADEDB=1``
is CI's promise that the service container is up, so a skip under it would report an
integration suite that never ran as green (FR-008, SC-005).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import conftest


def _run_guard(monkeypatch: pytest.MonkeyPatch, *, reachable: bool, flag: str | None) -> None:
    monkeypatch.setattr(conftest, "_arcadedb_reachable", lambda: reachable)
    if flag is None:
        monkeypatch.delenv("GRAPHKNOWS_REQUIRE_ARCADEDB", raising=False)
    else:
        monkeypatch.setenv("GRAPHKNOWS_REQUIRE_ARCADEDB", flag)
    conftest.require_arcadedb()


def test_unreachable_skips_when_the_flag_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(pytest.skip.Exception):
        _run_guard(monkeypatch, reachable=False, flag=None)


def test_unreachable_fails_when_the_flag_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(pytest.fail.Exception):
        _run_guard(monkeypatch, reachable=False, flag="1")


def test_reachable_passes_through_under_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _run_guard(monkeypatch, reachable=True, flag="1")


def test_no_test_module_keeps_its_own_reachability_guard() -> None:
    """One shared guard: a private copy is a module CI cannot hold to the flag."""
    tests_root = Path(__file__).resolve().parent.parent
    offenders = [
        path.relative_to(tests_root).as_posix()
        for path in tests_root.rglob("test_*.py")
        if path != Path(__file__).resolve()
        and "def _arcadedb_reachable" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
