"""The `python -m evaluation <command>` dispatcher routes to the right module."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from evaluation import __main__ as dispatcher


@pytest.fixture
def dispatched(monkeypatch):
    """Record `(module, argv)` instead of importing and running the command."""
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        dispatcher.importlib,
        "import_module",
        lambda name: SimpleNamespace(main=lambda argv: calls.append((name, argv))),
    )
    return calls


@pytest.mark.parametrize(
    ("command", "module"),
    [
        ("baseline", "evaluation.common.baseline"),
        ("panel", "evaluation.common.panel"),
        ("deadweight", "evaluation.deadweight"),
        ("identity", "evaluation.identity"),
        ("scenario", "evaluation.scenarios"),
        ("download", "evaluation.common.download"),
    ],
)
def test_harness_command_routes_with_remaining_argv(dispatched, monkeypatch, command, module):
    monkeypatch.setattr(sys, "argv", ["evaluation", command, "--rows", "locomo"])
    dispatcher.main()
    assert dispatched == [(module, ["--rows", "locomo"])]


def test_bare_flags_still_run_the_default_benchmark(dispatched, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluation", "--limit", "5"])
    dispatcher.main()
    assert dispatched == [("evaluation.locomo.cli", ["--limit", "5"])]


def test_unknown_command_is_rejected(dispatched, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluation", "nonsense"])
    with pytest.raises(SystemExit, match="nonsense"):
        dispatcher.main()
