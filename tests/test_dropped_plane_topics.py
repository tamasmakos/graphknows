"""The topic plane is gone: no package, no CLI, no MCP tool, no facade method.

A plane is only dropped once nothing can still reach it, so this checks the
three deleted surfaces *and* that no module left behind imports the package or
advertises a topic tool.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from graphknows.memory import Memory
from graphknows.server.mcp import tools

REPO_ROOT = Path(__file__).resolve().parents[1]

DROPPED_PATHS = (
    "graphknows/topics",
    "graphknows/cli/topics.py",
    "graphknows/server/mcp/tools/topics.py",
)


def test_topic_modules_are_gone() -> None:
    for relative in DROPPED_PATHS:
        assert not (REPO_ROOT / relative).exists(), f"{relative} still exists"
    assert importlib.util.find_spec("graphknows.topics") is None


def test_no_module_imports_the_topic_package() -> None:
    """Nothing under `graphknows/` reaches for the deleted package."""
    importers = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "graphknows").rglob("*.py")
        if "graphknows.topics" in path.read_text(encoding="utf-8")
    ]
    assert importers == []


def test_no_topic_tool_is_advertised() -> None:
    """`tools.__all__` is the authoritative MCP inventory (see its docstring)."""
    assert [name for name in tools.__all__ if "topic" in name] == []


def test_memory_exposes_no_topic_verb() -> None:
    assert [name for name in dir(Memory) if "topic" in name] == []


def test_no_topic_console_script() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "graphknows-topics" not in pyproject
