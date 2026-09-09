"""The advertised MCP tool surface: no namespace argument, no ungated destructive tool."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DESTRUCTIVE = {"memory_drop_namespace", "memory_purge"}

# The gate is applied at import time, so it can only be observed in a fresh
# interpreter. Run from a scratch cwd as well, so a developer's own ``.env``
# cannot decide the result.
_PROBE = """
import asyncio, json
from graphknows.server.mcp import tools
from graphknows.server.mcp._app import app
from graphknows.server.mcp.tools.admin import ENABLED_ADMIN_TOOLS

print(json.dumps({
    "advertised": sorted(t.name for t in asyncio.run(app.list_tools())),
    "inventory": sorted(tools.__all__),
    "enabled": sorted(ENABLED_ADMIN_TOOLS),
}))
"""


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _tool_surface(tmp_path: Path, flag: str | None) -> dict[str, list[str]]:
    """Import the server in a child process and report what it advertises."""
    env = dict(os.environ)
    env.pop("GRAPHKNOWS_ENABLE_ADMIN_TOOLS", None)
    if flag is not None:
        env["GRAPHKNOWS_ENABLE_ADMIN_TOOLS"] = flag
    # cwd is scratch so no ``.env`` is picked up; PYTHONPATH keeps the child on
    # this checkout rather than any installed copy.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH", "")]))
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.splitlines()[-1])


@pytest.mark.asyncio
async def test_no_tool_accepts_namespace() -> None:
    """The namespace is bound by the process, so no inputSchema exposes it."""
    from graphknows.server.mcp._app import app

    offenders = [
        tool.name
        for tool in await app.list_tools()
        if "namespace" in tool.inputSchema.get("properties", {})
    ]
    assert offenders == []


def test_destructive_tools_absent_without_flag(tmp_path: Path) -> None:
    """Unset flag: the destructive tools are not advertised and not in the inventory."""
    surface = _tool_surface(tmp_path, flag=None)

    assert DESTRUCTIVE.isdisjoint(surface["advertised"])
    assert DESTRUCTIVE.isdisjoint(surface["inventory"])
    assert surface["enabled"] == []
    # The rest of the surface is untouched by the gate.
    assert "memory_query" in surface["advertised"]
    assert "memory_doctor" in surface["advertised"]


def test_destructive_tools_absent_when_flag_is_false(tmp_path: Path) -> None:
    """An explicit falsy value is as shut as an unset one."""
    surface = _tool_surface(tmp_path, flag="false")

    assert DESTRUCTIVE.isdisjoint(surface["advertised"])
    assert surface["enabled"] == []


def test_destructive_tools_advertised_with_flag(tmp_path: Path) -> None:
    """With the flag on the gate opens, and the inventory says so."""
    surface = _tool_surface(tmp_path, flag="1")

    assert DESTRUCTIVE.issubset(surface["advertised"])
    assert DESTRUCTIVE.issubset(surface["inventory"])
    assert set(surface["enabled"]) == DESTRUCTIVE


def test_enabled_admin_tools_are_named_at_startup(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An enabled destructive surface is announced, never silently present."""
    from graphknows.server.mcp import stdio_server

    with caplog.at_level("WARNING", logger="graphknows.startup"):
        monkeypatch.setattr(stdio_server, "ENABLED_ADMIN_TOOLS", ())
        stdio_server._announce_admin_tools()
        assert caplog.text == ""

        monkeypatch.setattr(stdio_server, "ENABLED_ADMIN_TOOLS", tuple(sorted(DESTRUCTIVE)))
        stdio_server._announce_admin_tools()

    assert "memory_drop_namespace" in caplog.text
    assert "memory_purge" in caplog.text
