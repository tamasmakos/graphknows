"""MCP tools: web — Memory Mirror link helper."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.web.links import mirror_url


@app.tool()
def memory_mirror(session_id: str = "", scope: str = "ltm") -> dict[str, Any]:
    """Return a local Memory Mirror web UI link.

    Start the UI with ``graphknows-web`` first, then open the returned URL to
    inspect memory graphs and recall results in a lightweight browser view.

    Args:
        session_id: Optional session namespace to preselect in the UI.
        scope: Initial graph scope, either ``"ltm"`` or ``"stm"``.

    Returns:
        Dict with ``url``, ``session_id``, and ``scope`` fields.
    """
    selected_scope = scope if scope in {"stm", "ltm"} else "ltm"
    return {
        "url": mirror_url(session_id=session_id, scope=selected_scope),
        "session_id": session_id,
        "scope": selected_scope,
    }
