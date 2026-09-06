"""Stdlib-only stdio JSON-RPC stub server for the client round-trip test.

Speaks the same newline-delimited JSON-RPC framing as ``graphknows-mcp``
(initialize, notifications/initialized, tools/list, tools/call) without
importing anything from graphknows — the real server pulls in the ML/
extraction stack, and ``integrations.client`` must stay usable without it.

A "boom" tool call replies with a JSON-RPC error, and an ``echo`` call with
a ``delay_s`` argument replies from a background thread after that delay —
letting the round-trip test prove replies are matched to callers by
JSON-RPC id, not by the order they arrive on stdout.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

_stdout_lock = threading.Lock()


def _respond(payload: dict[str, Any]) -> None:
    with _stdout_lock:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def _handle(message: dict[str, Any]) -> None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    if request_id is None:
        return  # notification (e.g. notifications/initialized) — no reply

    if method == "initialize":
        _respond(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "fake-mcp-server", "version": "0.0.0"},
                },
            }
        )
    elif method == "tools/list":
        _respond(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": [{"name": "echo", "description": "echoes arguments"}]},
            }
        )
    elif method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name == "boom":
            _respond(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": "boom"},
                }
            )
            return

        def reply() -> None:
            delay_s = arguments.get("delay_s")
            if delay_s:
                time.sleep(float(delay_s))
            _respond(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"structuredContent": {"echoed": arguments}},
                }
            )

        threading.Thread(target=reply, daemon=True).start()
    else:
        _respond(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"unknown method {method}"},
            }
        )


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        _handle(message)


if __name__ == "__main__":
    main()
