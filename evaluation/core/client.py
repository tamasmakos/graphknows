"""Small JSON-RPC stdio client for the graphknows MCP server."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


class MCPClientError(RuntimeError):
    """Raised when the MCP server returns a JSON-RPC error or malformed response."""


@dataclass
class GraphknowsMCPClient:
    """Async JSON-RPC client for graphknows-mcp over stdio."""

    command: Sequence[str] | str = "graphknows-mcp"
    request_timeout_s: float = 120.0
    env: dict[str, str] = field(default_factory=dict)
    initialize: bool = True

    _proc: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)
    _request_id: int = field(default=0, init=False)

    async def __aenter__(self) -> "GraphknowsMCPClient":
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        """Start the MCP subprocess and initialize the protocol if requested."""
        if self._proc is not None:
            return
        args = shlex.split(self.command) if isinstance(self.command, str) else list(self.command)
        env = {**os.environ, **self.env}
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        if self.initialize:
            await self.request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "graphknows-evaluation", "version": "0.1.0"},
                },
            )
            await self.notify("notifications/initialized", {})

    async def close(self) -> None:
        """Terminate the MCP subprocess."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.stdin:
            proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            proc.terminate()
            await proc.wait()

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification."""
        proc = self._require_proc()
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._write_payload(proc, payload)

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC request and return the response result."""
        proc = self._require_proc()
        self._request_id += 1
        request_id = self._request_id
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        self._write_payload(proc, payload)

        while True:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=self.request_timeout_s)
            if not line:
                stderr = await self._read_stderr_tail(proc)
                raise MCPClientError(f"MCP server closed stdout. stderr: {stderr}")
            response = json.loads(line.decode("utf-8"))
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise MCPClientError(json.dumps(response["error"], sort_keys=True))
            return response.get("result")

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool metadata."""
        result = await self.request("tools/list", {})
        tools = result.get("tools", []) if isinstance(result, dict) else []
        return list(tools)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call one MCP tool and normalize FastMCP structured/text results."""
        result = await self.request("tools/call", {"name": name, "arguments": arguments or {}})
        return normalize_tool_result(result)

    def _require_proc(self) -> asyncio.subprocess.Process:
        """Return the active subprocess or raise if the client is not started."""
        if self._proc is None:
            raise MCPClientError("MCP client is not started")
        if self._proc.stdin is None or self._proc.stdout is None:
            raise MCPClientError("MCP subprocess streams are unavailable")
        return self._proc

    @staticmethod
    def _write_payload(proc: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
        """Write one JSON-RPC payload to subprocess stdin."""
        if proc.stdin is None:
            raise MCPClientError("MCP subprocess stdin is closed")
        proc.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")

    @staticmethod
    async def _read_stderr_tail(proc: asyncio.subprocess.Process) -> str:
        """Read a small stderr tail without blocking forever."""
        if proc.stderr is None:
            return ""
        try:
            data = await asyncio.wait_for(proc.stderr.read(4096), timeout=0.2)
        except TimeoutError:
            return ""
        return data.decode("utf-8", errors="replace")


def normalize_tool_result(result: Any) -> Any:
    """Normalize common MCP tool result wrappers into the underlying payload."""
    if not isinstance(result, dict):
        return result
    if "structuredContent" in result and result["structuredContent"] is not None:
        return result["structuredContent"]
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and first.get("type") == "text":
            text = str(first.get("text", ""))
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
    return result
