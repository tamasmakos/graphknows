"""Async JSON-RPC stdio client for the graphknows MCP server.

Stdlib-only (asyncio + json) so it stays in the core install: any out-of-process
consumer can drive ``graphknows-mcp`` without pulling the ML/extraction extras.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from graphknows.exceptions import GraphKnowsError

log = logging.getLogger(__name__)


class MCPClientError(GraphKnowsError):
    """Raised when the MCP server returns a JSON-RPC error or malformed response."""


@dataclass
class GraphKnowsMCPClient:
    """Async JSON-RPC client for ``graphknows-mcp`` over stdio."""

    command: Sequence[str] | str = "graphknows-mcp"
    # One memory_ingest call runs the whole extraction stack for a document, and
    # the FIRST one also pays the extraction model's cold start (~100s on CPU for
    # the relex model). At the old 120s the first ingest raced its own model load
    # and lost: documents failed silently, leaving a partially-populated graph
    # (38 of 63 chunks, no relations at all) that still answered questions and so
    # looked like a valid run. Generous by default; the retry/abort behaviour that
    # matters lives at the call sites, not here.
    request_timeout_s: float = float(os.environ.get("GRAPHKNOWS_MCP_TIMEOUT_S", "900"))
    env: dict[str, str] = field(default_factory=dict)
    initialize: bool = True

    _proc: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)
    _request_id: int = field(default=0, init=False)
    _pending: dict[int, asyncio.Future[dict[str, Any]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _reader_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> GraphKnowsMCPClient:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        """Start the MCP subprocess and initialize the protocol if requested."""
        if self._proc is not None:
            return
        args = shlex.split(self.command) if isinstance(self.command, str) else list(self.command)

        # Safety for Docker: ensure --rm is present for 'docker run' to avoid container leaks,
        # and prefer 'docker exec' if targeting the singleton container.
        if len(args) > 1 and args[0] == "docker":
            if args[1] == "run" and "--rm" not in args:
                args.insert(2, "--rm")
            elif args[1] == "exec" and "-i" not in args and "-it" not in args:
                args.insert(2, "-i")

        env = {**os.environ, **self.env}
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=4 * 1024 * 1024,  # 4 MB — large responses (top_k=20) exceed the 64KB default
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        if self.initialize:
            await self.request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "graphknows-client", "version": "1.0.0"},
                },
            )
            await self.notify("notifications/initialized", {})

    async def close(self) -> None:
        """Terminate the MCP subprocess."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                # Swallow the reader task's own cancellation (we cancelled it),
                # but re-raise if *this* close() coroutine is being cancelled.
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
            except Exception as exc:
                # Teardown is best-effort, but an unlogged swallow here hides a
                # reader task that died earlier for an unrelated reason.
                log.debug("MCP reader task raised during close: %s", exc)
            self._reader_task = None
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPClientError("MCP client closed"))
        self._pending.clear()

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

    async def _read_loop(self) -> None:
        """Background task: own proc.stdout and dispatch responses to waiting futures."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        while True:
            line = await proc.stdout.readline()
            if not line:
                stderr = await self._read_stderr_tail(proc)
                err = MCPClientError(f"MCP server closed stdout. stderr: {stderr}")
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(err)
                self._pending.clear()
                return
            try:
                response = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue  # malformed line — skip, don't kill the reader
            rid = response.get("id")
            if rid is None:
                continue  # notification or response without id
            pending = self._pending.get(rid)
            if pending is None:
                continue  # no one waiting — ignore
            if not pending.done():
                pending.set_result(response)

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC request and return the response result."""
        proc = self._require_proc()
        self._request_id += 1
        request_id = self._request_id
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = fut
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        try:
            self._write_payload(proc, payload)
            response = await asyncio.wait_for(fut, timeout=self.request_timeout_s)
        except TimeoutError:
            self._pending.pop(request_id, None)
            raise
        except BaseException:
            self._pending.pop(request_id, None)
            raise
        else:
            self._pending.pop(request_id, None)

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
