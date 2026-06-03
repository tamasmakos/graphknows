"""Unit tests for native evaluation harness behavior."""

from __future__ import annotations

import pytest

from evaluation.core.native.harness import (
    NativeConfig,
    NativeEvaluationRun,
    cleanup_eval_sessions,
)


class FakeClient:
    """Small fake MCP client for harness tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        self.calls.append((name, arguments or {}))
        return {"scope": arguments.get("scope", ""), "session_id": arguments.get("session_id", ""), "errors": []}


@pytest.mark.asyncio
async def test_cleanup_refuses_non_eval_sessions() -> None:
    run = NativeEvaluationRun(run_id="eval-test", config=NativeConfig())
    run.eval_sessions.add("prod-session")

    with pytest.raises(ValueError, match="refusing to purge"):
        await cleanup_eval_sessions(FakeClient(), run)


@pytest.mark.asyncio
async def test_cleanup_uses_scoped_memory_purge() -> None:
    client = FakeClient()
    run = NativeEvaluationRun(run_id="eval-test", config=NativeConfig())
    run.eval_sessions.add("eval-test-stm")

    await cleanup_eval_sessions(client, run)

    assert client.calls == [
        ("memory_purge", {"scope": "both", "session_id": "eval-test-stm"})
    ]
