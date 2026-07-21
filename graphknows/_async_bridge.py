"""Shared background event-loop bridge for STM async store calls.

A single background thread with its own persistent event loop is created once.
All async store calls are dispatched into this loop — so async clients, which
are bound to the loop they were created on, always see the same loop.
"""

from __future__ import annotations

import asyncio
from threading import Thread
from typing import Any

_BG_LOOP: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_BG_THREAD: Thread = Thread(
    target=_BG_LOOP.run_forever, daemon=True, name="graphknows-stm-async-worker"
)
_BG_THREAD.start()

# Upper bound on a single bridged store call. Generous enough for a slow
# retrieval (embedding + several queries + rerank) but finite: without it a
# wedged store call blocks the calling thread forever with no diagnostic, and
# the caller cannot distinguish "slow" from "never returning".
_CALL_TIMEOUT_S: float = 300.0


def _run(coro: Any, timeout: float = _CALL_TIMEOUT_S) -> Any:
    """Submit *coro* to the persistent background event loop and block until done.

    Safe to call from any thread (including the main asyncio.run thread) because
    it targets a *different* dedicated loop — no nesting conflict.

    Raises:
        TimeoutError: If the call does not finish within *timeout* seconds. The
            coroutine is cancelled so it cannot keep occupying the shared loop.
    """
    future = asyncio.run_coroutine_threadsafe(coro, _BG_LOOP)
    try:
        return future.result(timeout)
    except TimeoutError:
        future.cancel()
        raise
