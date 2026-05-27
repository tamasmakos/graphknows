"""Shared background event-loop bridge for STM async store calls.

A single background thread with its own persistent event loop is created once.
All async store calls are dispatched into this loop — so asyncpg pools, which
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


def _run(coro: Any) -> Any:
    """Submit *coro* to the persistent background event loop and block until done.

    Safe to call from any thread (including the main asyncio.run thread) because
    it targets a *different* dedicated loop — no nesting conflict.
    """
    future = asyncio.run_coroutine_threadsafe(coro, _BG_LOOP)
    return future.result()
