"""The server-side per-call timeout (FR-030).

Lives in the server layer, not in :mod:`graphknows.bounds`, for two reasons.
The architectural one: ``bounds`` sits below ``models``, which may not reach
``settings``, and the budget is a deployment setting. The design one: the
ceiling is a property of the *transport*. An in-process caller owns its own
deadline, and a library call that loads the extraction models on first use
legitimately outruns any fixed budget.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from graphknows.exceptions import ConfigurationError
from graphknows.settings import get_settings

_AsyncFunc = TypeVar("_AsyncFunc", bound=Callable[..., Awaitable[Any]])


def bounded(func: _AsyncFunc) -> _AsyncFunc:
    """Give one MCP tool call a wall-clock budget.

    The budget is read per call, so a deployment raises it with
    ``GRAPHKNOWS_MCP_CALL_TIMEOUT_S`` instead of a code change.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        budget = get_settings().mcp_call_timeout_s
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=budget)
        except TimeoutError as exc:
            raise ConfigurationError(
                f"request exceeded the {budget:.0f}s server-side timeout"
            ) from exc

    return wrapper  # type: ignore[return-value]
