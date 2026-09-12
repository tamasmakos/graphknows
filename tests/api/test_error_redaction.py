"""Store errors are redacted for MCP callers but kept whole in the log (FR-031).

``memory_doctor``, ``memory_purge`` and ``flush`` used to return
ArcadeDB's raw error body — including the request host — straight to an MCP
caller. Each now returns a generic message carrying a correlation id, while
the full detail (host, body) lands in a ``logger.error`` call tagged with
that same id, so redacting for the caller does not blind the operator
(Constitution V).

``doctor`` and ``purge_memory`` are driven against port 1, which refuses the
connection immediately (no service listens there), so they stay hermetic and
fast — the same trick ``tests/api/test_public_errors.py`` already relies on.
``flush`` goes through the embedder before it ever reaches the store,
so it is driven with a stubbed ``_get_retriever`` instead — the same seam
``tests/memory/services/test_flush_step_isolation.py`` uses for its own
connect-failure case — carrying a ``StoreError`` shaped like the real one.
"""

from __future__ import annotations

import logging
import re
from unittest.mock import AsyncMock

import pytest

from graphknows.exceptions import StoreError
from graphknows.memory import Memory
from graphknows.settings import GraphKnowsSettings

_UNREACHABLE_HOST = "127.0.0.1:1"
_UNREACHABLE_URL = f"http://{_UNREACHABLE_HOST}"
_CORRELATION_RE = re.compile(r"correlation_id=([0-9a-f]{32})")


def _settings() -> GraphKnowsSettings:
    return GraphKnowsSettings(arcadedb_url=_UNREACHABLE_URL)


async def test_doctor_redacts_the_host_but_logs_it(caplog: pytest.LogCaptureFixture) -> None:
    memory = Memory(settings=_settings())
    with caplog.at_level(logging.ERROR, logger="graphknows.memory"):
        result = await memory.doctor()

    message = result["arcadedb"]
    assert _UNREACHABLE_HOST not in message
    match = _CORRELATION_RE.search(message)
    assert match, message
    assert match.group(1) in caplog.text
    assert _UNREACHABLE_HOST in caplog.text


async def test_purge_memory_redacts_the_host_but_logs_it(caplog: pytest.LogCaptureFixture) -> None:
    """A session-scoped purge: ``delete_session`` raises on a real failure

    (unlike ``clear_all``, which soft-fails per label), so it reliably
    exercises the redaction path here.
    """
    memory = Memory(settings=_settings())
    with caplog.at_level(logging.ERROR, logger="graphknows.memory"):
        result = await memory.purge_memory(session_id="s1")

    assert result["deleted"] is False
    joined = " ".join(result["errors"])
    assert _UNREACHABLE_HOST not in joined
    match = _CORRELATION_RE.search(joined)
    assert match, joined
    assert match.group(1) in caplog.text
    assert _UNREACHABLE_HOST in caplog.text


async def test_flush_memory_redacts_the_host_but_logs_it(caplog: pytest.LogCaptureFixture) -> None:
    memory = Memory(settings=_settings())
    memory._get_retriever = AsyncMock(  # type: ignore[method-assign]
        side_effect=StoreError(
            f"ArcadeDB request to {_UNREACHABLE_URL}/api/v1/exists/mem failed: "
            "[Errno 111] Connection refused"
        )
    )
    with caplog.at_level(logging.ERROR, logger="graphknows.memory"):
        out = await memory.flush()

    assert any(e.startswith("connect:") for e in out["errors"]), out["errors"]
    joined = " ".join(out["errors"])
    assert _UNREACHABLE_HOST not in joined
    match = _CORRELATION_RE.search(joined)
    assert match, joined
    assert match.group(1) in caplog.text
    assert _UNREACHABLE_HOST in caplog.text
