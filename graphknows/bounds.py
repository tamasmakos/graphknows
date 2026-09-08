"""Request-size limits enforced at the trust boundary (FR-030).

Every public entry point — the ``Memory`` facade, ``DETRetriever.retrieve``,
and the MCP tools that call them — rejects an over-bound request with a
:class:`~graphknows.exceptions.ConfigurationError` rather than silently
clamping it: a caller that asked for too much learns that, instead of
getting a quietly truncated answer.
"""

from __future__ import annotations

import json
from typing import Any

from graphknows.exceptions import ConfigurationError

#: ``Message.content`` and document ``text`` (characters).
MAX_TEXT_CHARS = 65_536
#: Each of session_id/user_id/agent_id/run_id (characters).
MAX_SCOPE_ID_CHARS = 128
#: ``metadata``, JSON-serialised (bytes).
MAX_METADATA_BYTES = 16 * 1024
#: ``recall_memory``/``search`` top_k.
MAX_TOP_K = 100
#: ``ltm_entities`` limit.
MAX_LIMIT = 1_000
#: Default server-side wall-clock budget for one call. Overridable per
#: deployment via ``GRAPHKNOWS_MCP_CALL_TIMEOUT_S``: nothing preloads the
#: extraction models, so the first request after a cold start pays their load
#: time and a fixed ceiling would fail it (observed in CI, 2026-09-07).
CALL_TIMEOUT_S = 120.0


def check_text_length(value: str, field: str) -> None:
    """Reject *value* over :data:`MAX_TEXT_CHARS` rather than truncate it."""
    if len(value) > MAX_TEXT_CHARS:
        raise ConfigurationError(f"{field} exceeds {MAX_TEXT_CHARS} characters (got {len(value)})")


def check_scope_id(value: str, field: str) -> None:
    """Reject a scope id (session/user/agent/run) over :data:`MAX_SCOPE_ID_CHARS`."""
    if len(value) > MAX_SCOPE_ID_CHARS:
        raise ConfigurationError(
            f"{field} exceeds {MAX_SCOPE_ID_CHARS} characters (got {len(value)})"
        )


def check_metadata_size(metadata: dict[str, Any] | None) -> None:
    """Reject *metadata* whose JSON encoding exceeds :data:`MAX_METADATA_BYTES`."""
    if not metadata:
        return
    size = len(json.dumps(metadata).encode("utf-8"))
    if size > MAX_METADATA_BYTES:
        raise ConfigurationError(
            f"metadata exceeds {MAX_METADATA_BYTES} bytes serialised (got {size})"
        )


def check_top_k(top_k: int) -> None:
    """Reject a ``top_k`` over :data:`MAX_TOP_K`."""
    if top_k > MAX_TOP_K:
        raise ConfigurationError(f"top_k exceeds {MAX_TOP_K} (got {top_k})")


def check_limit(limit: int) -> None:
    """Reject a ``limit`` over :data:`MAX_LIMIT`."""
    if limit > MAX_LIMIT:
        raise ConfigurationError(f"limit exceeds {MAX_LIMIT} (got {limit})")
