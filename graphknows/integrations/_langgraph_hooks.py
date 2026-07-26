"""Framework-agnostic memory node helpers.

``recall`` and ``remember`` operate on a plain state ``dict`` and a
graphknows memory (in-process :class:`graphknows.Memory` or out-of-process
:class:`graphknows.integrations.client.RemoteMemory` — both expose the same verbs). They are
used as LangGraph nodes via ``functools.partial(recall, memory=mem)`` but depend
on nothing from langgraph, so the same helpers drop into any graph framework
whose nodes are ``state -> partial_state`` callables.
"""

from __future__ import annotations

from typing import Any, Protocol


class _Memory(Protocol):
    async def ingest_memory(
        self,
        messages: str | list[dict[str, Any]] | dict[str, Any] = ...,
        *,
        session_id: str = ...,
    ) -> dict[str, Any]: ...
    async def recall_memory(
        self,
        query: str,
        *,
        session_id: str = ...,
        top_k: int = ...,
        scope: str = ...,
        cross_session: bool = ...,
    ) -> dict[str, Any]: ...


def _message_timestamp(message: Any) -> str | None:
    """Return a message's timestamp, from a dict key or an attribute."""
    ts = getattr(message, "timestamp", None)
    if ts is None and isinstance(message, dict):
        ts = message.get("timestamp")
    return str(ts) if ts else None


def _last_message(state: dict[str, Any], messages_key: str) -> Any | None:
    """Return the last entry of a LangGraph-style messages list, or ``None``."""
    messages = state.get(messages_key) or []
    return messages[-1] if messages else None


def _message_text(message: Any) -> str:
    """Return one message's text content.

    Handles both plain-string content and LangChain multimodal content (a list
    of blocks, where text blocks are ``{"type": "text", "text": ...}``).
    """
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return " ".join(p for p in parts if p)
    return str(content or "")


# graphknows' Role is a four-value enum, but the labels that reach a node come
# from whatever the caller's framework uses. LangChain's own message objects
# report ``type`` as "human"/"ai", neither of which is a Role — passing them
# through raises a pydantic ValidationError on the first real LangChain state.
_ROLE_ALIASES = {"human": "user", "ai": "assistant", "agent": "assistant", "bot": "assistant"}
_ROLES = {"user", "assistant", "system", "tool"}


def _message_role(message: Any) -> tuple[str, str | None]:
    """Return one message's ``(role, speaker_name)``.

    A multi-party conversation labels turns with participant names ("Gina",
    "Jon"), which no four-value enum can hold. Rather than reject the turn or
    silently discard the label, an unrecognised label is recorded as the
    speaker ``name`` and the turn is stored as a ``user`` turn — the identity
    survives on the field meant for it instead of being lost in a coercion.
    """
    raw = getattr(message, "type", None)
    if raw is None and isinstance(message, dict):
        raw = message.get("role")
    if not raw:
        return "user", None
    label = str(raw)
    if label in _ROLES:
        return label, None
    if (mapped := _ROLE_ALIASES.get(label.lower())) is not None:
        return mapped, None
    return "user", label


def _last_text(state: dict[str, Any], messages_key: str) -> str:
    """Return the last message's text from a LangGraph-style messages list."""
    last = _last_message(state, messages_key)
    return _message_text(last) if last is not None else ""


async def recall(
    state: dict[str, Any],
    *,
    memory: _Memory,
    session_id: str,
    messages_key: str = "messages",
    top_k: int = 5,
    scope: str = "both",
    out_key: str = "memories",
    cross_session: bool = True,
) -> dict[str, Any]:
    """Recall memories for the latest user turn and return them as a state update.

    Reads the last message in ``state[messages_key]`` as the query; writes the
    ranked hit texts to ``state[out_key]`` and each hit's retrieval channel
    (e.g. ``"stm"``/``"vector"``/``"entity"``) to ``state[out_key + "_sources"]``,
    so a caller can see honestly which channel answered. Returns only the
    partial update.

    ``cross_session`` defaults to True — an agent's memory is expected to span
    the conversations it has already had, so the graph half of the search is
    namespace-wide. Set it False to confine recall to this session's own graph.
    The unflushed short-term buffer is session-scoped either way.
    """
    query = _last_text(state, messages_key)
    sources_key = f"{out_key}_sources"
    if not query:
        return {out_key: [], sources_key: []}
    result = await memory.recall_memory(
        query,
        session_id=session_id,
        top_k=top_k,
        scope=scope,
        cross_session=cross_session,
    )
    raw_hits = result.get("hits", []) if isinstance(result, dict) else []
    hits = [h for h in raw_hits if isinstance(h, dict)]
    return {
        out_key: [h.get("text", "") for h in hits],
        sources_key: [h.get("sources", "") for h in hits],
    }


async def remember(
    state: dict[str, Any],
    *,
    memory: _Memory,
    session_id: str,
    messages_key: str = "messages",
) -> dict[str, Any]:
    """Persist the latest turn into memory. Returns an empty state update.

    Ingests the last message in ``state[messages_key]`` as a one-message
    conversation (``[{"role": ..., "content": ...}]``), not a bare string —
    a bare string takes the immediate-extraction document path instead of the
    short-term turn buffer. A no-op when there is no message or no text.
    """
    last = _last_message(state, messages_key)
    if last is None:
        return {}
    text = _message_text(last)
    if text:
        role, name = _message_role(last)
        turn: dict[str, Any] = {"role": role, "content": text}
        if name:
            turn["name"] = name
        # A turn's timestamp is the anchor its relative dates ("next Friday")
        # resolve against. Without it they resolve against ingestion time, which
        # silently dates the whole conversation to whenever it was imported.
        if (ts := _message_timestamp(last)) is not None:
            turn["timestamp"] = ts
        await memory.ingest_memory([turn], session_id=session_id)
    return {}
