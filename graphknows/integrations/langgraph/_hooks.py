"""Framework-agnostic memory node helpers.

``recall`` and ``remember`` operate on a plain state ``dict`` and an in-process
:class:`graphknows.Memory`. They are
used as LangGraph nodes via ``functools.partial(recall, memory=mem)`` but depend
on nothing from langgraph, so the same helpers drop into any graph framework
whose nodes are ``state -> partial_state`` callables.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from graphknows.models.hit import Hit, render_memories

logger = logging.getLogger(__name__)


class _Memory(Protocol):
    async def ingest_memory(
        self,
        messages: str | list[dict[str, Any]] | dict[str, Any] = ...,
        *,
        session_id: str = ...,
        infer: bool = ...,
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


# A TURN's role is one of four names, but the labels that reach a node come
# from whatever the caller's framework uses. LangChain's own message objects
# report ``type`` as "human"/"ai", neither of which is one of them — storing
# them raw would split the same speaker across two role values.
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


def resolve_infer(infer: bool | Callable[[str], bool], text: str) -> bool:
    """Apply an extraction policy to one turn's text.

    ``infer`` is either a flat answer or a predicate asked per turn, so a
    caller can decide per turn whether it mints entities. Shared by every
    write path so the two cannot drift.
    """
    return bool(infer(text)) if callable(infer) else bool(infer)


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

    Two more keys carry what the flat text list cannot: ``_hits`` holds the
    :class:`~graphknows.models.hit.Hit` objects (``speaker``, ``ts``, ``score``,
    ``metadata`` …) so a caller can attribute and date a memory, and ``_facts``
    holds the structured fact-sheet. Both were previously computed and discarded
    here, leaving every agent built on this node to answer from undated,
    unattributed sentences.

    ``_context`` is the one most callers want: the hits already run through
    :func:`~graphknows.models.hit.render_memories` — deduped, dated, attributed,
    chronological, budgeted — as a string that drops straight into a prompt.
    Handing back only the parts meant every integration re-assembled them, and
    they disagreed about where a date comes from. ``_dated``/``_undated`` are
    that render's own counts: a recall silently falling back to the undated
    branch reads downstream as a mysteriously worse answer with nothing to
    point at.

    ``cross_session`` defaults to True — an agent's memory is expected to span
    the conversations it has already had, so the graph half of the search is
    namespace-wide. Set it False to confine recall to this session's own graph.
    The unflushed short-term buffer is session-scoped either way.
    """
    query = _last_text(state, messages_key)
    sources_key, hits_key, facts_key = (
        f"{out_key}_sources",
        f"{out_key}_hits",
        f"{out_key}_facts",
    )
    context_key, dated_key, undated_key = (
        f"{out_key}_context",
        f"{out_key}_dated",
        f"{out_key}_undated",
    )
    if not query:
        return {
            out_key: [],
            sources_key: [],
            hits_key: [],
            facts_key: [],
            context_key: "",
            dated_key: 0,
            undated_key: 0,
        }
    result = await memory.recall_memory(
        query,
        session_id=session_id,
        top_k=top_k,
        scope=scope,
        cross_session=cross_session,
    )
    raw_hits = result.get("hits", []) if isinstance(result, dict) else []
    hits = [h for h in raw_hits if isinstance(h, Hit)]
    if len(hits) != len(raw_hits):
        # Only Memory mints hits, so a non-Hit here is broken wiring, not a
        # tolerable variation. Dropping it silently renders as "nothing
        # recalled" — indistinguishable from an empty memory from outside.
        logger.warning(
            "recall: dropped %d recall result(s) that were not Hit objects; "
            "the memory passed in is not returning the documented shape.",
            len(raw_hits) - len(hits),
        )
    rendered = render_memories(hits)
    return {
        out_key: [h.text for h in hits],
        sources_key: [h.sources for h in hits],
        hits_key: hits,
        facts_key: list(result.get("facts", []) or []) if isinstance(result, dict) else [],
        context_key: rendered.text,
        dated_key: rendered.dated,
        undated_key: rendered.undated,
    }


async def remember(
    state: dict[str, Any],
    *,
    memory: _Memory,
    session_id: str,
    messages_key: str = "messages",
    infer: bool | Callable[[str], bool] = True,
) -> dict[str, Any]:
    """Persist the latest turn into memory. Returns an empty state update.

    Ingests the last message in ``state[messages_key]`` as a one-message
    conversation (``[{"role": ..., "content": ...}]``), not a bare string —
    a bare string takes the immediate-extraction document path instead of the
    short-term turn buffer. A no-op when there is no message or no text.

    ``infer`` may be a predicate asked per turn (see :func:`resolve_infer`).
    Without it a caller wanting per-turn extraction control had to abandon this
    node and drive ``ingest_memory`` itself.
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
        await memory.ingest_memory([turn], session_id=session_id, infer=resolve_infer(infer, text))
    return {}
