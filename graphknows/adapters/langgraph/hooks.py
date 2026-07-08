"""Framework-agnostic memory node helpers.

``recall`` and ``remember`` operate on a plain state ``dict`` and a
graphknows memory (in-process :class:`graphknows.Memory` or out-of-process
:class:`graphknows.client.RemoteMemory` — both expose the same verbs). They are
used as LangGraph nodes via ``functools.partial(recall, memory=mem)`` but depend
on nothing from langgraph, so the same helpers drop into any graph framework
whose nodes are ``state -> partial_state`` callables.
"""

from __future__ import annotations

from typing import Any, Protocol


class _Memory(Protocol):
    async def ingest_memory(self, text: str, *, session_id: str, **kw: Any) -> dict[str, Any]: ...
    async def recall_memory(
        self, query: str, *, session_id: str, top_k: int, scope: str
    ) -> dict[str, Any]: ...


def _last_text(state: dict[str, Any], messages_key: str) -> str:
    """Return the last message's text from a LangGraph-style messages list.

    Handles both plain-string content and LangChain multimodal content (a list
    of blocks, where text blocks are ``{"type": "text", "text": ...}``).
    """
    messages = state.get(messages_key) or []
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is None and isinstance(last, dict):
        content = last.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return " ".join(p for p in parts if p)
    return str(content or "")


async def recall(
    state: dict[str, Any],
    *,
    memory: _Memory,
    session_id: str,
    messages_key: str = "messages",
    top_k: int = 5,
    scope: str = "both",
    out_key: str = "memories",
) -> dict[str, Any]:
    """Recall memories for the latest user turn and return them as a state update.

    Reads the last message in ``state[messages_key]`` as the query; writes the
    ranked hit texts to ``state[out_key]``. Returns only the partial update.
    """
    query = _last_text(state, messages_key)
    if not query:
        return {out_key: []}
    result = await memory.recall_memory(query, session_id=session_id, top_k=top_k, scope=scope)
    hits = result.get("hits", []) if isinstance(result, dict) else []
    return {out_key: [h.get("text", "") for h in hits if isinstance(h, dict)]}


async def remember(
    state: dict[str, Any],
    *,
    memory: _Memory,
    session_id: str,
    messages_key: str = "messages",
) -> dict[str, Any]:
    """Persist the latest turn into memory. Returns an empty state update.

    Ingests the last message in ``state[messages_key]``; a no-op when empty.
    """
    text = _last_text(state, messages_key)
    if text:
        await memory.ingest_memory(text, session_id=session_id)
    return {}
