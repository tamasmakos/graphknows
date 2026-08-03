"""A fake ``Memory`` for tests that cross the integration seam.

The seam is narrow on purpose — ``ingest_memory`` and ``recall_memory`` are the
only two methods the LangGraph hooks, ``GraphKnowsMemory`` and
``GraphKnowsStore`` depend on — so one fake serves all three. It lived
duplicated in two test modules before, which is how the two copies came to
return different hits for the same call.

``recall_memory`` returns :class:`~graphknows.models.hit.Hit` objects, exactly
as the real facade does: a fake that returned dicts would let a caller that
subscripts hits pass here and fail against a live ``Memory``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from graphknows.models.hit import Hit

HITS = [
    Hit(text="Mochi is a ragdoll", sources="stm", speaker="Gina", ts="20 January, 2023"),
    Hit(text="Dr. Nagy is in Budapest", sources="vector", speaker="Jon", ts="1 February, 2023"),
]
FACTS = ["Mochi -[IS_A]-> ragdoll"]


class FakeMemory:
    """Records what it was asked for; returns a fixed, typed result."""

    def __init__(self) -> None:
        self.ingested: list[tuple[str, Any]] = []
        self.ingest_kwargs: list[dict[str, Any]] = []
        self.recall_calls: list[dict[str, Any]] = []
        self.flushed: list[str] = []
        self.close = AsyncMock()

    async def flush(self, *, session_id: str = "", **kw: Any) -> dict[str, Any]:
        self.flushed.append(session_id)
        return {"turns": 0, "errors": []}

    async def stats(self, session_id: str = "") -> dict[str, Any]:
        return {"entities": 0, "relations": 0, "topics": 0}

    async def ingest_memory(
        self, messages: Any, *, session_id: str = "", **kw: Any
    ) -> dict[str, Any]:
        self.ingested.append((session_id, messages))
        self.ingest_kwargs.append(kw)
        return {"chunks": 1}

    async def recall_memory(
        self,
        query: str,
        *,
        session_id: str = "",
        top_k: int = 5,
        scope: str = "both",
        cross_session: bool = False,
    ) -> dict[str, Any]:
        self.recall_calls.append(
            {
                "query": query,
                "session_id": session_id,
                "top_k": top_k,
                "scope": scope,
                "cross_session": cross_session,
            }
        )
        return {"hits": list(HITS), "facts": list(FACTS)}


__all__ = ["FACTS", "HITS", "FakeMemory"]
