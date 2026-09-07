"""``memory_query`` must be an adapter over the facade, not a second retriever.

It used to drive the retriever directly, which made the MCP transport disagree
with every in-process caller about three things at once: hits carried no
speaker and no timestamp, ``top_k`` was silently capped at 100, and
``scope="stm"`` searched raw chunks instead of the buffered TURN rows. The
class docstring promised "every transport sees identical behaviour"; on the
read path it was not true.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from graphknows.exceptions import ConfigurationError
from graphknows.models.hit import Hit
from graphknows.server.mcp.tools.query import memory_query


class _FakeRuntime:
    """Records the recall it was asked for; returns typed hits.

    Mirrors the facade's own FR-030 bound (top_k above 100 raises) so a test
    here can assert the tool does not intercept or reinterpret that — only
    forward it, the same way it forwards ``scope``.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def recall_memory(
        self,
        query: str,
        *,
        session_id: str = "",
        top_k: int = 5,
        scope: str = "stm",
        cross_session: bool = False,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "query": query,
                "session_id": session_id,
                "top_k": top_k,
                "scope": scope,
                "cross_session": cross_session,
            }
        )
        if top_k > 100:
            raise ConfigurationError(f"top_k exceeds 100 (got {top_k})")
        return {
            "hits": [
                Hit(
                    text="Mochi is a ragdoll",
                    speaker="Gina",
                    ts="20 January, 2023",
                    score=0.42,
                    sources="vector",
                    chunk_id="c1",
                    doc_id="d1",
                )
            ],
            "facts": ["Mochi -[IS_A]-> ragdoll"],
            # A sentinel, not a plausible scope: the tool must report what the
            # facade says it searched, never re-derive it.
            "scope": "scope-as-the-facade-resolved-it",
        }


@pytest.fixture
def runtime() -> _FakeRuntime:
    fake = _FakeRuntime()

    async def _get_runtime(namespace: str = "") -> _FakeRuntime:
        return fake

    with patch("graphknows.server.mcp.tools.query.get_runtime", _get_runtime):
        yield fake


@pytest.mark.asyncio
async def test_query_routes_through_the_facade(runtime: _FakeRuntime) -> None:
    await memory_query("what breed is my cat?", session_id="s1", top_k=7, scope="ltm")

    assert runtime.calls == [
        {
            "query": "what breed is my cat?",
            "session_id": "s1",
            "top_k": 7,
            "scope": "ltm",
            "cross_session": False,
        }
    ]


@pytest.mark.asyncio
async def test_hits_carry_the_speaker_and_timestamp(runtime: _FakeRuntime) -> None:
    """The MCP shape dropped both, so every consumer downstream re-derived them."""
    out = await memory_query("cat", session_id="s1")

    assert out["hits"] == [
        {
            "text": "Mochi is a ragdoll",
            "speaker": "Gina",
            "ts": "20 January, 2023",
            "score": 0.42,
            "sources": "vector",
            "session_id": "",
            "chunk_id": "c1",
            "doc_id": "d1",
            "entities": [],
            "metadata": {},
        }
    ]


@pytest.mark.asyncio
async def test_top_k_over_the_bound_is_rejected_not_capped(runtime: _FakeRuntime) -> None:
    """FR-030 replaces the old "no ceiling" contract with a rejection, not a clamp.

    This rewrites ``test_top_k_is_not_capped`` (the single C-001 exception in
    launch-remediation): it used to assert that top_k=369 flowed through
    unchanged, evidence for a recall-vs-depth sweep on conv-30 where an old
    cap of 100 plateaued evidence_recall at 0.827 — read as gold chunks being
    unreachable — and lifting it reached 1.000 at k=369. FR-030 reinstates a
    ceiling at the trust boundary for resourcing reasons (top_k multiplies
    into the retriever's pool_k and the store's k_pre = top_k * 20), but as a
    rejection rather than the old silent clamp. The tool is a thin adapter:
    it must not swallow or reinterpret that rejection, only forward top_k
    untouched and let it propagate.
    """
    with pytest.raises(ConfigurationError):
        await memory_query("cat", top_k=369)

    assert runtime.calls[0]["top_k"] == 369


@pytest.mark.asyncio
async def test_scope_is_resolved_by_the_facade_not_here(runtime: _FakeRuntime) -> None:
    """An unknown scope falls back inside the facade, and it says so.

    The tool forwards the caller's scope untouched and reports back whatever
    the facade resolved it to. A second copy of the fallback rule here is how
    the two surfaces came to disagree about what ``"stm"`` even means.
    """
    out = await memory_query("cat", scope="nonsense")

    assert runtime.calls[0]["scope"] == "nonsense"
    assert out["scope"] == "scope-as-the-facade-resolved-it"


@pytest.mark.asyncio
async def test_response_reports_what_it_searched(runtime: _FakeRuntime) -> None:
    out = await memory_query("cat", scope="both")

    assert out["total"] == 1
    assert out["query"] == "cat"
    assert out["facts"] == ["Mochi -[IS_A]-> ragdoll"]
