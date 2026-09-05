"""The re-drive: marked chunks decode again, and only those (US2, Scenario 2.3).

``Memory.redecode(session_id)`` exists because a provider outage leaves a graph
that is *thin*, not broken — the chunks are stored, embedded and marked, and the
labels are the only thing missing. Re-ingesting the source to get them back
would re-cut the chunks and mint new ids; the re-drive reads the text already in
the graph instead, writes through the same ``_write_extracted`` funnel, and
clears the marker only on a decode that answered.

What is asserted here:

* only ``decode_state='failed'`` chunks reach the decoder — an ``'ok'`` chunk's
  text is never in an outgoing request;
* a chunk that decodes becomes ``'ok'``, one that fails again stays ``'failed'``;
* the second run has nothing left to do: ``attempted: 0``, no provider call, and
  a store identical to the one the first run left.

The provider is a stub object; nothing here reaches a network, a local model or
a real ``time.sleep``.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from graphknows.ingestion.extraction.llm import retry as retry_module
from graphknows.ingestion.extraction.llm.decoder import LLMDecoder
from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.memory import Memory
from graphknows.settings import GraphKnowsSettings

pytestmark = pytest.mark.unit

MODEL = "openrouter/stub/decoder-v1"
FULL_REPLY = json.dumps({"entities": [{"surface": "Acme", "label": "organization"}]})

#: The word that makes the stub provider refuse — the outage that never lifted.
POISON = "unreachable"


class _Provider:
    """The smallest thing ``LLMDecoder`` calls, refusing any request naming POISON."""

    model = MODEL

    def __init__(self) -> None:
        self.requests: list[str] = []

    def __call__(self, **kwargs: Any) -> list[str]:
        request = json.dumps(kwargs.get("messages", []))
        self.requests.append(request)
        if POISON in request:
            raise litellm.RateLimitError(message="slow down", llm_provider="stub", model=MODEL)
        return [FULL_REPLY]


class _Store:
    """The CHUNK rows a re-drive touches, held in a dict instead of ArcadeDB.

    ``__getattr__`` answers every other store call with an async no-op, so the
    funnel's ontology/frame/temporal writes cost this test nothing to stub.
    """

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = {c["chunk_id"]: dict(c) for c in chunks}

    def __getattr__(self, _name: str) -> Any:
        async def _noop(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _noop

    async def failed_decode_chunks(self, session_id: str) -> list[dict[str, Any]]:
        return [
            {k: c[k] for k in ("chunk_id", "session_id", "text", "speaker", "ts", "seq")}
            for c in self.chunks.values()
            if c["session_id"] == session_id and c["decode_state"] == "failed"
        ]

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {cid: dict(self.chunks[cid]) for cid in chunk_ids if cid in self.chunks}

    async def write_chunk(self, chunk: dict[str, Any]) -> None:
        self.chunks[chunk["chunk_id"]].update(chunk)


def _chunk(chunk_id: str, text: str, decode_state: str, **overrides: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "session_id": "s1",
        "text": text,
        "speaker": "",
        "ts": "",
        "seq": 0,
        "kind": "text",
        "memory_class": "raw_evidence",
        "decode_state": decode_state,
        **overrides,
    }


@pytest.fixture(autouse=True)
def _decoder_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real waiting, and no lexical label selection.

    Both for the reason ``test_decoder_failure`` gives: the backoff is asserted
    where it belongs, and whether a re-drive re-decodes the right chunks is not
    a question that should need a language model installed to answer.
    """
    monkeypatch.setattr(retry_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("GRAPHKNOWS_LEXICAL_LABELS", "false")


def _handler(store: _Store, provider: _Provider) -> _STMIngestHandler:
    decoder = LLMDecoder(GraphKnowsSettings(), lm=provider)
    # The relation vocabulary is selected from spaCy's observation of the chunk;
    # a re-drive is not the parser's business.
    decoder.parse = lambda text: SimpleNamespace(ents=())  # type: ignore[method-assign]
    return _STMIngestHandler(store, extractor=decoder, embed_fn=lambda t: [0.1, 0.2])


def _states(store: _Store) -> dict[str, str]:
    return {cid: c["decode_state"] for cid, c in store.chunks.items()}


@pytest.mark.asyncio
async def test_only_the_marked_chunks_are_redecoded() -> None:
    """Scenario 2.3: the marker is the work list, and success clears it."""
    store = _Store(
        [
            _chunk("c1", "Melanie works for Acme.", "failed"),
            _chunk("c2", f"Acme opened a studio, {POISON}.", "failed"),
            _chunk("c3", "Acme hired a dancer.", "ok"),
        ]
    )
    provider = _Provider()

    report = await _handler(store, provider).redecode("s1")

    assert report["session_id"] == "s1"
    assert (report["attempted"], report["recovered"], report["still_failed"]) == (2, 1, 1)
    # The one that answered lost its marker; the one that failed again kept it,
    # and the chunk that never failed was not touched at all.
    assert _states(store) == {"c1": "ok", "c2": "failed", "c3": "ok"}
    # A chunk-level failure is the chunk's, not the run's — and it is counted.
    assert report["abstentions"]["decoder_failed"] == 1
    assert report["errors"] == []
    # "Only the marked chunks are decoded", read off the wire rather than
    # inferred from the counts.
    assert not any("dancer" in request for request in provider.requests)


@pytest.mark.asyncio
async def test_the_chunk_is_rewritten_in_place_from_the_text_in_the_graph() -> None:
    """No re-ingest: same id, same text, and the row's own kind survives."""
    store = _Store([_chunk("t1", "Melanie works for Acme.", "failed", kind="turn", seq=4)])

    await _handler(store, _Provider()).redecode("s1")

    assert list(store.chunks) == ["t1"]
    row = store.chunks["t1"]
    assert (row["text"], row["kind"], row["seq"]) == ("Melanie works for Acme.", "turn", 4)
    assert row["memory_class"] == "raw_evidence"


@pytest.mark.asyncio
async def test_a_second_run_has_nothing_left_to_do() -> None:
    """Idempotent: running it twice changes nothing further (FR-024)."""
    store = _Store(
        [
            _chunk("c1", "Melanie works for Acme.", "failed"),
            _chunk("c2", "Acme opened a studio.", "failed", seq=1),
        ]
    )
    provider = _Provider()
    handler = _handler(store, provider)

    first = await handler.redecode("s1")
    settled = copy.deepcopy(store.chunks)
    calls_after_first = len(provider.requests)

    second = await handler.redecode("s1")

    assert (first["attempted"], first["recovered"]) == (2, 2)
    assert (second["attempted"], second["recovered"], second["still_failed"]) == (0, 0, 0)
    assert store.chunks == settled
    assert len(provider.requests) == calls_after_first


@pytest.mark.asyncio
async def test_memory_redecode_drives_one_session_and_closes_the_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public surface: ``Memory.redecode`` is the named operation (FR-024)."""
    calls: list[str] = []
    report = {
        "session_id": "s1",
        "attempted": 1,
        "recovered": 1,
        "still_failed": 0,
        "abstentions": {},
        "errors": [],
    }

    class _Service:
        async def connect(self) -> None:
            calls.append("connect")

        async def redecode(self, session_id: str) -> dict[str, Any]:
            calls.append(f"redecode:{session_id}")
            return report

        async def close(self) -> None:
            calls.append("close")

    import graphknows.ingestion as ingestion

    monkeypatch.setattr(ingestion, "build_stm_service", lambda settings, namespace: _Service())

    assert await Memory(GraphKnowsSettings()).redecode("s1") == report
    assert calls == ["connect", "redecode:s1", "close"]
