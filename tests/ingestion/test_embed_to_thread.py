"""The embedder never runs on the event loop (FR-021).

``embed_one`` is a blocking HTTP round-trip (a loaded model, when the local
backend is configured). Called straight from a coroutine it stalls the loop for
its whole duration: nothing else on that loop — the concurrent chunk writes, a
server's other requests — makes progress while a chunk is being embedded.

Every embedding call reachable from the async ingest surface is asserted here,
by the only evidence that cannot be faked: the identity of the thread the
embedder was invoked on. A recorder stands in for the embedder and remembers
where it ran; the loop's own thread must not be among the answers.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.stm.ingest import _STMIngestHandler

pytestmark = pytest.mark.unit

VECTOR = [0.1, 0.2]


class _Recorder:
    """An embedder that answers with a constant and remembers its thread."""

    def __init__(self) -> None:
        self.threads: set[int] = set()

    def __call__(self, text: str, *_args: Any, **_kwargs: Any) -> list[float]:
        self.threads.add(threading.get_ident())
        return list(VECTOR)


class _StubExtractor:
    """One entity per chunk — enough for the write funnel to have work."""

    def extract(self, text: str, *_args: Any, **_kwargs: Any) -> ExtractionResult:
        return ExtractionResult(entities=[{"name": "Gina", "type": "PERSON", "score": 0.9}])


@pytest.fixture(autouse=True)
def _no_lexical_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lexical selection wants the shared spaCy pipeline; threads are not its business."""
    monkeypatch.setenv("GRAPHKNOWS_LEXICAL_LABELS", "false")


def _store() -> MagicMock:
    store = MagicMock()
    for name in (
        "ensure_session",
        "ensure_file",
        "write_chunk",
        "write_entities_bulk",
        "write_class_matches",
        "write_instance_of",
        "write_temporal",
    ):
        setattr(store, name, AsyncMock(return_value=None))
    return store


def _handler(embed_fn: Any) -> _STMIngestHandler:
    return _STMIngestHandler(_store(), extractor=_StubExtractor(), embed_fn=embed_fn)


def _assert_off_loop(recorder: _Recorder) -> None:
    assert recorder.threads, "the embedder was never called — the drive proves nothing"
    assert threading.get_ident() not in recorder.threads, (
        "the embedder ran on the event loop thread and blocked it"
    )


@pytest.mark.asyncio
async def test_the_file_path_embeds_off_the_loop() -> None:
    """``ingest_text`` -> the per-chunk call, taken when ``embed_fn`` is injected."""
    recorder = _Recorder()

    await _handler(recorder).ingest_text("Gina opened a clothing store.", session_id="s1")

    _assert_off_loop(recorder)


@pytest.mark.asyncio
async def test_the_turn_path_embeds_off_the_loop() -> None:
    """``ingest_turn`` embeds the window surface it is about to store."""
    recorder = _Recorder()

    await _handler(recorder).ingest_turn(
        session_id="s1",
        file_id="f1",
        seq=0,
        role="user",
        text="Gina opened a store.",
        speaker="Gina",
    )

    _assert_off_loop(recorder)


@pytest.mark.asyncio
async def test_the_redecode_path_embeds_off_the_loop() -> None:
    """A re-drive re-embeds the stored text before rewriting the chunk."""
    recorder = _Recorder()
    row = {"chunk_id": "c1", "text": "Gina opened a store.", "speaker": "", "ts": "", "seq": 0}

    await _handler(recorder)._redecode_chunk("s1", row, {"kind": "text"})

    _assert_off_loop(recorder)


@pytest.mark.asyncio
async def test_the_batch_precompute_embeds_off_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default embedder takes the batch route, and that route leaves the loop too.

    The precompute is chosen by identity against ``embed_one``, so the recorder
    is installed as the module's own function rather than injected.
    """
    from graphknows.storage import embedder

    recorder = _Recorder()

    def _embed_batch(texts: list[str], model_name: str = "") -> np.ndarray:
        recorder(texts[0])
        return np.array([VECTOR] * len(texts), dtype=np.float32)

    monkeypatch.setattr(embedder, "embed_one", recorder)
    monkeypatch.setattr(embedder, "embed", _embed_batch)
    handler = _STMIngestHandler(_store(), extractor=_StubExtractor(), embed_fn=None)
    assert handler._embed_fn is recorder

    await handler.ingest_text("Gina opened a clothing store.", session_id="s1")

    _assert_off_loop(recorder)
