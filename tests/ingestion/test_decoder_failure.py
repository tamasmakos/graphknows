"""A failed decoder call abstains, is counted, and never eats the chunk (US2).

The failure mode this closes: a provider outage mid-ingest produced a graph
indistinguishable from one the decoder had legitimately read as empty — and, on
the message path, the chunk was not written at all, because the exception left
``_extract`` before ``_write_extracted`` ran. Both are silent data loss.

Scenario coverage (spec.md US2):

* **2.1** — budget spent -> ``decoder_failed``, chunk still stored and embedded.
* **2.2** — a call that answered with nothing -> ``decoder_empty``, a *different*
  gate, so a report can tell an outage from an empty chunk.
* **2.4** — no chunk is extracted by the local decoder under any failure.
* **2.5** — a run-permanent failure is an error on the result, not a per-chunk
  burn of the retry budget.

The provider is a stub object; nothing here reaches a network, a local model or
a real ``time.sleep``.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import litellm
import pytest

from graphknows.ingestion.extraction.llm import retry as retry_module
from graphknows.ingestion.extraction.llm.decoder import LLMDecoder
from graphknows.ingestion.extraction.llm.retry import failure_gate
from graphknows.ingestion.parsers.text import ParsedFile, TextSegment
from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.settings import GraphKnowsSettings

pytestmark = pytest.mark.unit

MODEL = "openrouter/stub/decoder-v1"
#: Both chunks name "Acme", so one reply anchors in either and a gate that fires
#: is the decoder's outcome rather than an artefact of which chunk it landed on.
TEXTS = ("Melanie works for Acme.", "Acme opened a studio.")

#: A well-formed reply carrying nothing — Scenario 2.2's "the call succeeded".
EMPTY_REPLY = json.dumps({"entities": []})
FULL_REPLY = json.dumps({"entities": [{"surface": "Acme", "label": "organization"}]})


# --- The provider's typed failures, built the way the route delivers them ----


def _rate_limit() -> Exception:
    return litellm.RateLimitError(message="slow down", llm_provider="stub", model=MODEL)


def _authentication() -> Exception:
    return litellm.AuthenticationError(message="no key", llm_provider="stub", model=MODEL)


def _context_overflow() -> Exception:
    return litellm.ContextWindowExceededError(message="too long", model=MODEL, llm_provider="stub")


def _api_error(status_code: int) -> Exception:
    return litellm.APIError(
        status_code=status_code,
        message="upstream",
        llm_provider="stub",
        model=MODEL,
        request=httpx.Request("POST", "https://stub.invalid"),
    )


class _Provider:
    """The smallest thing ``LLMDecoder`` calls: raises *failure*, else replies."""

    model = MODEL

    def __init__(self, *, failure: Exception | None = None, reply: str = FULL_REPLY) -> None:
        self._failure = failure
        self._reply = reply
        # The chunks are decoded from a thread pool, and a lost increment would
        # make the attempt counts below flaky rather than wrong.
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self, **kwargs: Any) -> list[str]:
        with self._lock:
            self.calls += 1
        if self._failure is not None:
            raise self._failure
        return [self._reply]


# --- The harness: one file ingest through a real handler and a fake store ----


@pytest.fixture(autouse=True)
def _decoder_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real waiting, and no lexical label selection.

    The backoff is asserted in ``test_retry_budget``; here it is only endured.
    Lexical selection is switched off through its own setting because it wants
    the shared spaCy pipeline, and whether a decode failure is counted is not a
    question that should need a language model installed to answer.
    """
    monkeypatch.setattr(retry_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("GRAPHKNOWS_LEXICAL_LABELS", "false")


def _decoder(provider: _Provider) -> LLMDecoder:
    decoder = LLMDecoder(GraphKnowsSettings(), lm=provider)
    # The relation vocabulary is selected from spaCy's observation of the chunk.
    # A decode failure is not the parser's business, and loading a language model
    # to prove that would make this test depend on one being installed.
    decoder.parse = lambda text: SimpleNamespace(ents=())  # type: ignore[method-assign]
    return decoder


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


async def _ingest(provider: _Provider) -> tuple[Any, MagicMock]:
    """Ingest ``TEXTS`` as one file through the assisted decoder."""
    store = _store()
    handler = _STMIngestHandler(store, extractor=_decoder(provider), embed_fn=lambda t: [0.1, 0.2])
    parsed = ParsedFile(
        source_id="f1",
        title="t",
        parser_name="text",
        segments=[TextSegment(text=t, heading_path="", char_offset=i) for i, t in enumerate(TEXTS)],
    )
    return await handler.ingest_parsed(parsed, session_id="s1"), store


def _written(store: MagicMock) -> list[dict[str, Any]]:
    return [call.args[0] for call in store.write_chunk.await_args_list]


# --- Which failure costs the chunk, and which costs the run ------------------


@pytest.mark.parametrize(
    ("exc", "gate"),
    [
        (_rate_limit(), "decoder_failed"),
        (litellm.Timeout(message="timed out", model=MODEL, llm_provider="stub"), "decoder_failed"),
        (_api_error(500), "decoder_failed"),
        (ValueError("a bug of ours is not the provider's verdict"), "decoder_failed"),
        (_context_overflow(), "decode_context_overflow"),
        (_authentication(), None),
        (litellm.BadRequestError(message="malformed", model=MODEL, llm_provider="stub"), None),
        (litellm.NotFoundError(message="no model", model=MODEL, llm_provider="stub"), None),
        (_api_error(402), None),
        (_api_error(403), None),
    ],
    ids=lambda value: value if isinstance(value, str) else type(value).__name__,
)
def test_the_classification_table(exc: Exception, gate: str | None) -> None:
    """contracts/decoder.md §4, read after the budget rather than during it.

    The oversized chunk is the case worth pinning: it carries status 400, so a
    table that read the status before the type would report it as the malformed
    request it is not — and an operator would go looking for a bad key.
    """
    assert failure_gate(exc) == gate


# --- Scenario 2.1: the budget is spent, the chunk survives -------------------


@pytest.mark.asyncio
async def test_a_spent_retry_budget_abstains_and_still_stores_the_chunk() -> None:
    provider = _Provider(failure=_rate_limit())

    result, store = await _ingest(provider)

    assert result.abstentions["decoder_failed"] == len(TEXTS)
    # Retried within the budget, then given up on — not retried for ever.
    assert provider.calls == len(TEXTS) * GraphKnowsSettings().decode_attempts
    # Stored, embedded and therefore retrievable: the outage cost the labels,
    # not the text (FR-018, SC-004).
    written = _written(store)
    assert [chunk["text"] for chunk in written] == list(TEXTS)
    assert all(chunk["embedding"] == [0.1, 0.2] for chunk in written)
    # A chunk-level failure is not the run's failure.
    assert result.errors == []


@pytest.mark.asyncio
async def test_an_oversized_chunk_abstains_under_its_own_gate() -> None:
    """A wait cannot shrink the chunk, so overflow is neither retried nor
    reported as the same thing as an outage (contracts/decoder.md §4)."""
    provider = _Provider(failure=_context_overflow())

    result, store = await _ingest(provider)

    assert result.abstentions["decode_context_overflow"] == len(TEXTS)
    assert "decoder_failed" not in result.abstentions
    assert provider.calls == len(TEXTS)  # one attempt each; no budget burnt
    assert len(_written(store)) == len(TEXTS)


# --- Scenario 2.2: nothing returned is not the same as nothing answered ------


@pytest.mark.asyncio
async def test_an_empty_answer_is_counted_under_a_different_gate() -> None:
    provider = _Provider(reply=EMPTY_REPLY)

    result, store = await _ingest(provider)

    assert result.abstentions["decoder_empty"] == len(TEXTS)
    assert "decoder_failed" not in result.abstentions
    assert result.errors == []
    assert len(_written(store)) == len(TEXTS)


@pytest.mark.asyncio
async def test_a_productive_answer_fires_neither_gate() -> None:
    """The gates are only worth reading if silence means silence."""
    result, _store = await _ingest(_Provider())

    assert "decoder_empty" not in result.abstentions
    assert "decoder_failed" not in result.abstentions


# --- Scenario 2.4: no per-chunk fallback, under any failure ------------------


@pytest.mark.asyncio
async def test_no_failure_reaches_the_local_decoder() -> None:
    """FR-022/SC-005. Structural, not a promise: the decoder that failed carries
    no local model to fall back *to*, and the chunk it gave up on is written with
    no entities rather than with GLiNER's."""
    result, store = await _ingest(_Provider(failure=_rate_limit()))

    assert not hasattr(LLMDecoder, "gliner")
    assert not hasattr(LLMDecoder, "relation_verifier")
    assert result.entity_count == 0
    assert store.write_entities_bulk.await_count == 0


# --- Scenario 2.5: run-permanent is an error, not a per-chunk burn -----------


@pytest.mark.asyncio
async def test_a_rejected_credential_is_an_error_and_burns_no_budget() -> None:
    provider = _Provider(failure=_authentication())

    result, store = await _ingest(provider)

    assert len(result.errors) == 1
    assert "AuthenticationError" in result.errors[0]
    # Not an abstention: no chunk declined to emit, the run could not decode.
    assert "decoder_failed" not in result.abstentions
    assert "decode_context_overflow" not in result.abstentions
    # The batch leaves on the first refusal instead of re-asking per chunk, and
    # nothing is retried (FR-021).
    assert provider.calls <= len(TEXTS)
    # Still stored: a re-drive needs the text to be in the graph.
    assert len(_written(store)) == len(TEXTS)


@pytest.mark.asyncio
async def test_the_run_error_is_reported_once_and_not_again_next_run() -> None:
    """A handler outlives an ingest. A rejected key re-reported on the next run —
    after it was replaced — is a false alarm, so the list is drained."""
    store = _store()
    handler = _STMIngestHandler(
        store, extractor=_decoder(_Provider(failure=_authentication())), embed_fn=lambda t: [0.1]
    )
    parsed = ParsedFile(
        source_id="f1",
        title="t",
        parser_name="text",
        segments=[TextSegment(text=t, heading_path="", char_offset=i) for i, t in enumerate(TEXTS)],
    )

    first = await handler.ingest_parsed(parsed, session_id="s1")
    handler._extractor = _decoder(_Provider())
    second = await handler.ingest_parsed(parsed, session_id="s1")

    assert len(first.errors) == 1
    assert second.errors == []


# --- The message path decodes through the same funnel ------------------------


@pytest.mark.asyncio
async def test_a_failed_turn_is_still_written() -> None:
    """``ingest_turn`` decodes through ``_extract``, where a raise used to
    abandon the turn before it was stored — the failure cost the chunk itself."""
    store = _store()
    provider = _Provider(failure=_rate_limit())
    handler = _STMIngestHandler(store, extractor=_decoder(provider), embed_fn=lambda t: [0.1])

    entities = await handler.ingest_turn(
        session_id="s1", file_id="f1", seq=0, role="user", text=TEXTS[0], speaker="Melanie"
    )

    assert entities == 0
    assert handler._abstentions["decoder_failed"] == 1
    assert [chunk["text"] for chunk in _written(store)] == [TEXTS[0]]
