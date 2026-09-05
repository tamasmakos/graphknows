"""The decoder seam, driven by a stub provider: one call per chunk, fanned out.

No network anywhere: a ``dspy.LM`` subclass stands in for the provider, counts
the requests it is handed and watches how many are in flight at once. That is
enough to pin the three things the decoder actually promises — one structured
request per chunk (FR-010), the strict-schema wiring that makes the request
worth making (research.md R4), and the bounded concurrency that makes a batch of
them survivable — plus the two attributes whose *absence* is the capability
signal downstream reads (contracts/decoder.md §1).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import dspy
import litellm
import pytest

from graphknows.ingestion.extraction.llm.decoder import LLMDecoder
from graphknows.ingestion.extraction.llm.schema import FrameCandidate
from graphknows.settings import GraphKnowsSettings

TEXT = "Melanie works for Acme. Priya works for Acme too."
ENTITY_LABELS = ("person", "organization")
RELATION_SPEC = {"worksFor": "the subject is employed by the object"}
EMPLOYED = FrameCandidate(
    frame="Being_employed",
    trigger="works",
    trigger_offset=8,
    core_elements=(("Employee", "the one employed"), ("Employer", "the one employing")),
)

STUB_MODEL = "openrouter/stub/decoder-v1"

ENTITIES = [
    {"surface": "Melanie", "label": "person"},
    {"surface": "Acme", "label": "organization"},
]

REPLY = json.dumps(
    {
        "entities": ENTITIES,
        "relations": [
            {
                "head": "Melanie",
                "predicate": "worksFor",
                "tail": "Acme",
                "evidence": "Melanie works for Acme",
                "confidence": 0.9,
            }
        ],
        "frames": [
            {
                "frame": "Being_employed",
                "trigger": "works",
                "roles": {"Employee": ["Melanie"], "Employer": ["Acme"]},
            }
        ],
    }
)


class _StubProvider(dspy.LM):  # type: ignore[misc]
    """Stands in for the provider: records every request and the peak fan-out."""

    def __init__(self, reply: str = REPLY, latency: float = 0.0) -> None:
        super().__init__(model=STUB_MODEL, api_key="stub", cache=False)
        self.reply = reply
        self.latency = latency
        self.requests: list[Any] = []
        self.peak_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        return len(self.requests)

    def __call__(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> list[str]:
        with self._lock:
            self.requests.append(messages if messages is not None else prompt)
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        time.sleep(self.latency)
        with self._lock:
            self._in_flight -= 1
        return [self.reply]


def _decoder(provider: _StubProvider) -> LLMDecoder:
    """A decoder wired to *provider*, on settings read from the environment."""
    return LLMDecoder(GraphKnowsSettings(), lm=provider)


def _sent(provider: _StubProvider) -> str:
    """Everything the provider was told, as one blob to search."""
    return json.dumps(provider.requests)


# --- One chunk, one request --------------------------------------------------


def test_one_chunk_costs_exactly_one_request() -> None:
    provider = _StubProvider()

    _decoder(provider).extract(TEXT, ENTITY_LABELS, RELATION_SPEC, frame_candidates=(EMPLOYED,))

    assert provider.calls == 1


def test_a_response_that_does_not_parse_still_costs_exactly_one_request() -> None:
    # FR-010 holds on the failure path too: no second adapter, no re-ask. A
    # failed parse IS a retryable signal (FR-020), so the budget is pinned to
    # one attempt here — what must not fire is dspy's own JSON-mode re-ask,
    # which would make even a single attempt cost two requests. The budget's
    # own counting is tests/extraction/test_retry_budget.py's business.
    provider = _StubProvider(reply='{"not_the_signature": true}')
    decoder = LLMDecoder(GraphKnowsSettings(GRAPHKNOWS_DECODE_ATTEMPTS=1), lm=provider)

    results = decoder.extract_batch([TEXT], [ENTITY_LABELS], [RELATION_SPEC])

    assert provider.calls == 1
    assert results[0].entities == []


def test_the_three_sections_come_back_populated() -> None:
    result = _decoder(_StubProvider()).extract(
        TEXT, ENTITY_LABELS, RELATION_SPEC, frame_candidates=(EMPLOYED,)
    )

    assert sorted(entity["name"] for entity in result.entities) == ["Acme", "Melanie"]
    assert [relation["relation"] for relation in result.relations] == ["worksFor"]
    assert result.frames[0]["roles"] == {"Employee": ["Melanie"], "Employer": ["Acme"]}


def test_the_request_carries_the_closed_schema_and_the_speaker() -> None:
    provider = _StubProvider()
    decoder = _decoder(provider)

    # A speaker who does not appear in the chunk, so finding the name in the
    # request proves the speaker travelled with it.
    decoder.extract(TEXT, ENTITY_LABELS, RELATION_SPEC, "Caroline", frame_candidates=(EMPLOYED,))

    sent = _sent(provider)
    assert "worksFor" in sent
    assert "Being_employed" in sent
    assert "organization" in sent
    assert "Caroline" in sent


def test_a_section_with_no_input_is_not_requested_and_the_omission_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Only the entity section is asked for, so only it comes back.
    provider = _StubProvider(reply=json.dumps({"entities": ENTITIES}))
    decoder = _decoder(provider)

    with caplog.at_level(logging.DEBUG, logger="graphknows.ingestion.extraction.llm.decoder"):
        result = decoder.extract(TEXT, ENTITY_LABELS)

    assert sorted(entity["name"] for entity in result.entities) == ["Acme", "Melanie"]

    sent = _sent(provider)
    assert "relation_spec" not in sent
    assert "frame_candidates" not in sent
    assert "decode section omitted: relation_spec" in caplog.text
    assert "decode section omitted: frame_candidates" in caplog.text


# --- The strict-schema wiring (research.md R4) -------------------------------


def test_the_call_goes_through_a_pinned_json_adapter() -> None:
    # Never the ambient adapter: ChatAdapter sends no response_format at all,
    # and its fallback re-call would double the cost of every failure.
    decoder = _decoder(_StubProvider())

    assert isinstance(decoder.adapter, dspy.JSONAdapter)


def test_the_model_is_registered_as_schema_capable_at_construction() -> None:
    # Unregistered, litellm knows nothing of this id and JSONAdapter would
    # downgrade the request to {"type": "json_object"} (research.md R4).
    _decoder(_StubProvider())

    assert litellm.supports_response_schema(STUB_MODEL)


# --- The bounded fan-out -----------------------------------------------------


def test_a_batch_runs_concurrently_up_to_the_configured_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPHKNOWS_DECODE_CONCURRENCY", "2")
    provider = _StubProvider(latency=0.05)
    decoder = _decoder(provider)
    texts = [TEXT] * 4

    results = decoder.extract_batch(
        texts,
        [ENTITY_LABELS] * 4,
        [RELATION_SPEC] * 4,
        frame_candidates=[(EMPLOYED,)] * 4,
    )

    assert provider.calls == 4
    assert len(results) == 4
    assert 1 < provider.peak_in_flight <= 2


def test_an_empty_batch_calls_nothing() -> None:
    provider = _StubProvider()

    assert _decoder(provider).extract_batch([]) == []
    assert provider.calls == 0


# --- Absence is the capability ----------------------------------------------


def test_the_decoder_exposes_no_local_model_attributes() -> None:
    decoder = _decoder(_StubProvider())

    assert not hasattr(decoder, "gliner")
    assert not hasattr(decoder, "relation_verifier")


def test_the_shared_spacy_pipeline_is_reachable_without_loading_it() -> None:
    # `nlp` is a property so the seam's consumers get the ONE resident pipeline;
    # asserting the descriptor keeps this test off en_core_web_lg.
    assert isinstance(LLMDecoder.nlp, property)
    assert callable(LLMDecoder.parse)
