"""The decoder-local retry budget: which signals, how many, how long a wait.

No network and no real sleeping: the failures are litellm's own exception
objects, constructed here, and ``time.sleep`` is captured so the waits can be
asserted instead of endured.

The three cases quickstart.md §3 names are :func:`test_a_rate_limit_is_retried_
until_it_succeeds`, :func:`test_a_permanent_failure_costs_exactly_one_attempt`
and :func:`test_retry_after_from_litellm_response_headers_wins_over_the_backoff`;
the rest pin the classification table of contracts/decoder.md §4, whose whole
point is that it reads exception *types*, never message substrings (FR-020).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import litellm
import pytest
from dspy.utils.exceptions import AdapterParseError

from graphknows.ingestion.extraction.llm import retry as retry_module
from graphknows.ingestion.extraction.llm.decoder import LLMDecoder
from graphknows.ingestion.extraction.llm.retry import RetryBudget, is_retryable, retry_after_s
from graphknows.settings import GraphKnowsSettings

MODEL = "openrouter/stub/decoder-v1"


# --- The failures, as the provider's own typed objects ----------------------


def _rate_limit() -> Exception:
    return litellm.RateLimitError(message="slow down", llm_provider="stub", model=MODEL)


def _authentication() -> Exception:
    return litellm.AuthenticationError(message="no key", llm_provider="stub", model=MODEL)


def _api_error(status_code: int) -> Exception:
    return litellm.APIError(
        status_code=status_code,
        message="upstream",
        llm_provider="stub",
        model=MODEL,
        request=httpx.Request("POST", "https://stub.invalid"),
    )


def _parse_error() -> Exception:
    # dspy renders the signature's output fields into the message, so the stub
    # has to answer `output_fields` — nothing else is read.
    return AdapterParseError(
        adapter_name="JSONAdapter",
        signature=SimpleNamespace(output_fields={"entities": None}),
        lm_response="",
    )


def _with_retry_after(exc: Exception, value: Any) -> Exception:
    """*exc* as the route actually delivers it: headers on the exception itself."""
    exc.litellm_response_headers = {"Retry-After": value}  # type: ignore[attr-defined]
    return exc


class _Stub:
    """A call that raises its way down *failures*, then returns ``"ok"``."""

    def __init__(self, *failures: Exception) -> None:
        self._failures = list(failures)
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)
        return "ok"


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every wait the budget asks for, in order — and none of them endured."""
    waits: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", waits.append)
    return waits


BUDGET = RetryBudget(attempts=3, backoff_base_s=1.0, backoff_cap_s=30.0)


# --- Quickstart §3, case 1: transient, retried, bounded ----------------------


def test_a_rate_limit_is_retried_until_it_succeeds(slept: list[float]) -> None:
    call = _Stub(_rate_limit(), _rate_limit())

    assert BUDGET.run(call) == "ok"

    assert call.calls == 3
    # Full jitter: each wait lies under a ceiling that doubles, and under the cap.
    assert len(slept) == 2
    assert all(0.0 <= wait <= min(30.0, 1.0 * 2**n) for n, wait in enumerate(slept))


def test_the_budget_bounds_the_attempts_and_re_raises_the_last_failure(
    slept: list[float],
) -> None:
    call = _Stub(*[_rate_limit() for _ in range(5)])

    with pytest.raises(litellm.RateLimitError):
        BUDGET.run(call)

    assert call.calls == 3
    assert len(slept) == 2


def test_a_single_attempt_budget_never_sleeps(slept: list[float]) -> None:
    call = _Stub(_rate_limit())

    with pytest.raises(litellm.RateLimitError):
        RetryBudget(attempts=1).run(call)

    assert call.calls == 1
    assert slept == []


# --- Quickstart §3, case 2: permanent, one attempt, no budget burnt ----------


def test_a_permanent_failure_costs_exactly_one_attempt(slept: list[float]) -> None:
    call = _Stub(_authentication())

    with pytest.raises(litellm.AuthenticationError):
        BUDGET.run(call)

    assert call.calls == 1
    assert slept == []


def test_an_oversized_chunk_costs_exactly_one_attempt(slept: list[float]) -> None:
    # Chunk-permanent, not run-permanent — but a wait cannot shrink the chunk
    # either, so it leaves on the first attempt just the same.
    call = _Stub(
        litellm.ContextWindowExceededError(message="too long", model=MODEL, llm_provider="stub")
    )

    with pytest.raises(litellm.ContextWindowExceededError):
        BUDGET.run(call)

    assert call.calls == 1
    assert slept == []


# --- Quickstart §3, case 3: Retry-After wins, clamped to the cap -------------


def test_retry_after_from_litellm_response_headers_wins_over_the_backoff(
    slept: list[float],
) -> None:
    call = _Stub(_with_retry_after(_rate_limit(), "7"))

    assert BUDGET.run(call) == "ok"

    assert slept == [7.0]


def test_retry_after_is_clamped_to_the_cap(slept: list[float]) -> None:
    call = _Stub(_with_retry_after(_rate_limit(), "120"))

    assert BUDGET.run(call) == "ok"

    assert slept == [30.0]


def test_retry_after_is_ignored_when_the_setting_says_so(slept: list[float]) -> None:
    budget = RetryBudget(backoff_base_s=1.0, backoff_cap_s=30.0, honour_retry_after=False)

    assert budget.run(_Stub(_with_retry_after(_rate_limit(), "7"))) == "ok"

    assert slept != [7.0]
    assert 0.0 <= slept[0] <= 1.0


@pytest.mark.parametrize("value", ["Wed, 21 Oct 2015 07:28:00 GMT", "", None, "soon"])
def test_an_unparseable_retry_after_falls_back_to_the_computed_backoff(value: Any) -> None:
    # Only the delta-seconds form is honoured; anything else must not poison the
    # wait with a NaN or an exception.
    assert retry_after_s(_with_retry_after(_rate_limit(), value)) is None


def test_an_exception_carrying_no_headers_asks_for_nothing() -> None:
    assert retry_after_s(_rate_limit()) is None


# --- The classification table (contracts/decoder.md §4) ----------------------


@pytest.mark.parametrize(
    "exc",
    [
        _rate_limit(),
        litellm.Timeout(message="timed out", model=MODEL, llm_provider="stub"),
        litellm.APIConnectionError(message="dropped", llm_provider="stub", model=MODEL),
        litellm.ServiceUnavailableError(message="down", llm_provider="stub", model=MODEL),
        _api_error(500),
        _api_error(502),
        _parse_error(),
    ],
    ids=lambda exc: type(exc).__name__ + str(getattr(exc, "status_code", "")),
)
def test_the_retryable_signals_are_retryable(exc: Exception) -> None:
    assert is_retryable(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        litellm.AuthenticationError(message="no key", llm_provider="stub", model=MODEL),
        litellm.BadRequestError(message="malformed", model=MODEL, llm_provider="stub"),
        litellm.NotFoundError(message="no model", model=MODEL, llm_provider="stub"),
        litellm.ContextWindowExceededError(message="too long", model=MODEL, llm_provider="stub"),
        _api_error(402),
        _api_error(403),
        ValueError("a bug in our own code is not a provider signal"),
    ],
    ids=lambda exc: type(exc).__name__ + str(getattr(exc, "status_code", "")),
)
def test_the_permanent_signals_are_not_retryable(exc: Exception) -> None:
    assert is_retryable(exc) is False


def test_a_rate_limit_message_on_the_wrong_type_is_not_retried() -> None:
    # The whole of FR-020: the classification reads the type, never the text.
    assert is_retryable(RuntimeError("429 RateLimitError: rate limit exceeded")) is False


# --- The knobs, and the decoder that spends them -----------------------------


def test_the_budget_reads_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHKNOWS_DECODE_ATTEMPTS", "5")
    monkeypatch.setenv("GRAPHKNOWS_DECODE_BACKOFF_BASE_S", "0.25")
    monkeypatch.setenv("GRAPHKNOWS_DECODE_BACKOFF_CAP_S", "4")
    monkeypatch.setenv("GRAPHKNOWS_DECODE_HONOUR_RETRY_AFTER", "false")

    budget = RetryBudget.from_settings(GraphKnowsSettings())

    assert budget == RetryBudget(
        attempts=5, backoff_base_s=0.25, backoff_cap_s=4.0, honour_retry_after=False
    )


TEXT = "Melanie works for Acme."
REPLY = json.dumps({"entities": [{"surface": "Melanie", "label": "person"}]})


class _FlakyProvider:
    """The smallest thing ``LLMDecoder`` will call: *failures* 429s, then a reply."""

    model = MODEL

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def __call__(self, **kwargs: Any) -> list[str]:
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise _rate_limit()
        return [REPLY]


def test_the_decoder_spends_the_budget_on_a_rate_limited_chunk(slept: list[float]) -> None:
    # The budget is only worth having if the decoder calls it: two 429s in front
    # of a good reply cost three provider requests for the one chunk.
    provider = _FlakyProvider(failures=2)

    result = LLMDecoder(GraphKnowsSettings(), lm=provider).extract(TEXT, ("person",))

    assert provider.calls == 3
    assert len(slept) == 2
    assert [entity["name"] for entity in result.entities] == ["Melanie"]


def test_the_decoder_gives_a_chunk_up_when_the_budget_runs_out(slept: list[float]) -> None:
    provider = _FlakyProvider(failures=99)
    decoder = LLMDecoder(GraphKnowsSettings(), lm=provider)

    with pytest.raises(litellm.RateLimitError):
        decoder.extract(TEXT, ("person",))

    assert provider.calls == GraphKnowsSettings().decode_attempts
    assert len(slept) == provider.calls - 1
