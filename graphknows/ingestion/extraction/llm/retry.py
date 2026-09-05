"""The decoder's own retry budget — typed signals only, spent per chunk.

``build_lm`` pins ``num_retries=0`` so litellm stacks no hidden attempts under
this one (research.md R5): the whole budget lives here, where it can key on the
provider's exception **types** rather than on substrings of their messages
(FR-020), and where a permanent failure is handed straight back without burning
an attempt (FR-021).

Three outcomes, per contracts/decoder.md §4:

* **retryable** — ``RateLimitError`` (429), ``Timeout`` (408/504),
  ``APIConnectionError``, ``ServiceUnavailableError`` (503), an ``APIError``
  carrying 500 or 502, and ``AdapterParseError`` (a reply that does not parse,
  or an empty one). Retried within the budget; on exhaustion the last exception
  is re-raised for the caller to count as ``decoder_failed``.
* **chunk-permanent** — ``ContextWindowExceededError``. A wait cannot shrink the
  chunk, so it leaves at once, unretried.
* **run-permanent** — ``AuthenticationError`` (401), ``BadRequestError``
  (400/422), ``NotFoundError`` (404), an ``APIError`` carrying 402 or 403. Also
  at once, for the ingest handler to record on ``IngestResult.errors``.

Only the first is this module's business while the budget is being spent: the
other two — and anything unknown — leave through the same ``raise``, consuming
no budget. :func:`is_retryable` therefore asks one question, not three: *is this
worth trying again?* Once trying is over, :func:`failure_gate` asks the second
one — *what does this cost, the chunk or the run?* — off the same typed signals,
so the classification table lives in one module rather than two.

The wait is exponential with **full jitter** — ``uniform(0, min(cap, base *
2**n))`` — so a batch that trips one rate limit does not re-converge on the
provider in lockstep. A ``Retry-After`` the provider supplied wins over the
computed wait, still clamped to the cap; it is read from
``exc.litellm_response_headers`` and never from ``exc.response.headers``, which
is empty on this route because ``RateLimitError`` rebuilds a synthetic response.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

#: Jitter source. ``SystemRandom`` because bandit (B311) refuses the module
#: functions in this repo's gate; the entropy is irrelevant to a backoff.
_jitter = random.SystemRandom()

#: ``APIError`` is one class over many statuses, so these two are read off the
#: status code rather than off the type: 500/502 is a server that may yet
#: answer, while 402/403 (credits, moderation) is run-permanent.
_RETRYABLE_STATUS = frozenset({500, 502})

#: The statuses that end the run rather than the chunk (contracts/decoder.md §4):
#: a malformed or refused request (400/422), rejected or absent credentials
#: (401/403), exhausted credits (402), an unknown model (404). Read off the
#: status rather than off five class names because litellm spreads them over two
#: exception hierarchies — ``PermissionDeniedError`` is not a ``litellm.APIError``
#: — while every one of them carries the code.
_RUN_PERMANENT_STATUS = frozenset({400, 401, 402, 403, 404, 422})

#: The gate names of contracts/decoder.md §5 that a *failure* lands under. The
#: third, ``decoder_empty``, is a successful call's outcome and is counted where
#: the answer is read, not here.
DECODER_FAILED = "decoder_failed"
DECODE_CONTEXT_OVERFLOW = "decode_context_overflow"


@functools.cache
def _retryable_classes() -> tuple[type[BaseException], ...]:
    """The typed retry signals, imported on first classification.

    litellm and dspy are the assisted path's imports, not the base install's;
    a process that never decodes must not pay for them at import time.
    """
    from dspy.utils.exceptions import AdapterParseError
    from litellm import exceptions as litellm_exceptions

    return (
        litellm_exceptions.RateLimitError,
        litellm_exceptions.Timeout,
        litellm_exceptions.APIConnectionError,
        litellm_exceptions.ServiceUnavailableError,
        AdapterParseError,
    )


@functools.cache
def _api_error() -> type[BaseException]:
    """``litellm.APIError`` — the one signal read by status rather than by type."""
    from litellm import exceptions as litellm_exceptions

    return litellm_exceptions.APIError


def _status_code(exc: BaseException) -> int | None:
    """The HTTP status an exception carries, or ``None`` when it carries none."""
    code = getattr(exc, "status_code", None)
    if code is None:
        return None
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def is_retryable(exc: BaseException) -> bool:
    """Whether *exc* is worth another attempt — by type, never by message (FR-020).

    Everything unlisted is permanent by default: an unrecognised failure costs
    one attempt, not the whole budget.
    """
    if isinstance(exc, _retryable_classes()):
        return True
    return isinstance(exc, _api_error()) and _status_code(exc) in _RETRYABLE_STATUS


@functools.cache
def _context_window_error() -> type[BaseException]:
    """``litellm.ContextWindowExceededError`` — chunk-permanent, its own gate."""
    from litellm import exceptions as litellm_exceptions

    return litellm_exceptions.ContextWindowExceededError


def failure_gate(exc: BaseException) -> str | None:
    """The abstention gate *exc* costs its chunk, or ``None`` when it costs the run.

    The counting half of contracts/decoder.md §4, read once trying is over:

    * ``decode_context_overflow`` — the chunk plus its closed vocabularies did
      not fit. Tested first, and by type, because the overflow carries status 400
      and would otherwise read as the malformed request it is not.
    * ``None`` — run-permanent: rejected or absent credentials, a request the
      provider refuses outright. Not an abstention at all — the chunk declined
      nothing, the run could not decode — so it is recorded on
      ``IngestResult.errors`` instead (FR-021).
    * ``decoder_failed`` — everything else, a spent retry budget included. A
      failure this table does not recognise, our own bugs among them, abstains
      for its own chunk rather than being read as the provider's verdict on
      every remaining one.
    """
    if isinstance(exc, _context_window_error()):
        return DECODE_CONTEXT_OVERFLOW
    if not is_retryable(exc) and _status_code(exc) in _RUN_PERMANENT_STATUS:
        return None
    return DECODER_FAILED


def retry_after_s(exc: BaseException) -> float | None:
    """Seconds the provider asked us to wait, or ``None`` when it asked nothing.

    Read from ``litellm_response_headers`` (research.md R5). Only the delta-seconds
    form is honoured; the HTTP-date form parses to ``None`` and falls back to the
    computed backoff rather than growing a date parser for a header this route has
    not been observed to send.
    """
    headers = getattr(exc, "litellm_response_headers", None)
    items = getattr(headers, "items", None)
    if items is None:
        return None
    for name, value in items():
        if str(name).lower() != "retry-after":
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None
    return None


@dataclass(frozen=True)
class RetryBudget:
    """One chunk's allowance of attempts, and the wait it spends between them.

    The defaults are the shipped ones (FR-020); ``from_settings`` is how the
    decoder gets the configured values, so the knobs stay in one place.
    """

    attempts: int = 3
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 30.0
    honour_retry_after: bool = True

    @classmethod
    def from_settings(cls, settings: Any) -> RetryBudget:
        """The budget as configured (``GRAPHKNOWS_DECODE_*``)."""
        return cls(
            attempts=settings.decode_attempts,
            backoff_base_s=settings.decode_backoff_base_s,
            backoff_cap_s=settings.decode_backoff_cap_s,
            honour_retry_after=settings.decode_honour_retry_after,
        )

    def run(self, call: Callable[[], T]) -> T:
        """Call *call*, retrying it while the failure is retryable and budget is left.

        The exception that ends it is re-raised untouched — the caller reads its
        type to pick the abstention gate, so nothing may be wrapped here.
        """
        for attempt in range(1, self.attempts + 1):
            try:
                decoded = call()
            except Exception as exc:
                if attempt >= self.attempts or not is_retryable(exc):
                    raise
                wait = self._wait_s(exc, attempt)
                log.warning(
                    "decode attempt %d/%d failed (%s: %s); retrying in %.2fs",
                    attempt,
                    self.attempts,
                    type(exc).__name__,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                return decoded
        raise AssertionError("unreachable: the last attempt either returns or raises")

    def _wait_s(self, exc: BaseException, attempt: int) -> float:
        """The wait before attempt *attempt* + 1, in seconds.

        ``Retry-After`` when the provider supplied one and we honour it, else
        full jitter under a ceiling that doubles per attempt. Both are clamped to
        ``backoff_cap_s``: the cap bounds a single wait however it was chosen.
        """
        if self.honour_retry_after and (asked := retry_after_s(exc)) is not None:
            return min(asked, self.backoff_cap_s)
        ceiling = min(self.backoff_cap_s, self.backoff_base_s * 2 ** (attempt - 1))
        return _jitter.uniform(0.0, ceiling)
