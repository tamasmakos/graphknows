"""Async OpenRouter chat client used for answer generation and LLM judging.

Plain chat-completions over httpx — no prompt-compiler framework in the loop,
so what we send is exactly the prompt text in each benchmark's ``prompts.py``.
Handles the OpenRouter failure taxonomy learned from long eval runs:

* 402 / credit exhaustion → raise :class:`CreditExhaustedError` (stop the run)
* 429 rate limit → backoff 30s·n
* 400 provider rejection → short backoff; a fresh request may land on a
  different upstream provider
* timeout / connection drop → short backoff retry (120s request timeout so a
  hung call can never wedge the run)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

import httpx

from graphknows.config.llm import get_openrouter_api_key, normalize_openrouter_model

log = logging.getLogger(__name__)

_API_BASE = "https://openrouter.ai/api/v1"
_MAX_ATTEMPTS = 6


class CreditExhaustedError(RuntimeError):
    """OpenRouter credits ran out — add credits and restart the run."""


def resolve_model(explicit: str = "", *env_vars: str) -> str:
    """Resolve a model id from an explicit value or the first set env var."""
    candidates = [explicit, *(os.environ.get(v, "") for v in env_vars)]
    for candidate in candidates:
        if candidate and candidate.strip():
            return normalize_openrouter_model(candidate.strip())
    return ""


class ChatLLM:
    """One model endpoint; reuse a single instance across a run."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> None:
        if not model:
            raise RuntimeError("No model configured. Set LLM_MODEL in .env.")
        key = api_key or get_openrouter_api_key()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in the environment.")
        # dspy/litellm-style prefixes ("openrouter/x/y") → raw OpenRouter id ("x/y")
        self.model = model.removeprefix("openrouter/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._headers = {"Authorization": f"Bearer {key}"}
        self._client = httpx.AsyncClient(base_url=_API_BASE, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Return ``(text, usage)`` for one chat completion.

        ``usage`` keys: ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
        ``cost_usd`` (OpenRouter-reported, 0.0 when absent).
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
            "usage": {"include": True},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await self._client.post(
                    "/chat/completions", json=body, headers=self._headers
                )
                if response.status_code == 402:
                    raise CreditExhaustedError(response.text[:500])
                if response.status_code in (429, 400) or response.status_code >= 500:
                    raise _RetryableHTTPError(response.status_code, response.text[:500])
                response.raise_for_status()
                payload = response.json()
                text = _first_content(payload)
                if text is None:
                    raise _RetryableHTTPError(200, "empty completion body")
                return text, _usage(payload)
            except CreditExhaustedError:
                raise
            except _RetryableHTTPError as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"LLM call failed after {_MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                wait = 30 * (attempt + 1) if exc.status == 429 else 5 * (attempt + 1)
                log.warning(
                    "Retryable LLM error (HTTP %s) attempt %d/%d, waiting %ds: %s",
                    exc.status, attempt + 1, _MAX_ATTEMPTS, wait, exc.detail[:200],
                )
                await asyncio.sleep(wait)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"LLM call failed after {_MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                wait = 5 * (attempt + 1)
                log.warning(
                    "Transient transport error attempt %d/%d, waiting %ds: %s",
                    attempt + 1, _MAX_ATTEMPTS, wait, exc,
                )
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def generate_json(
        self, system: str, user: str, *, max_tokens: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Chat completion that must return a JSON object; parsed leniently."""
        text, usage = await self.generate(system, user, max_tokens=max_tokens, json_mode=True)
        return parse_json_object(text), usage


class _RetryableHTTPError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.detail = detail


def _first_content(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices") or []
    if not choices:
        return None
    content = (choices[0].get("message") or {}).get("content")
    return content if isinstance(content, str) and content.strip() else None


def _usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cost_usd": float(usage.get("cost", 0.0) or 0.0),
    }


_THINKING_RE = re.compile(r"[<\[]/?(?:mem_)?thinking[>\]].*?[<\[]/(?:mem_)?thinking[>\]]", re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_answer(text: str) -> str:
    """Pull the final answer out of a reasoning-style completion.

    Strips ``<thinking>``/``<mem_thinking>`` blocks, then takes everything after
    the last ``ANSWER:`` marker when present.
    """
    cleaned = _THINKING_RE.sub("", text or "").strip()
    marker = cleaned.rfind("ANSWER:")
    if marker >= 0:
        cleaned = cleaned[marker + len("ANSWER:") :]
    return cleaned.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a completion, tolerating code fences and prose.

    Unwraps a nested ``{"final": {...}}`` wrapper some models emit. Returns an
    empty dict when nothing parseable is found (callers treat that as a miss).
    """
    candidates = [text]
    match = _JSON_BLOCK_RE.search(text or "")
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            inner = parsed.get("final")
            if isinstance(inner, dict) and len(parsed) == 1:
                return inner
            return parsed
    return {}
