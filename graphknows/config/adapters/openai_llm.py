"""OpenAILlm — wraps OpenRouter LLM client implementing ILlm."""

from __future__ import annotations

import logging

import httpx

from graphknows.config.adapters.ports.llm import ILlm

logger = logging.getLogger(__name__)


class OpenRouterLlm:
    """Adapter implementing ILlm over the OpenRouter chat completions API.

    Args:
        api_key: OpenRouter API key.
        model: Model identifier (default: meta-llama/llama-3.1-8b-instruct).
        timeout: HTTP request timeout in seconds (default: 30).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def complete(self, prompt: str, **kwargs: object) -> str:
        """Generate a completion via OpenRouter (synchronous).

        Args:
            prompt: Input prompt string.
            **kwargs: Optional overrides: model (str), temperature (float),
                max_tokens (int).

        Returns:
            Completion text as a string, or empty string on failure.
        """
        model = str(kwargs.get("model", self._model))
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        try:
            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return str(resp.json()["choices"][0]["message"]["content"])
        except Exception as exc:
            logger.warning("OpenRouterLlm.complete failed: %s", exc)
            return ""


# Verify structural compatibility at import time
def _assert_protocol() -> None:
    assert issubclass(OpenRouterLlm, ILlm)


_assert_protocol()
