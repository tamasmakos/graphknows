"""EmbedderAdapter — wraps embed/embed_one functions implementing IEmbedder."""

from __future__ import annotations

import numpy as np

from graphknows.adapters.ports.embedder import IEmbedder


class EmbedderAdapter:
    """Adapter implementing IEmbedder over the graphknows embedder module.

    Wraps the module-level ``embed`` and ``embed_one`` functions so they
    can be injected as a typed IEmbedder dependency.

    Args:
        model_name: OpenRouter model identifier (default: openai/text-embedding-3-small).
    """

    def __init__(self, model_name: str = "openai/text-embedding-3-small") -> None:
        self._model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one float vector per text."""
        from graphknows.retrieval.stores.embedder import embed as _embed

        result: np.ndarray = _embed(texts, self._model_name)
        return result.tolist()

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string and return a plain float list."""
        from graphknows.retrieval.stores.embedder import embed_one as _embed_one

        return _embed_one(text, self._model_name)


# Verify structural compatibility at import time
def _assert_protocol() -> None:
    assert issubclass(EmbedderAdapter, IEmbedder)


_assert_protocol()
