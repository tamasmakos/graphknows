"""GlinerExtractor — wraps GLiNER2EntityExtractor implementing IEntityExtractor."""

from __future__ import annotations

from typing import Any

from graphknows.adapters.ports.extractor import IEntityExtractor


class GlinerExtractor:
    """Adapter implementing IEntityExtractor over a GLiNER2-compatible extractor.

    Args:
        extractor: Any object with a ``predict_entities`` or compatible method.
            Typically a GLiNER2EntityExtractor instance.
    """

    def __init__(self, extractor: Any) -> None:
        self._extractor = extractor

    def extract(self, text: str) -> list[dict[str, object]]:
        """Extract named entities from text.

        Delegates to the underlying extractor. Supports extractors with
        ``predict_entities(text, labels)`` or ``extract(text)`` methods.

        Args:
            text: Input text to extract entities from.

        Returns:
            List of entity dicts with at least 'text' and 'label' keys.
        """
        # GLiNER2EntityExtractor uses predict_entities; fall back to extract()
        if hasattr(self._extractor, "predict_entities"):
            results: list[dict[str, object]] = self._extractor.predict_entities(text, [])
        elif hasattr(self._extractor, "extract"):
            results = self._extractor.extract(text)
        else:
            results = []
        return results


# Verify structural compatibility at import time
def _assert_protocol() -> None:
    assert issubclass(GlinerExtractor, IEntityExtractor)


_assert_protocol()
