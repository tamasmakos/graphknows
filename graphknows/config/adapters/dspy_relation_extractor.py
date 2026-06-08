"""Adapter for graphgen's DSPy relation extractor."""

from __future__ import annotations

from typing import Any

from graphknows.config.adapters.ports.relation_extractor import IRelationExtractor


class DspyRelationExtractorAdapter:
    """Lazy adapter implementing IRelationExtractor over graphgen DSPyExtractor."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._delegate: Any | None = None

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from graphknows.graphgen.extraction.relations import DSPyExtractor

            self._delegate = DSPyExtractor(config=self._config)
        return self._delegate

    async def extract_relations(
        self,
        text: str,
    ) -> tuple[list[tuple[str, str, str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
        """Extract relation triplets by delegating to graphgen's DSPy extractor."""
        return await self._get_delegate().extract_relations(  # type: ignore[no-any-return]
            text=text,
        )


def _assert_protocol() -> None:
    assert issubclass(DspyRelationExtractorAdapter, IRelationExtractor)


_assert_protocol()
