"""Top-level service wiring for graphknows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphknows.container import GraphKnowsContainer

from graphknows.config import GraphKnowsSettings


def build_service(settings: GraphKnowsSettings | None = None) -> GraphKnowsContainer:
    """Build the full GraphKnows service container."""
    from graphknows.container import GraphKnowsContainer

    return GraphKnowsContainer.from_settings(settings or GraphKnowsSettings())
