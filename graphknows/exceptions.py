"""Public exception hierarchy for graphknows.

Catch :class:`GraphKnowsError` to handle any error raised by the library.
"""

from __future__ import annotations


class GraphKnowsError(Exception):
    """Base class for all graphknows errors."""


class ConfigurationError(GraphKnowsError):
    """Invalid or missing configuration (bad settings, unsafe production secrets)."""


class StoreConnectionError(GraphKnowsError):
    """Could not connect to or query the ArcadeDB backend."""


class ExtractionError(GraphKnowsError):
    """Entity/relation extraction failed."""


class MissingExtraError(GraphKnowsError, ImportError):
    """A feature was used whose optional dependency extra is not installed.

    Subclasses :class:`ImportError` so existing ``except ImportError`` handlers
    keep working.
    """

    def __init__(self, feature: str, extra: str) -> None:
        super().__init__(
            f"{feature} requires the '{extra}' extra. "
            f"Install it with:  pip install 'graphknows[{extra}]'"
        )
        self.feature = feature
        self.extra = extra
