"""Public exception hierarchy for graphknows.

Catch :class:`GraphKnowsError` to handle any error raised by the library.
"""

from __future__ import annotations


class GraphKnowsError(Exception):
    """Base class for all graphknows errors."""


class ConfigurationError(GraphKnowsError):
    """Settings are invalid or incomplete (bad env vars, missing secrets)."""


class StoreError(GraphKnowsError):
    """A storage backend operation failed (connection, query, or transport)."""


class MissingExtraError(GraphKnowsError, ImportError):
    """A feature was used whose optional dependency extra is not installed.

    Only raise this for a dependency that really is behind an extra declared in
    ``pyproject.toml`` — the message tells the user to run
    ``pip install 'graphknows[<extra>]'``, and naming an extra that does not
    exist sends them to a command that installs nothing. For a *core* dependency
    that has somehow gone missing, raise :class:`BrokenInstallError` instead.

    Subclasses :class:`ImportError` so existing ``except ImportError`` handlers
    keep working.
    """

    #: Extras declared in pyproject.toml. Kept here so the test suite can assert
    #: that every call site names a real one.
    KNOWN_EXTRAS = frozenset({"assisted", "langgraph", "observability"})

    def __init__(self, feature: str, extra: str) -> None:
        super().__init__(
            f"{feature} requires the '{extra}' extra. "
            f"Install it with:  pip install 'graphknows[{extra}]'"
        )
        self.feature = feature
        self.extra = extra


class BrokenInstallError(GraphKnowsError, ImportError):
    """A *core* dependency is missing — the installation itself is broken.

    Distinct from :class:`MissingExtraError`: there is no extra to install, so
    the remedy is to repair the environment. This is what a stale container
    image looks like from the inside — the code is current, the site-packages
    are not — so the message names the offending distribution explicitly.

    Subclasses :class:`ImportError` so existing ``except ImportError`` handlers
    keep working.
    """

    def __init__(self, feature: str, distribution: str, hint: str = "") -> None:
        super().__init__(
            f"{feature} needs the '{distribution}' package, which is a core "
            f"dependency of graphknows but is not importable. The install is "
            f"incomplete — reinstall with:  pip install --force-reinstall graphknows"
            f"\n(If you are in a container, the image is stale relative to the "
            f"source it is running: rebuild it.)" + (f"\n{hint}" if hint else "")
        )
        self.feature = feature
        self.distribution = distribution
        self.hint = hint
