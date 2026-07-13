"""Package version — the single source of truth read by hatch and get_version()."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__version__ = "2.0.0"


def get_version() -> str:
    """Return the installed graphknows package version (falls back to __version__)."""
    try:
        return version("graphknows")
    except PackageNotFoundError:
        return __version__
