"""Backend connectivity health checks — config-layer portion only.

Only ``get_version()`` lives here (pure, no store dependencies).

``check_backends`` and ``get_memory_stats`` were moved to
``graphknows.runtime`` (``GraphKnowsRuntime.doctor`` /
``GraphKnowsRuntime.stats``) because they import
``graphknows.memory.stores``, which is above ``graphknows.config`` in the
layer hierarchy.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Return the installed graphknows package version."""
    try:
        return version("graphknows")
    except PackageNotFoundError:
        from graphknows import __version__

        return __version__
