"""Backend connectivity health checks — config-layer portion only.

Only ``get_version()`` lives here (pure, no store dependencies).

``check_backends`` and ``get_memory_stats`` were moved to
``graphknows.runtime`` (``GraphKnowsRuntime.doctor`` /
``GraphKnowsRuntime.stats``) because they import
``graphknows.memory.stores``, which is above ``graphknows.config`` in the
layer hierarchy.
"""

from __future__ import annotations

import subprocess  # nosec


def get_version() -> str:
    """Return current git commit SHA (short)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:12]  # nosec
    except Exception:
        return "unknown"
