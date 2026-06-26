"""URL helpers for the local Memory Mirror UI.

``mirror_url`` has been moved to ``graphknows.config.links`` (transport-neutral,
no web dependency) to break the runtime→web circular import.  This module
re-exports it for backward compatibility.
"""

from __future__ import annotations

from graphknows.config.links import DEFAULT_HOST, DEFAULT_PORT, mirror_url

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "mirror_url"]
