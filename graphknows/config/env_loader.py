"""Centralised .env file loading for graphknows."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Load a .env file and return its key-value pairs.

    Resolution order:
    1. GRAPHKNOWS_ENV_FILE env var (must be absolute path if set)
    2. ``path`` argument
    3. Returns empty dict if neither is set or file not found.
    """
    env_file_env = os.environ.get("GRAPHKNOWS_ENV_FILE")
    if env_file_env:
        resolved = Path(env_file_env)
    elif path is not None:
        resolved = path
    else:
        return {}

    if not resolved.exists():
        return {}
    return {k: v for k, v in dotenv_values(resolved).items() if isinstance(v, str)}
