"""Logging configuration helpers."""

import logging
import sys
from typing import Optional

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(debug: bool = False, level: Optional[int] = None) -> None:
    """Configure logging for CLI and pipeline entry points."""
    effective_level = level if level is not None else (logging.DEBUG if debug else logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=effective_level,
            format=DEFAULT_LOG_FORMAT,
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    root.setLevel(effective_level)
