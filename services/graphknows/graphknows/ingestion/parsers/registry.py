"""Parser registry: dispatch by file extension or explicit hint."""

from __future__ import annotations

import os

from .base import BaseParser
from .code import CodeParser
from .docx import DocxParser
from .text import PlainTextParser

_EXT_MAP: dict[str, type[BaseParser]] = {
    ".txt": PlainTextParser,
    ".md": PlainTextParser,
    ".py": CodeParser,
    ".js": CodeParser,
    ".ts": CodeParser,
    ".sql": CodeParser,
    ".sh": CodeParser,
    ".yaml": CodeParser,
    ".yml": CodeParser,
    ".json": CodeParser,
    ".docx": DocxParser,
}


def get_parser(path_or_hint: str = "") -> BaseParser:
    """Return the appropriate parser for a path or file extension hint.

    Args:
        path_or_hint: File path, URL, or extension string (e.g. ".py").
                      Defaults to PlainTextParser if unrecognized.

    Returns:
        Instantiated BaseParser subclass.
    """
    if path_or_hint.startswith(".") and "/" not in path_or_hint:
        ext = path_or_hint.lower()
    else:
        ext = os.path.splitext(path_or_hint)[-1].lower()
    cls = _EXT_MAP.get(ext, PlainTextParser)
    return cls()
