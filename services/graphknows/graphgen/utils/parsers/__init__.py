"""Parser package.

This module provides the document parsers.
Currently only LifeLogParser is supported.
"""

from graphgen.utils.parsers.base import BaseDocumentParser
from graphgen.utils.parsers.life import LifeLogParser

__all__ = [
    'BaseDocumentParser',
    'LifeLogParser',
]
