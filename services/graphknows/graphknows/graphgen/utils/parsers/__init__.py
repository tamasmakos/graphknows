"""Parser package.

This module provides the document parsers.
Currently only LifeLogParser is supported.
"""

from graphknows.graphgen.utils.parsers.base import BaseDocumentParser
from graphknows.graphgen.utils.parsers.life import LifeLogParser

__all__ = [
    'BaseDocumentParser',
    'LifeLogParser',
]
