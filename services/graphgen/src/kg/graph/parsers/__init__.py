"""Parser package.

This module provides the document parsers.
Currently only LifeLogParser is supported.
"""

from kg.graph.parsers.base import BaseDocumentParser
from kg.graph.parsers.life import LifeLogParser

__all__ = [
    'BaseDocumentParser',
    'LifeLogParser',
]
