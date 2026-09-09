"""Golden-layer writers, one per vertex family the schema declares.

Each writer owns one family — source, segment, fact, entity, symbol — and the
edges that family cannot exist without: a segment is written with its PART_OF,
a fact with its ASSERTED_IN evidence or not at all (FR-008).
"""

from __future__ import annotations

from graphknows.storage.arcadedb.writers.entity import EntityWrite, EntityWriter
from graphknows.storage.arcadedb.writers.fact import Evidence, FactWriter
from graphknows.storage.arcadedb.writers.segment import SegmentWriter
from graphknows.storage.arcadedb.writers.source import SourceWriter
from graphknows.storage.arcadedb.writers.symbol import SymbolWriter

__all__ = [
    "EntityWrite",
    "EntityWriter",
    "Evidence",
    "FactWriter",
    "SegmentWriter",
    "SourceWriter",
    "SymbolWriter",
]
