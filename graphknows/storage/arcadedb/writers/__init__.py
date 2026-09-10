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

WRITTEN_TYPES = frozenset(
    {
        # Vertices, one per writer family.
        "SOURCE",
        "SEGMENT",
        "ENTITY",
        "CONCEPT",
        "PREDICATE",
        "FACT",
        # The edges those families are written with.
        "PART_OF",
        "NEXT",
        "MENTIONS",
        "EVOKES",
        "INSTANCE_OF",
        "SUBJECT_OF",
        "OBJECT_IS",
        "USES",
        "ASSERTED_IN",
        "SUPERSEDES",
        "CONTRADICTS",
    }
)
"""Every type the write path writes — the written half of the dead-weight diff.

Declared beside the writers rather than in the reader that reports it, so a new
writer and its entry here move together: the read half is observed at runtime by
``GraphStore.read_types``, and ``Memory.flush`` reports the difference (FR-040).
"""

__all__ = [
    "WRITTEN_TYPES",
    "EntityWrite",
    "EntityWriter",
    "Evidence",
    "FactWriter",
    "SegmentWriter",
    "SourceWriter",
    "SymbolWriter",
]
