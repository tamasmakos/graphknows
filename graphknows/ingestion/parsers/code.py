"""The code parser: Python source into one segment per function and class.

Stdlib ``ast`` only — a definition's qualified symbol ("Store.put") is the
segment ``path``, so extraction cites a symbol rather than a line range.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from typing import TypeAlias

from graphknows.models.segment import Segment, SegmentKind
from graphknows.models.source import Source

Definition: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _line_starts(data: bytes) -> list[int]:
    """Byte offset of each line, so an ``ast`` position becomes a byte offset."""
    starts = [0]
    for line in data.split(b"\n")[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    return starts


def _definitions(
    body: Iterable[ast.stmt], prefix: tuple[str, ...]
) -> Iterator[tuple[Definition, str]]:
    """Every definition under *body* with its qualified name, outer before inner."""
    for node in body:
        if not isinstance(node, Definition):
            continue
        qualified = (*prefix, node.name)
        yield node, ".".join(qualified)
        yield from _definitions(node.body, qualified)


class PythonCodeParser:
    """Cut Python source at its definitions, keeping bodies whole.

    A class yields one segment covering the whole class and one more per method,
    so a query can land on either the type or the single function it names.
    """

    mimes = frozenset({"text/x-python"})
    version = "python-ast/1"

    def parse(self, source: Source, data: bytes) -> Iterator[Segment]:
        """Yield one segment per function and class, in source order."""
        starts = _line_starts(data)
        for node, qualified in _definitions(ast.parse(data).body, ()):
            start = starts[node.lineno - 1] + node.col_offset
            if node.end_lineno is None or node.end_col_offset is None:  # never for parsed source
                raise ValueError(f"{qualified}: node has no end position")
            end = starts[node.end_lineno - 1] + node.end_col_offset
            yield Segment(
                source_id=source.id,
                text=data[start:end].decode(),
                kind=SegmentKind.code,
                path=qualified,
                byte_range=(start, end),
                extractor_version=self.version,
            )
