"""FR-029 / SC-018: write_frame_instances was split, and stays under the ceiling.

``write_frame_instances`` was 432 lines — the longest function in the repo —
before the split named in FR-029. This asserts the split actually happened
(every named phase exists as its own function) and that none of the
resulting functions exceeds the 150-line ceiling SC-018 sets. Static (AST),
so it needs no ArcadeDB connection.
"""

from __future__ import annotations

import ast
import inspect

from graphknows.storage.arcadedb import graph_store

_LINE_CEILING = 150

# The orchestrator plus every private helper write_frame_instances' own
# phase comments (filler lookup, row derivation, diagnostics, one write per
# batch) were split into. This is the exact set FR-029/SC-018 scope the
# ceiling to — not every function in the module.
_SPLIT_FUNCTIONS = (
    "write_frame_instances",
    "_known_frame_fillers",
    "_build_frame_instance_rows",
    "_log_frame_instance_diagnostics",
    "_write_frame_instance_vertices",
    "_link_frame_instances_to_frames",
    "_write_frame_role_edges",
    "_write_frame_co_role_edges",
    "_write_frame_nesting_edges",
    "_frame_co_role_rel_rows",
)


def _split_function_line_counts() -> dict[str, int]:
    source = inspect.getsource(graph_store)
    tree = ast.parse(source)
    return {
        node.name: node.end_lineno - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in _SPLIT_FUNCTIONS
    }


def test_write_frame_instances_split_into_its_named_phases() -> None:
    """Every phase write_frame_instances' comments named is now its own function."""
    counts = _split_function_line_counts()
    assert set(counts) == set(_SPLIT_FUNCTIONS)


def test_no_function_from_the_split_exceeds_the_line_ceiling() -> None:
    """SC-018: no function produced by splitting write_frame_instances exceeds 150 lines."""
    counts = _split_function_line_counts()
    over_ceiling = {name: n for name, n in counts.items() if n > _LINE_CEILING}
    assert not over_ceiling, f"over the {_LINE_CEILING}-line ceiling: {over_ceiling}"
