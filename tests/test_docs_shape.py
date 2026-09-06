"""Docs that name code must match the code they name."""

from __future__ import annotations

import re
from pathlib import Path

from graphknows import worth
from graphknows.models.ingest_options import IngestOptions

REPO_ROOT = Path(__file__).resolve().parents[1]
WRITE_PATH = REPO_ROOT / "graphknows" / "ingestion" / "stm" / "ingest.py"


def test_ingest_options_metadata_names_stored_keys() -> None:
    """The metadata docstring names exactly the keys the write path reads (FR-016)."""
    stored = set(re.findall(r'metadata\.get\("([a-z_]+)"', WRITE_PATH.read_text(encoding="utf-8")))
    assert stored, f"no metadata keys found in {WRITE_PATH}"

    docstring = IngestOptions.__doc__ or ""
    metadata_doc = docstring.split("metadata:", 1)[1].split("infer:", 1)[0]
    named = set(re.findall(r"``([a-z_]+)``", metadata_doc))

    assert named == stored, f"docstring names {sorted(named)}, write path stores {sorted(stored)}"
    assert "dropped" in metadata_doc, "docstring must state that every other key is dropped"


def test_worth_gate_docstring_says_opt_in() -> None:
    """The entity-worth gate is opt-in, not a default; the docstring must say so (FR-025)."""
    docstring = worth.__doc__ or ""

    assert "opt-in" in docstring, "docstring must call the gate opt-in"
    assert "the default for ``infer``" not in docstring, "gate is not the default for infer"

    stm_source = (REPO_ROOT / "graphknows" / "server" / "mcp" / "tools" / "stm.py").read_text(
        encoding="utf-8"
    )
    memory_source = (REPO_ROOT / "graphknows" / "memory.py").read_text(encoding="utf-8")
    assert "infer: bool = True" in stm_source, "stm.py must still hardcode infer=True"
    assert "infer: bool = True" in memory_source, (
        "Memory.ingest_memory must still hardcode infer=True"
    )
