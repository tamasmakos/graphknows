"""The old core is gone and only one write path is left (FR-045).

The cutover is structural: the dialogue-era modules are deleted outright rather
than shimmed, flagged or migrated, so the check is that they cannot be imported
and that nothing in the tree still reaches for them. ``bounds.py``, ``nlp.py``
and ``linguistics.py`` are the modules judged sound and deliberately kept.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import graphknows
from graphknows.ingestion.extraction.entities import hygiene

pytestmark = pytest.mark.unit

CORE = Path(graphknows.__file__).resolve().parent

DELETED_MODULES = (
    "graphknows.symbolic.wordnet",
    "graphknows.worth",
)

KEPT_MODULES = ("graphknows.bounds", "graphknows.nlp", "graphknows.linguistics")


@pytest.mark.parametrize("module_path", DELETED_MODULES)
def test_deleted_module_is_gone(module_path: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_path)


@pytest.mark.parametrize("module_path", KEPT_MODULES)
def test_kept_module_survives(module_path: str) -> None:
    assert importlib.import_module(module_path) is not None


def test_no_module_still_imports_the_deleted_core() -> None:
    """A shim would keep the old names alive somewhere; nothing may name them."""
    dead_names = (
        "graphknows.worth",
        "graphknows.symbolic.wordnet",
        "worth_extracting",
    )
    offenders = {
        str(module.relative_to(CORE)): name
        for module in CORE.rglob("*.py")
        for name in dead_names
        if name in module.read_text(encoding="utf-8")
    }
    assert offenders == {}, offenders


def test_worth_gate_left_the_public_surface() -> None:
    assert "worth_extracting" not in graphknows.__all__


def test_hygiene_keeps_no_dialogue_only_rules() -> None:
    """The junk gate judges names, not conversations: no greeting or speaker rule."""
    dialogue_rules = ("_is_backchannel", "_is_vocative_address", "_SPEAKER_COLON_RE")
    assert [rule for rule in dialogue_rules if hasattr(hygiene, rule)] == []
    assert hygiene.is_graph_entity_name("Hey Jon"), "a greeting shape is no longer special-cased"


def test_hygiene_is_the_only_name_gate_left() -> None:
    """The flush-time resolver is gone, so the mint point is the single seam."""
    assert not hygiene.is_graph_entity_name("things"), "junk hubs are dropped before they exist"


def test_flush_has_one_write_path() -> None:
    """No second write path: flush runs no entity resolver and reports no merges."""
    source = (CORE / "memory.py").read_text(encoding="utf-8")
    assert "resolve_entities" not in source
    for retired_key in ("merged_entities", "merge_rate", "entities_dropped"):
        assert retired_key not in source, f"flush still reports {retired_key}"
