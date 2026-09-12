"""The frame plane is gone: FrameNet is a concept+predicate emitter and nothing more.

A plane is only dropped once nothing can still reach it, so this checks the
deleted modules *and* that no module left behind imports them, that the golden
schema declares no frame-specific type or graph-ranking column, and that no
writer writes one. What replaces the plane is FR-023's shape: a frame is a
concept, its core frame elements are predicates on that concept.

The old ``graph_store.py`` is not scanned here: it is the old core's write path
against a schema that no longer declares any of these types, and it goes whole
at T037 rather than being carved up twice.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from graphknows.models.symbols import ConceptRef, PredicateRef
from graphknows.settings import GraphKnowsSettings
from graphknows.storage.arcadedb import _schema
from graphknows.symbolic.framenet import emitter

REPO_ROOT = Path(__file__).resolve().parents[1]

DROPPED_MODULES = (
    "graphknows.symbolic.framenet.fe",
    "graphknows.symbolic.framenet.framenet",
    "graphknows.symbolic.framenet.index",
    "graphknows.symbolic.framenet.relations",
    "graphknows.symbolic.framenet.srl",
    "graphknows.channels.frames",
    "graphknows.channels.frame_facts",
    "graphknows.ranking.frame_boost",
    "graphknows.ingestion.extraction.relations.frame_srl",
)

# What a frame-specific type or a graph-ranking column is called in DDL. `FE` is
# matched with word boundaries via the split below, not as a substring.
DROPPED_DECLARATIONS = (
    "FRAME",
    "SEMTYPE",
    "FRAME_INSTANCE",
    "PLAYS_ROLE",
    "PAGERANK",
    "COMMUNITY_ID",
    "GRAPH_EMBEDDING",
)


@pytest.fixture(autouse=True)
def _no_corpus_cache():
    """The emitter memoises per pack name; a stub corpus must not leak between tests."""
    emitter.concepts.cache_clear()
    emitter.predicates.cache_clear()
    yield
    emitter.concepts.cache_clear()
    emitter.predicates.cache_clear()


@pytest.fixture
def stub_corpus(monkeypatch):
    """One frame, one core element and one peripheral one — no corpus on disk."""
    frame = SimpleNamespace(
        name="Motion",
        definition="<def-root>A <fen>Theme</fen> moves.</def-root>",
        FE={
            "Theme": SimpleNamespace(coreType="Core", definition="The moving object."),
            "Time": SimpleNamespace(coreType="Peripheral", definition="When it moved."),
        },
    )
    monkeypatch.setattr(emitter, "_framenet", lambda: SimpleNamespace(frames=lambda: [frame]))


def test_frame_plane_modules_are_gone() -> None:
    for module in DROPPED_MODULES:
        assert importlib.util.find_spec(module) is None, f"{module} still exists"


def test_no_module_reaches_a_dropped_frame_module() -> None:
    """Nothing under `graphknows/` imports what the plane left behind."""
    importers = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "graphknows").rglob("*.py")
        if any(module in path.read_text(encoding="utf-8") for module in DROPPED_MODULES)
    ]
    assert importers == []


def test_a_frame_is_a_concept(stub_corpus) -> None:
    """FR-023: a frame is a concept, indexed by its definition like any other."""
    (concept,) = emitter.concepts("dialogue")
    assert isinstance(concept, ConceptRef)
    assert (concept.uri, concept.label, concept.pack) == ("fn:Motion", "Motion", "dialogue")
    assert concept.definition == "A Theme moves."  # markup stripped: it gets embedded


def test_a_core_frame_element_is_a_predicate_on_that_concept(stub_corpus) -> None:
    """Only core elements earn a predicate; the frame is their domain."""
    (predicate,) = emitter.predicates("dialogue")
    assert isinstance(predicate, PredicateRef)
    assert predicate.id == "fn:Motion:Theme"
    assert predicate.canonical == "theme"
    assert predicate.domain == "fn:Motion"


def test_the_golden_schema_declares_no_frame_type_or_ranking_column() -> None:
    """The declaring surface of the new core (T018's DDL), not the old store's."""
    words = {word for _, statement in _schema._CORE_DDL for word in statement.upper().split()}
    words |= {word.split("(")[0] for word in words}
    declared = [name for name in DROPPED_DECLARATIONS if name in words]
    assert declared == []
    assert not [word for word in words if word.endswith(".FE") or word == "FE"]


def test_no_writer_writes_a_frame_type_or_ranking_column() -> None:
    """The writing surface of the new core (T020's writers)."""
    writers = (REPO_ROOT / "graphknows" / "storage" / "arcadedb" / "writers").rglob("*.py")
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in writers
        if any(name in path.read_text(encoding="utf-8") for name in DROPPED_DECLARATIONS)
    ]
    assert offenders == []


def test_settings_declare_no_frame_plane_knob() -> None:
    """A knob for a plane nobody writes is a switch wired to nothing."""
    fields = GraphKnowsSettings.model_fields
    assert [name for name in fields if "frame" in name or name == "enable_fe_type_gate"] == [
        "decode_frames"  # the decoder's own role-filling section, not the plane
    ]
