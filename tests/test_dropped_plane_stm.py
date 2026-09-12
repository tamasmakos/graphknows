"""The STM plane is gone: no package, no service factory, no closed `Role`.

A plane is only dropped once nothing can still reach it, so this checks the
deleted package *and* that no module left behind imports it, builds its
service, or types a speaker against the four-value enum. What replaces it is
the neutral pair: a session is a `Source`, a turn is a `Segment`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from graphknows import ingestion
from graphknows.memory import Memory
from graphknows.models import message
from graphknows.models.segment import SegmentKind
from graphknows.models.source import Source
from graphknows.settings import GraphKnowsSettings

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_stm_package_is_gone() -> None:
    assert not (REPO_ROOT / "graphknows" / "ingestion" / "stm").exists()
    assert importlib.util.find_spec("graphknows.ingestion.stm") is None


def test_no_module_reaches_the_stm_package_or_its_factory() -> None:
    """Nothing under `graphknows/` imports the package or builds its service."""
    importers = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "graphknows").rglob("*.py")
        if any(
            reached in path.read_text(encoding="utf-8")
            for reached in ("graphknows.ingestion.stm", "build_stm_service")
        )
    ]
    assert importers == []


def test_the_closed_role_enum_is_gone() -> None:
    """A speaker label is a free string now — `Segment.role` is the shape."""
    assert not hasattr(message, "Role")
    assert message.Message(role="Gina", content="hi").role == "Gina"


def test_memory_exposes_no_stm_only_verb() -> None:
    """`redecode` drove the STM handler's decode-failure retry; it went with it."""
    assert not hasattr(Memory, "redecode")


def test_a_session_is_a_source_and_a_turn_a_segment() -> None:
    """The buffered turns of one session map onto the neutral pair (FR-046)."""
    memory = Memory(GraphKnowsSettings())
    source, segments = memory._session_as_source(
        "s1",
        [
            message.Message(role="Gina", content="hi"),
            message.Message(role="assistant", content="hello"),
        ],
    )
    assert isinstance(source, Source)
    assert source.uri == "session:s1"
    assert [segment.kind for segment in segments] == [SegmentKind.turn, SegmentKind.turn]
    assert [segment.role for segment in segments] == ["Gina", "assistant"]
    # Distinct byte ranges into the joined transcript, so no two turns of one
    # session collapse onto the same derived segment id.
    assert len({segment.id for segment in segments}) == 2


def test_ingestion_exports_no_stm_factory() -> None:
    assert "build_stm_service" not in ingestion.__all__
