"""Where the typing pass joins the write path, and what it is allowed to change.

Three boundaries, all of which the ontology_kg prototype crossed and should not
have: the pass may ADD an entity and may sharpen an entity's INSTANCE_OF, but it
may never delete a relex entity or overwrite the ``type`` relex extracted. And
with the flag off the write path must be byte-identical to what it was.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.extraction.entities import typing_model
from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.symbolic.ontology.loader import OntologyTerm


class _StubIndex:
    classes: ClassVar[list[OntologyTerm]] = [
        OntologyTerm(uri="u:Act", label="Act", definition="Something done.", kind="class"),
        OntologyTerm(uri="u:Person", label="Person", definition="A human.", kind="class"),
    ]
    properties: ClassVar[list[OntologyTerm]] = []

    @property
    def property_labels(self) -> list[str]:
        return []


def _handler() -> tuple[_STMIngestHandler, MagicMock]:
    store = MagicMock()
    for name in (
        "ensure_session",
        "ensure_file",
        "write_chunk",
        "write_entities_bulk",
        "write_class_matches",
        "write_groundings",
        "write_temporal",
        "write_frames",
        "write_frame_instances",
    ):
        setattr(store, name, AsyncMock(return_value=None))
    handler = _STMIngestHandler(store, extractor=None, embed_fn=None, ontology_index=_StubIndex())
    return handler, store


class _Extraction:
    def __init__(self, entities: list[dict[str, Any]]) -> None:
        self.entities = entities
        self.relations: list[dict[str, Any]] = []


async def _write(handler: _STMIngestHandler, text: str, entities: list[dict[str, Any]]) -> None:
    await handler._write_extracted(
        session_id="s1",
        file_id="f1",
        chunk_id="c1",
        text=text,
        seq=0,
        kind="window",
        speaker="Mel",
        ts="",
        embedding=[0.1],
        extraction=_Extraction(entities),
        temporal=[],
        memory_class="ltm",
    )


TEXT = "Mel: I do research."
RELEX_ENTITIES = [{"name": "Mel", "type": "Person", "confidence": 0.9}]


@pytest.fixture
def typed(monkeypatch: pytest.MonkeyPatch):
    """Turn the pass on with a scripted model that types 'research' as Act."""
    monkeypatch.setenv("GRAPHKNOWS_DEFINITION_TYPING", "1")

    class _Fake:
        def extract_entities(self, text, schema, **kw):
            return {"entities": {"Act": [{"text": "research", "confidence": 0.83}]}}

    monkeypatch.setattr(typing_model, "_model", _Fake())


@pytest.mark.asyncio
async def test_with_the_flag_off_no_typing_call_is_made(monkeypatch) -> None:
    monkeypatch.delenv("GRAPHKNOWS_DEFINITION_TYPING", raising=False)
    called = MagicMock(side_effect=AssertionError("typing ran with the flag off"))
    monkeypatch.setattr(typing_model, "type_lines", called)

    handler, store = _handler()
    await _write(handler, TEXT, list(RELEX_ENTITIES))

    names = [e["name"] for e in store.write_entities_bulk.await_args.args[2]]
    assert names == ["Mel"]


@pytest.mark.asyncio
async def test_a_surface_only_the_typing_pass_found_becomes_an_entity(typed) -> None:
    handler, store = _handler()
    await _write(handler, TEXT, list(RELEX_ENTITIES))

    written = {e["name"]: e for e in store.write_entities_bulk.await_args.args[2]}
    assert "research" in written
    assert written["research"]["type"] == "Act"


@pytest.mark.asyncio
async def test_the_typing_pass_never_drops_a_relex_entity(typed) -> None:
    handler, store = _handler()
    await _write(handler, TEXT, list(RELEX_ENTITIES))

    assert "Mel" in {e["name"] for e in store.write_entities_bulk.await_args.args[2]}


@pytest.mark.asyncio
async def test_instance_of_carries_the_models_confidence_not_a_flat_one(typed) -> None:
    """The prototype wrote 1.0 for everything, which erases the hedge."""
    handler, store = _handler()
    await _write(handler, TEXT, list(RELEX_ENTITIES))

    rows = {m["name"]: m for m in store.write_groundings.await_args.args[0]}
    assert rows["research"]["uri"] == "u:Act"
    assert rows["research"]["score"] == pytest.approx(0.83)
