"""Unit tests for DialogCapture in isolation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.stm.dialog import DialogCapture, Turn

pytestmark = pytest.mark.asyncio


def _make_capture(extractor=None, embed_fn=None):
    stm = MagicMock()
    vec = MagicMock()
    for attr in [
        "write_dialog_node",
        "write_entity_node",
        "write_turn_node",
        "write_turn_sequence",
        "write_turn_entity_edge",
    ]:
        setattr(stm, attr, AsyncMock(return_value=None))
    vec.write_stm_turn = AsyncMock(return_value=None)
    embed = embed_fn or MagicMock(return_value=[0.1] * 10)
    capture = DialogCapture(stm, vec, extractor=extractor, embed_fn=embed)
    return capture, stm, vec, embed


def _turn(index: int = 0, role: str = "user", text: str = "Hello") -> Turn:
    return Turn(id=f"t{index}", index=index, role=role, text=text)


async def test_write_turn_creates_turn_node():
    capture, stm, vec, _ = _make_capture()
    result = await capture.write_turn("s1", "d1", _turn(0))
    stm.write_turn_node.assert_called_once()
    assert result.turn_id == "t0"


async def test_write_turn_links_previous_turn():
    capture, stm, vec, _ = _make_capture()
    await capture.write_turn("s1", "d1", _turn(0))
    await capture.write_turn("s1", "d1", _turn(1))
    stm.write_turn_sequence.assert_called_once()


async def test_write_turn_no_sequence_on_first_turn():
    capture, stm, vec, _ = _make_capture()
    await capture.write_turn("s1", "d1", _turn(0))
    stm.write_turn_sequence.assert_not_called()


async def test_write_turn_extracts_entities():
    extractor = MagicMock()
    extractor.extract.return_value = ExtractionResult(
        entities=[{"name": "Bob", "type": "PERSON", "score": 0.95}]
    )
    capture, stm, vec, _ = _make_capture(extractor=extractor)
    result = await capture.write_turn("s1", "d1", _turn(0, text="Bob is here"))
    assert result.entity_count == 1
    stm.write_entity_node.assert_called()


async def test_write_turn_embeds_text():
    embed = MagicMock(return_value=[0.5] * 10)
    capture, stm, vec, _ = _make_capture(embed_fn=embed)
    await capture.write_turn("s1", "d1", _turn(0, text="embed me"))
    embed.assert_called_once_with("embed me")
    vec.write_stm_turn.assert_called_once()
