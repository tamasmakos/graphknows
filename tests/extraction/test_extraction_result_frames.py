"""`ExtractionResult` carries a third section for LLM-decoded frames.

It defaults empty, so the local decoder's result is byte-for-byte what it was
before the field existed (data-model.md §4).
"""

from __future__ import annotations

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult


def test_frames_defaults_empty_and_is_per_instance() -> None:
    first, second = ExtractionResult(), ExtractionResult()

    assert first.frames == []
    assert first.frames is not second.frames

    second.frames.append({"frame": "Giving", "trigger": "gave", "roles": {}})
    assert first.frames == []
