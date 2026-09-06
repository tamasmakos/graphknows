"""Every gate that declines to write is counted (#139 / B15).

The defect this closes: `_write_relations` built a full report — blocked_domain,
blocked_range, unsupported, pruned_endpoint, conformance, hallucination — and
its caller discarded the return value. The numbers were computed and thrown
away, reaching a per-chunk log line and nothing else. Same shape as the
verifier_score that was computed and never read.

An abstention nobody can count is indistinguishable from an extractor that
found nothing, which is the difference between "the gate is working" and "the
gate ate the graph".
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.stm.ingest import ExtractedChunk, _STMIngestHandler
from graphknows.models.ingest import IngestResult

pytestmark = pytest.mark.unit


def _handler() -> tuple[_STMIngestHandler, MagicMock]:
    store = MagicMock()
    for name in (
        "ensure_session",
        "ensure_file",
        "write_chunk",
        "write_entities_bulk",
        "write_class_matches",
        "write_instance_of",
        "write_temporal",
        "write_frames",
        "write_frame_instances",
    ):
        setattr(store, name, AsyncMock(return_value=None))
    return _STMIngestHandler(store, extractor=None, embed_fn=None), store


class _Extraction:
    def __init__(self, entities: list[dict[str, Any]]) -> None:
        self.entities = entities
        self.relations: list[dict[str, Any]] = []


def test_the_result_carries_a_tally() -> None:
    r = IngestResult(session_id="s", file_id="f", title="", abstentions={"blocked_range": 3})
    assert r.abstentions == {"blocked_range": 3}


def test_the_tally_defaults_to_empty_not_absent() -> None:
    """Empty means 'no gate fired'. That is a different claim from 'no gate ran',
    and a caller must be able to distinguish them without a KeyError."""
    assert IngestResult(session_id="s", file_id="f", title="").abstentions == {}


def test_only_gates_that_fired_are_reported() -> None:
    """A tally full of zeros hides the ones that matter."""
    from collections import defaultdict

    tally: dict[str, int] = defaultdict(int)
    tally["blocked_range"] += 2
    tally["unsupported"] += 0
    reported = {k: v for k, v in tally.items() if v}
    assert reported == {"blocked_range": 2}


def test_the_flush_contract_exposes_it() -> None:
    """flush_memory's return dict is the only place an operator sees these."""
    import inspect

    from graphknows.memory import Memory

    src = inspect.getsource(Memory.flush_memory)
    assert '"abstentions": abstentions' in src, "flush stopped reporting the tally"


def test_the_flush_contract_exposes_entity_coverage() -> None:
    """Isolated-entity count (issue #160) is only ever visible via flush's report."""
    import inspect

    from graphknows.memory import Memory

    src = inspect.getsource(Memory.flush_memory)
    assert '"entity_coverage": entity_coverage' in src, "flush stopped reporting entity_coverage"


def test_flush_reports_per_stage_seconds() -> None:
    """`elapsed_s` alone cannot say WHICH stage took the time.

    A flush covers drain, entity resolution, pagerank, node2vec, louvain and
    topics. When one of them takes 50 minutes, a single total means the next
    person starts by guessing — which is how "the ingest is slow" stayed an
    unanswered question.
    """
    import inspect

    from graphknows.memory import Memory

    src = inspect.getsource(Memory.flush_memory)
    assert '"stage_s": stage_s' in src, "flush stopped reporting per-stage timings"
    # Timed in a `finally`, so a stage that raised still reports how long it ran.
    assert "finally:" in src and "stage_s[name] =" in src, (
        "a failed stage would report no duration — which is when you most want it"
    )


@pytest.mark.asyncio
async def test_untyped_entity_share_is_computable_from_the_tally() -> None:
    """59% of conv-30's entities carried literal ENTITY/EVENT — a type that
    can't satisfy any relation's domain/range — and nothing counted it (#160).
    """
    handler, _store = _handler()
    entities = [
        {"name": "Mel", "type": "PERSON", "confidence": 0.9},
        {"name": "the meeting", "type": "ENTITY", "confidence": 0.5},
        {"name": "the trip", "type": "EVENT", "confidence": 0.5},
        {"name": "something", "type": "", "confidence": 0.5},
    ]
    await handler._write_extracted(
        ExtractedChunk(
            session_id="s1",
            file_id="f1",
            chunk_id="c1",
            text="Mel: I went.",
            seq=0,
            kind="window",
            speaker="Mel",
            ts="",
            embedding=[0.1],
            extraction=_Extraction(entities),
            temporal=[],
            memory_class="ltm",
        )
    )

    assert handler._abstentions["entities"] == 4
    assert handler._abstentions["untyped_entities"] == 3


@pytest.mark.asyncio
async def test_ingest_file_result_carries_the_tally() -> None:
    """`IngestResult` from a file ingest dropped `abstentions` entirely — a
    file ingest reported no tally at all, unlike the message path (#160)."""
    from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
    from graphknows.ingestion.parsers.text import ParsedFile, TextSegment

    class _FakeExtractor:
        def extract(self, *args: Any, **kw: Any) -> ExtractionResult:
            return ExtractionResult(entities=[{"name": "the meeting", "type": "ENTITY"}])

    store = MagicMock()
    for name in (
        "ensure_session",
        "ensure_file",
        "write_chunk",
        "write_entities_bulk",
        "write_class_matches",
        "write_instance_of",
        "write_temporal",
    ):
        setattr(store, name, AsyncMock(return_value=None))
    handler = _STMIngestHandler(store, extractor=_FakeExtractor(), embed_fn=lambda t: [0.1])
    parsed = ParsedFile(
        source_id="f1",
        title="t",
        parser_name="text",
        segments=[TextSegment(text="Mel: hi.", heading_path="", char_offset=0)],
    )
    result = await handler.ingest_parsed(parsed, session_id="s1")

    assert result.abstentions == {
        "entities": 1,
        "untyped_entities": 1,
        "senses_ungrounded": 1,
    }
