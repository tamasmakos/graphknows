"""The frame candidates are selected locally, and the roles come back filled.

A decoder that gets ONE call per chunk (FR-010) cannot wait for a first pass to
tell it which vocabulary to use, and cannot afford a second model pass to fill
frame roles afterwards. Two things have to hold for that:

* the candidates the ingest handler parsed in the batch stage reach the decoder
  in the same call as the entity and relation vocabularies (FR-011, R2/R3);
* the roles that come back are consumed as they are — no local span model is
  reached for, because in that mode there is none to reach for (FR-008).

The FrameNet gates downstream of the fill are covered against the real corpus in
``tests/frames/test_fe_plane.py``; they are stubbed here so this file states the
wiring without needing the corpus installed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.stm.ingest import _STMIngestHandler

pytestmark = pytest.mark.unit


class _Ent:
    def __init__(self, label: str) -> None:
        self.label_ = label


class _Doc:
    """Enough spaCy Doc for the passes under test: named entities, no tokens."""

    def __init__(self, labels: tuple[str, ...] = ()) -> None:
        self.ents = tuple(_Ent(label) for label in labels)

    def __iter__(self) -> Any:
        return iter(())


class _Decoder:
    """A single-call decoder: it answers the seam and has no ``gliner``."""

    def __init__(self, result: ExtractionResult | None = None) -> None:
        self.nlp = object()
        self.calls: list[dict[str, Any]] = []
        self._result = result if result is not None else ExtractionResult()

    def parse(self, _text: str) -> _Doc:
        return _Doc(("PERSON",))

    def extract_batch(
        self,
        texts: list[str],
        extra_entity_labels: Any = None,
        extra_relation_labels: Any = None,
        speakers: Any = (),
        *,
        frame_candidates: Any = (),
    ) -> list[ExtractionResult]:
        self.calls.append(
            {
                "texts": list(texts),
                "extra_relation_labels": extra_relation_labels,
                "frame_candidates": list(frame_candidates),
            }
        )
        return [self._result for _ in texts]


class _Hit:
    """Duck-typed FrameHit — the parsed trigger the batch stage hands on."""

    frame = "Ingestion"
    trigger = "ate"
    offset = 6
    fes: tuple[tuple[str, int, int], ...] = ()


def _handler(decoder: Any, *, enable_frames: bool = True) -> _STMIngestHandler:
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
    return _STMIngestHandler(
        store,
        extractor=decoder,
        embed_fn=lambda _text: [0.1],
        enable_frames=enable_frames,
    )


def _stub_srl(monkeypatch: pytest.MonkeyPatch, hits: list[Any]) -> list[str]:
    """Make the parser answer ``hits`` for every chunk; return the parse log."""
    import graphknows.symbolic.framenet.srl as srl

    parsed: list[str] = []

    def _parse(text: str) -> list[Any]:
        parsed.append(text)
        return hits

    monkeypatch.setattr(srl, "srl_enabled", lambda: True)
    monkeypatch.setattr(srl, "prewarm", lambda texts: len(texts))
    monkeypatch.setattr(srl, "parse", _parse)
    return parsed


def _stub_frame_ontology(monkeypatch: pytest.MonkeyPatch) -> None:
    """Core FEs without the nltk corpus — the shape is what is under test."""
    import graphknows.ingestion.extraction.relations.frame_srl as frame_srl

    monkeypatch.setattr(
        frame_srl,
        "frame_ontology",
        lambda name: {"core_fes": [("Ingestor", "the eater"), ("Ingestibles", "the food")]},
    )


@pytest.mark.asyncio
async def test_the_prewarm_hands_back_what_it_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The batch parse is single-use: its hits are returned, not re-derived later."""
    parsed = _stub_srl(monkeypatch, [_Hit()])
    handler = _handler(_Decoder())

    hits = await handler._prewarm_frame_srl(["Mel ate cake.", "Gina ate soup."])

    assert [[h.frame for h in chunk] for chunk in hits] == [["Ingestion"], ["Ingestion"]]
    assert parsed == ["Mel ate cake.", "Gina ate soup."]


@pytest.mark.asyncio
async def test_frames_off_still_gives_one_empty_hit_list_per_chunk() -> None:
    """Callers index the result positionally, so it is never shorter than the batch."""
    handler = _handler(_Decoder(), enable_frames=False)

    assert await handler._prewarm_frame_srl(["a", "b", "c"]) == [[], [], []]


@pytest.mark.asyncio
async def test_the_candidates_reach_the_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parsed trigger arrives as a closed candidate, with its core FEs."""
    _stub_frame_ontology(monkeypatch)
    decoder = _Decoder()
    handler = _handler(decoder)

    await handler._decode_batch(["Mel ate cake."], None, [[_Hit()]], None)

    assert len(decoder.calls) == 1, "a single-call decoder was called more than once"
    (candidate,) = decoder.calls[0]["frame_candidates"][0]
    assert (candidate.frame, candidate.trigger, candidate.trigger_offset) == ("Ingestion", "ate", 6)
    assert candidate.core_elements == (("Ingestor", "the eater"), ("Ingestibles", "the food"))


def test_a_frame_too_small_to_clear_the_gate_is_never_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One core FE cannot fill two roles, so offering it only buys a wasted section."""
    import graphknows.ingestion.extraction.relations.frame_srl as frame_srl

    monkeypatch.setattr(
        frame_srl, "frame_ontology", lambda name: {"core_fes": [("Ingestor", "the eater")]}
    )
    handler = _handler(_Decoder())

    assert handler._frame_candidates("Mel ate cake.", [_Hit()], None) == []


@pytest.mark.asyncio
async def test_the_vocabularies_are_built_without_a_first_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole file is decoded in one call per chunk, spec included.

    The two-pass path extracts entities, derives the relation vocabulary from
    what it typed, then extracts again. Here the spec is offered with the first
    and only call, derived from spaCy's observation of the chunk (R2).
    """
    from graphknows.ingestion.parsers.text import ParsedFile, TextSegment

    decoder = _Decoder()
    handler = _handler(decoder, enable_frames=False)
    parsed = ParsedFile(
        source_id="f1",
        title="t",
        parser_name="text",
        segments=[TextSegment(text="Mel ate cake.", heading_path="", char_offset=0)],
    )

    await handler.ingest_parsed(parsed, session_id="s1")

    assert len(decoder.calls) == 1, "the relation vocabulary cost a second pass"
    assert decoder.calls[0]["extra_relation_labels"] == [{}]
    assert decoder.calls[0]["frame_candidates"] == [[]]


def _stub_fe_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FrameNet gates, without the corpus. Covered for real in tests/frames."""
    from graphknows.symbolic.framenet import fe

    monkeypatch.setattr(fe, "frame_elements", lambda frame: {})
    monkeypatch.setattr(fe, "violation", lambda frame, filled: None)
    monkeypatch.setattr(fe, "core_sets_satisfied", lambda frame, filled: True)


def test_decoded_roles_are_consumed_without_a_second_model_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fills ride in with the extraction; nothing here asks a model again.

    ``_Decoder`` has no ``gliner`` at all, so a fill attempt would raise rather
    than quietly cost a call — which is the point of reading the capability off
    the attribute instead of a flag.
    """
    _stub_fe_gates(monkeypatch)
    handler = _handler(_Decoder())

    out, nesting = handler._frame_instances(
        "Marta ate cake in London.",
        "chunk1",
        "Marta",
        None,
        [_Hit()],
        None,
        [
            {
                "frame": "Ingestion",
                "trigger": "ate",
                "trigger_offset": 6,
                "roles": {"Ingestor": ["Marta"], "Ingestibles": ["London"]},
            }
        ],
    )

    assert [item["frame"] for item in out] == ["Ingestion"]
    assert out[0]["roles"] == {"Ingestor": ["Marta"], "Ingestibles": ["London"]}
    assert nesting == []


def test_without_the_fills_a_gliner_less_decoder_writes_no_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No local fallback fill: FR-008 forbids reaching for a local model here."""
    _stub_fe_gates(monkeypatch)
    handler = _handler(_Decoder())

    assert handler._frame_instances("Mel ate cake.", "chunk1", "Mel", None, [_Hit()]) == ([], [])
