"""Scenario 1.2 / FR-015: the two decoders are interchangeable downstream.

One conversation is driven through both decoders into its own fake store, and
the records that come out are compared. The claim is not that the two produce
the same *facts* — a different producer is exactly what the release gate
measures — but that they produce the same **kinds** of record: entities with
types, relations with a predicate from the offered vocabulary and a per-relation
confidence, frame instances with named roles. A downstream channel handed one of
these cannot tell which decoder minted it, and none of them asks.

What is real here and what stands in:

* the LLM arm is the shipped ``LLMDecoder``, driven against a stubbed provider —
  no network, no local model;
* the local arm's GLiNER2 forward pass is *environment* (a downloaded
  checkpoint), so its spans are canned and mapped into an ``ExtractionResult``
  by the local decoder's own ``_dedupe_entities``/``_build_relation``;
* the spaCy model is environment too, so both arms are handed the same stub
  parse — the difference under test is the decoder, not the parser.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from collections import defaultdict
from typing import Any

import pytest

from graphknows.ingestion.extraction.entities.extractor import (
    ExtractionResult,
    _build_relation,
    _dedupe_entities,
)
from graphknows.ingestion.extraction.llm.decoder import LLMDecoder
from graphknows.ingestion.extraction.llm.schema import FrameCandidate
from graphknows.ingestion.stm.ingest import ExtractedChunk, _STMIngestHandler
from graphknows.settings import GraphKnowsSettings, MemoryMode

pytestmark = pytest.mark.unit

CHUNKS = ("Melanie works for Acme.", "Melanie ate cake.")

#: The closed relation vocabulary chunk 1 is decoded against.
RELATION_SPEC = {"worksFor": "the relation between a person and their employer"}

#: The frame the LOCAL pass selects for chunk 2 in either mode; only the role
#: filling differs (a span model there, the same one call here).
CANDIDATE = FrameCandidate(
    frame="Ingestion",
    trigger="ate",
    trigger_offset=8,
    core_elements=(("Ingestor", "the eater"), ("Ingestibles", "the food")),
)

ROLES = {"Ingestor": ["Melanie"], "Ingestibles": ["cake"]}

#: Per chunk: what the LLM answers, and the vocabularies it is asked against.
_REQUESTS = (
    (
        CHUNKS[0],
        json.dumps(
            {
                "entities": [
                    {"surface": "Melanie", "label": "person"},
                    {"surface": "Acme", "label": "organization"},
                ],
                "relations": [
                    {
                        "head": "Melanie",
                        "predicate": "worksFor",
                        "tail": "Acme",
                        "evidence": CHUNKS[0],
                        "confidence": 0.9,
                    }
                ],
            }
        ),
        RELATION_SPEC,
        (),
    ),
    (
        CHUNKS[1],
        json.dumps(
            {
                "entities": [
                    {"surface": "Melanie", "label": "person"},
                    {"surface": "cake", "label": "food"},
                ],
                "frames": [{"frame": "Ingestion", "trigger": "ate", "roles": ROLES}],
            }
        ),
        None,
        (CANDIDATE,),
    ),
)

#: The spans GLiNER2 would return for the same two chunks, at its own offsets.
_SPANS = (
    [
        {"text": "Melanie", "label": "person", "score": 1.0, "start": 0, "end": 7},
        {"text": "Acme", "label": "organization", "score": 1.0, "start": 18, "end": 22},
    ],
    [
        {"text": "Melanie", "label": "person", "score": 1.0, "start": 0, "end": 7},
        {"text": "cake", "label": "food", "score": 1.0, "start": 12, "end": 16},
    ],
)

#: Written by both arms, differing by construction: ``source`` names the
#: producer, and ``verifier_score`` is the slot the LLM's confidence occupies
#: while the local arm leaves it for the DeBERTa gate (FR-014).
PROVENANCE = ("source", "verifier_score")


class _Doc:
    """Enough spaCy Doc for the passes under test: no tokens, no entities."""

    ents: tuple[Any, ...] = ()

    def __iter__(self) -> Any:
        return iter(())


class _Hit:
    """Duck-typed FrameHit — the parsed trigger the batch stage hands on."""

    frame = "Ingestion"
    trigger = "ate"
    offset = 8
    fes: tuple[tuple[str, int, int], ...] = ()


class _FakeStore:
    """Records every write instead of performing it."""

    def __init__(self) -> None:
        self.calls: dict[str, list[tuple[Any, Any]]] = defaultdict(list)

    def __getattr__(self, name: str) -> Any:
        async def record(*args: Any, **kwargs: Any) -> None:
            self.calls[name].append((args, kwargs))

        return record


class _LocalDecoder:
    """Stands in for GLiNER2EntityExtractor: it carries the local span model."""

    gliner = object()
    relation_verifier = object()
    nlp = object()

    def parse(self, _text: str) -> _Doc:
        return _Doc()


def _stub_provider() -> Any:
    """The provider, stubbed: no network, one canned structured answer per chunk."""
    dspy = pytest.importorskip("dspy")

    class _StubProvider(dspy.LM):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__(model="openrouter/stub/decoder-v1", api_key="stub", cache=False)
            self.reply = "{}"

        def __call__(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> list[str]:
            return [self.reply]

    return _StubProvider()


def _llm_arm() -> tuple[Any, list[ExtractionResult]]:
    """The shipped decoder, one call per chunk, against the stubbed provider."""
    provider = _stub_provider()
    decoder = LLMDecoder(GraphKnowsSettings(GRAPHKNOWS_MODE=MemoryMode.llm_assisted), lm=provider)
    results: list[ExtractionResult] = []
    for text, reply, spec, candidates in _REQUESTS:
        provider.reply = reply
        results.append(decoder.extract(text, None, spec, frame_candidates=candidates))
    return decoder, results


def _local_arm() -> tuple[Any, list[ExtractionResult]]:
    """The local decoder's own span-to-record mapping, over its canned spans."""
    results: list[ExtractionResult] = []
    for index, spans in enumerate(_SPANS):
        entities = _dedupe_entities(list(spans), {})
        relations: list[dict[str, Any]] = []
        if index == 0:
            built, reason = _build_relation("Melanie", "worksFor", "Acme", 0.9, "gliner", entities)
            assert built is not None, reason
            relations.append(built)
        results.append(ExtractionResult(entities=entities, relations=relations))
    return _LocalDecoder(), results


async def _drive(decoder: Any, results: list[ExtractionResult]) -> _FakeStore:
    """Write both chunks through the one funnel every ingest goes through."""
    store = _FakeStore()
    handler = _STMIngestHandler(store, extractor=decoder, embed_fn=lambda _text: [0.1])
    for seq, (text, extraction) in enumerate(zip(CHUNKS, results, strict=True)):
        await handler._write_extracted(
            ExtractedChunk(
                session_id="s1",
                file_id="f1",
                chunk_id=f"chunk-{seq}",
                text=text,
                seq=seq,
                kind="text",
                speaker="",
                ts="",
                embedding=[0.1],
                extraction=extraction,
                temporal=[],
                memory_class="raw_evidence",
                srl_hits=[],
            )
        )
    return store


def _stub_frame_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FrameNet gates and the local role fill, without corpus or model.

    Both are corpus/checkpoint environment; they are stubbed identically for the
    two arms so what is compared is the record each path builds.
    """
    from graphknows.ingestion.extraction.relations import frame_srl
    from graphknows.symbolic.framenet import fe

    monkeypatch.setattr(fe, "frame_elements", lambda frame: {})
    monkeypatch.setattr(fe, "violation", lambda frame, filled: None)
    monkeypatch.setattr(fe, "core_sets_satisfied", lambda frame, filled: True)
    monkeypatch.setattr(
        frame_srl,
        "extract_frame_instances_srl",
        lambda text, speaker, speakers, gliner, hits: [
            frame_srl.FrameInstance(
                frame=CANDIDATE.frame,
                roles={role: list(fills) for role, fills in ROLES.items()},
                match_sim=1.0,
                role_confidence=1.0,
                trigger_offset=CANDIDATE.trigger_offset,
            )
        ],
    )


def _entities(store: _FakeStore) -> list[dict[str, Any]]:
    return [e for args, _ in store.calls["write_entities_bulk"] for e in args[2]]


def _relations(store: _FakeStore) -> list[dict[str, Any]]:
    return [
        {
            "head": write.src_name,
            "relation": write.predicate,
            "tail": write.dst_name,
            "fact": write.fact,
            "chunk_id": write.chunk_id,
            "confidence": write.confidence,
            "source": write.source,
            "src_type": write.src_type,
            "dst_type": write.dst_type,
            "ontology_property": write.ontology_property,
            "verifier_score": write.verifier_score,
            "ontology_blocked": write.ontology_blocked,
            "speaker": write.speaker,
        }
        for (write,), _kwargs in store.calls["write_relation"]
    ]


def _keys(records: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    return [tuple(sorted(record)) for record in records]


@pytest.fixture
def arms(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[Any, list[ExtractionResult]]]:
    """``{arm: (decoder, per-chunk result)}`` over the same conversation."""
    # The resident spaCy model is environment; the same parse is handed to both
    # arms so the comparison is between decoders, not between parsers.
    monkeypatch.setattr(LLMDecoder, "nlp", property(lambda self: object()))
    monkeypatch.setattr(LLMDecoder, "parse", lambda self, text: _Doc())
    return {"local": _local_arm(), "llm": _llm_arm()}


@pytest.fixture
def written(arms: dict[str, tuple[Any, list[ExtractionResult]]]) -> dict[str, _FakeStore]:
    """The conversation written twice — once per arm, into its own store."""
    return {name: asyncio.run(_drive(decoder, r)) for name, (decoder, r) in arms.items()}


@pytest.fixture
def frames(
    arms: dict[str, tuple[Any, list[ExtractionResult]]], monkeypatch: pytest.MonkeyPatch
) -> dict[str, list[dict[str, Any]]]:
    """Chunk 2's frame instances per arm: filled locally there, decoded here."""
    _stub_frame_gates(monkeypatch)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, (decoder, results) in arms.items():
        handler = _STMIngestHandler(
            _FakeStore(), extractor=decoder, embed_fn=None, enable_frames=True
        )
        instances, _nesting = handler._frame_instances(
            CHUNKS[1], "chunk-1", "", None, [_Hit()], {}, results[1].frames or None
        )
        out[name] = instances
    return out


def test_both_decoders_write_entities_with_types(written: dict[str, _FakeStore]) -> None:
    for arm, store in written.items():
        entities = _entities(store)
        assert entities, f"{arm} wrote no entity at all"
        assert all(e.get("type") for e in entities), f"{arm} wrote an untyped entity"
    assert _keys(_entities(written["local"])) == _keys(_entities(written["llm"]))


def test_both_decoders_write_relations_with_an_offered_predicate_and_a_confidence(
    written: dict[str, _FakeStore],
) -> None:
    for arm, store in written.items():
        relations = _relations(store)
        assert relations, f"{arm} wrote no relation at all"
        for rel in relations:
            assert rel["relation"] in RELATION_SPEC, f"{arm} wrote an out-of-vocabulary predicate"
            assert isinstance(rel["confidence"], float), f"{arm} wrote a relation with no score"
    assert _keys(_relations(written["local"])) == _keys(_relations(written["llm"]))


def test_both_decoders_write_frame_instances_with_named_roles(
    frames: dict[str, list[dict[str, Any]]],
) -> None:
    offered = {name for name, _definition in CANDIDATE.core_elements}
    for arm, instances in frames.items():
        assert instances, f"{arm} filled no frame instance"
        for instance in instances:
            assert instance["frame"] == CANDIDATE.frame
            assert set(instance["roles"]) <= offered, f"{arm} named a role off the candidate"
            assert all(instance["roles"].values()), f"{arm} wrote an empty role"
    assert frames["local"] == frames["llm"]


def test_the_records_differ_only_in_who_produced_them(written: dict[str, _FakeStore]) -> None:
    """Provenance is the whole difference; every field a channel reads matches."""
    local, llm = (_relations(written[arm]) for arm in ("local", "llm"))
    assert [{k: v for k, v in r.items() if k not in PROVENANCE} for r in local] == [
        {k: v for k, v in r.items() if k not in PROVENANCE} for r in llm
    ]
    assert [r["source"] for r in local] == ["gliner"]
    assert [r["source"] for r in llm] == ["llm"]
    assert _entities(written["local"]) == _entities(written["llm"])


def test_nothing_downstream_branches_on_which_decoder_produced_the_result() -> None:
    """The funnel names no decoder and no mode — only capabilities (FR-015).

    Identifiers only: ``GLiNER2`` appears in the prose of this class, and a
    docstring is not a branch.
    """
    tree = ast.parse(inspect.getsource(_STMIngestHandler))
    named = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    named |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    named |= {
        alias.asname or alias.name.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom | ast.Import)
        for alias in node.names
    }

    producers = {"LLMDecoder", "GLiNER2EntityExtractor", "MemoryMode", "Decoder", "build_decoder"}
    assert not named & producers, f"the write funnel names a producer: {named & producers}"
