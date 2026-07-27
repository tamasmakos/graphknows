"""Unit tests for the dynamic spaCy-hinted relex extractor.

The model swap (GLiNER2 -> knowledgator/gliner-relex-large-v1.0) changed what the
extractor's relations MEAN, so these tests changed with it. GLiNER2 could only
corroborate an SVO triple, which is why the old suite tested surface-matching a
model relation against a mined one (lenient tails, subject-agreement rescue) —
none of that exists now: relex returns span-grounded typed head/tail pairs, so
its output is the fact. What survives unchanged is the SVO structural keep, which
still carries the identity/possession facts relex has no predicate label for.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graphknows.ingestion.extraction.entities.extractor import (
    _BASE_ENTITY_LABELS,
    ExtractionResult,
    GLiNER2EntityExtractor,
    _SpacySchemaMiner,
)


def test_extraction_result_has_entities_and_relations_fields():
    result = ExtractionResult(
        entities=[{"name": "OpenAI", "type": "ORG", "score": 0.9}],
        relations=[
            {
                "head": "OpenAI",
                "head_type": "ORG",
                "relation": "BUILD",
                "tail": "ChatGPT",
                "tail_type": "PRODUCT",
                "score": 1.0,
            }
        ],
    )

    assert result.entities[0]["name"] == "OpenAI"
    assert result.relations[0]["relation"] == "BUILD"


def _mock_extractor(svo_triples, relex_relations=None, relex_entities=None):
    """Extractor with a mocked relex model and a hand-built dynamic schema."""
    schema = SimpleNamespace(
        entity_spec={"person": {"description": "dynamic person", "threshold": 0.35}},
        relation_spec={"WORK_AS": "dynamic relation"},
        label_to_type={"person": "PERSON"},
        svo_triples=svo_triples,
    )
    schema_miner = MagicMock()
    schema_miner.build.return_value = schema
    schema_miner.build_batch.return_value = [schema]

    ents = (
        relex_entities
        if relex_entities is not None
        else [{"text": "Caroline", "label": "person", "score": 0.95}]
    )
    mock_model = MagicMock()
    # relex.inference() -> (entities_per_text, relations_per_text)
    mock_model.inference.return_value = ([ents], [relex_relations or []])
    extractor = GLiNER2EntityExtractor(
        model_loader=lambda: mock_model,
        schema_miner=schema_miner,
    )
    return extractor, schema, mock_model


def _svo(head, relation, tail, *, head_named, tail_named, verb="work", prep="as"):
    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "verb": verb,
        "prep": prep,
        "head_named": head_named,
        "tail_named": tail_named,
    }


def _rich(head, relation, tail, pattern, *, head_named, tail_named):
    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "pattern": pattern,
        "head_named": head_named,
        "tail_named": tail_named,
    }


def _rel(head, relation, tail, score, *, head_type="person", tail_type="occupation"):
    return {
        "head": {"text": head, "type": head_type},
        "tail": {"text": tail, "type": tail_type},
        "relation": relation,
        "score": score,
    }


def test_spacy_schema_miner_mines_entity_and_svo_schema():
    """Real spaCy: entity schema from NER, relation schema + triples from SVO."""
    miner = _SpacySchemaMiner()
    schema = miner.build("Caroline: I work as a nurse at Mercy Hospital.")

    assert schema.entity_spec, "spaCy NER should mine at least one entity label"
    assert all(k == k.lower() for k in schema.entity_spec), "schema keys are lowercased"
    assert schema.svo_triples, "SVO should mine at least one triple"


# --- relex relations are facts in their own right ---------------------------


def test_relex_relation_is_kept_as_a_fact():
    # No SVO triple at all: the relation exists solely because relex emitted it.
    extractor, _schema, _m = _mock_extractor(
        [], relex_relations=[_rel("Caroline", "WORK_AS", "nurse", 0.91)]
    )

    result = extractor.extract("Caroline: I work as a nurse.")

    r = result.relations[0]
    assert (r["head"], r["relation"], r["tail"]) == ("Caroline", "WORK_AS", "nurse")
    assert r["source"] == "relex"
    assert r["score"] == pytest.approx(0.91)


def test_relex_types_its_own_arguments():
    """relex types head and tail natively — richer than the NER/supersense fallback."""
    extractor, _schema, _m = _mock_extractor(
        [], relex_relations=[_rel("Caroline", "WORK_AS", "nurse", 0.9)]
    )

    r = extractor.extract("Caroline: I work as a nurse.").relations[0]

    assert r["head_type"] == "PERSON"  # label_to_type maps the observed schema key
    assert r["tail_type"] == "OCCUPATION"  # base-inventory label, upper-cased


def test_subthreshold_relex_relation_is_dropped():
    extractor, _schema, _m = _mock_extractor(
        [], relex_relations=[_rel("Caroline", "WORK_AS", "nurse", 0.2)]
    )

    assert extractor.extract("Caroline: I work as a nurse.").relations == []


def test_pronoun_arguments_are_dropped():
    """relex has no speaker resolution, so a bare pronoun has no recoverable referent.

    Keeping them would MERGE every speaker's facts onto one ENTITY node called "I".
    """
    extractor, _schema, _m = _mock_extractor(
        [], relex_relations=[_rel("I", "WORK_AS", "nurse", 0.99)]
    )

    assert extractor.extract("Caroline: I work as a nurse.").relations == []


def test_pronoun_entities_are_dropped():
    extractor, _schema, _m = _mock_extractor(
        [],
        relex_entities=[
            {"text": "I", "label": "person", "score": 0.99},
            {"text": "Caroline", "label": "person", "score": 0.95},
        ],
    )

    names = [e["name"] for e in extractor.extract("Caroline: I am here.").entities]
    assert names == ["Caroline"]


def test_relex_is_given_the_mined_labels_plus_the_base_inventory():
    """relex is zero-shot: it can only emit a type it was handed.

    Passing only the chunk's observed NER labels collapses the type space (a chunk
    whose NER saw only PERSON types every span person), so the base inventory is
    unioned in.
    """
    extractor, schema, mock_model = _mock_extractor([])

    extractor.extract("Caroline: I work as a nurse.")

    kwargs = mock_model.inference.call_args.kwargs
    assert kwargs["relations"] == list(schema.relation_spec.keys())
    assert "person" in kwargs["labels"]  # the observed label
    for base in ("occupation", "organization", "event"):
        assert base in kwargs["labels"], f"base label {base!r} must reach relex"
    assert len(kwargs["labels"]) == len(set(kwargs["labels"])), "labels must be deduped"


def test_relex_receives_the_pair_and_relation_thresholds():
    extractor, _schema, mock_model = _mock_extractor([])

    extractor.extract("Caroline: I work as a nurse.")

    kwargs = mock_model.inference.call_args.kwargs
    assert kwargs["adjacency_threshold"] > 0, "unbounded entity pairing invents relations"
    assert kwargs["relation_threshold"] > 0


# --- the SVO structural keep survives the model swap ------------------------


def test_structural_svo_kept_when_both_named():
    svo = [_svo("Gina", "VISIT", "Rome", head_named=True, tail_named=True, verb="visit", prep="")]
    extractor, *_ = _mock_extractor(svo)  # relex emits no relations

    r = extractor.extract("Gina: I visited Rome.").relations[0]

    assert (r["relation"], r["tail"]) == ("VISIT", "Rome")
    assert r["source"] == "svo"
    assert r["score"] == pytest.approx(0.6)


def test_svo_with_common_noun_object_dropped_when_relex_is_silent():
    """A bare common-noun tail needs the model to vouch for it."""
    svo = [_svo("Caroline", "WORK_AS", "nurse", head_named=True, tail_named=False)]
    extractor, *_ = _mock_extractor(svo)

    assert extractor.extract("Caroline: I work as a nurse.").relations == []


def test_relex_and_svo_do_not_double_count_the_same_fact():
    svo = [_svo("Caroline", "WORK_AS", "nurse", head_named=True, tail_named=True)]
    extractor, *_ = _mock_extractor(
        svo, relex_relations=[_rel("Caroline", "WORK_AS", "nurse", 0.9)]
    )

    result = extractor.extract("Caroline: I work as a nurse.")

    assert len(result.relations) == 1
    assert result.relations[0]["source"] == "relex", "the model's score should win"


def test_rich_copular_kept_structurally():
    """A copular IS_A with a named head survives at the 0.6 structural prior."""
    svo = [_rich("Caroline", "IS_A", "nurse", "copular", head_named=True, tail_named=False)]
    extractor, *_ = _mock_extractor(svo)

    r = extractor.extract("Caroline: I am a nurse.").relations[0]

    assert (r["head"], r["relation"], r["tail"]) == ("Caroline", "IS_A", "nurse")
    assert r["source"] == "svo" and r["score"] == pytest.approx(0.6)


def test_rich_possessive_kept_with_named_head_only():
    svo = [_rich("Caroline", "HAS", "studio", "possessive", head_named=True, tail_named=False)]
    extractor, *_ = _mock_extractor(svo)

    r = extractor.extract("Caroline: That is my studio.").relations[0]
    assert (r["relation"], r["source"]) == ("HAS", "svo")


def test_rich_noun_prep_dropped_unless_both_named():
    """noun_prep is not a named-head pattern; a common-noun tail drops it."""
    svo = [_rich("Paris", "FROM", "friend", "noun_prep", head_named=True, tail_named=False)]
    extractor, *_ = _mock_extractor(svo)

    assert extractor.extract("Paris letter from a friend.").relations == []


def test_rich_copular_dropped_when_head_not_named():
    svo = [_rich("brother", "IS_A", "nurse", "copular", head_named=False, tail_named=False)]
    extractor, *_ = _mock_extractor(svo)

    assert extractor.extract("My brother is a nurse.").relations == []


# --- schema / lifecycle -----------------------------------------------------


def test_explicit_relation_schema_overrides_mined_svo_schema():
    """An explicitly passed taxonomy schema (escape hatch) reaches relex."""
    extractor, _schema, mock_model = _mock_extractor([])
    extractor._relation_schema = {"created": "one entity created another"}

    extractor.extract("OpenAI built ChatGPT.")

    assert mock_model.inference.call_args.kwargs["relations"] == ["created"]


def test_extract_still_runs_when_spacy_mines_no_entity_schema():
    """An empty spaCy observation must NOT skip the model.

    This asserted the opposite until measured: the label set handed to relex is
    the observed spaCy labels UNIONed with _BASE_ENTITY_LABELS, so an empty
    observation is not an empty schema — it is exactly when the base inventory
    earns its keep. Skipping was a cost optimisation that assumed spaCy sees
    what matters, which holds for documents and fails for dialogue. On LoCoMo
    conv-30, 170 of 369 turns (46%) observed no spaCy label and were therefore
    never extracted from at all, including plain factual statements like
    "I'm starting a dance studio".
    """
    schema = SimpleNamespace(
        entity_spec={}, relation_spec={}, label_to_type={}, svo_triples=[], speakers=frozenset()
    )
    schema_miner = MagicMock()
    schema_miner.build.return_value = schema
    mock_model = MagicMock()
    # relex.inference() -> (entities_per_text, relations_per_text)
    mock_model.inference.return_value = (
        [[{"text": "dance studio", "label": "activity", "score": 0.9}]],
        [[]],
    )
    extractor = GLiNER2EntityExtractor(model_loader=lambda: mock_model, schema_miner=schema_miner)

    result = extractor.extract("No named entities here.")
    assert [e["name"] for e in result.entities] == ["dance studio"]

    mock_model.inference.assert_called_once()
    # The base inventory is what it falls back to.
    assert "activity" in mock_model.inference.call_args.kwargs["labels"]


def test_extract_returns_empty_for_blank_text():
    schema = SimpleNamespace(
        entity_spec={}, relation_spec={}, label_to_type={}, svo_triples=[], speakers=frozenset()
    )
    schema_miner = MagicMock()
    schema_miner.build.return_value = schema
    mock_model = MagicMock()
    extractor = GLiNER2EntityExtractor(model_loader=lambda: mock_model, schema_miner=schema_miner)

    assert extractor.extract("   ") == ExtractionResult()
    mock_model.inference.assert_not_called()


def test_extract_filters_low_confidence_and_blank_entities():
    extractor, _schema, _m = _mock_extractor(
        [],
        relex_entities=[
            {"text": "Alice", "label": "person", "score": 0.9},
            {"text": "Bob", "label": "person", "score": 0.2},  # < threshold
            {"text": " ", "label": "person", "score": 0.9},  # blank
        ],
    )

    result = extractor.extract("Alice met Bob.")

    assert result.entities == [{"name": "Alice", "type": "PERSON", "score": 0.9}]


def test_ensure_model_loaded_raises_if_model_loader_returns_none():
    extractor = GLiNER2EntityExtractor(model_loader=lambda: None, schema_miner=MagicMock())

    with pytest.raises(RuntimeError, match="model failed to load"):
        extractor._ensure_model_loaded()


def test_extract_batch_calls_relex_once_per_chunk():
    """No sub-windowing: relex takes the whole chunk in one pass."""
    schema = SimpleNamespace(
        entity_spec={"person": {"description": "p", "threshold": 0.35}},
        relation_spec={},
        label_to_type={"person": "PERSON"},
        svo_triples=[],
    )
    schema_miner = MagicMock()
    schema_miner.build_batch.return_value = [schema, schema]

    mock_model = MagicMock()
    mock_model.inference.return_value = (
        [[{"text": "Caroline", "label": "person", "score": 0.9}]],
        [[]],
    )
    extractor = GLiNER2EntityExtractor(model_loader=lambda: mock_model, schema_miner=schema_miner)

    results = extractor.extract_batch(["chunk one text", "chunk two text"])

    assert len(results) == 2
    assert mock_model.inference.call_count == 2, "one whole-chunk call per chunk"
    assert results[0].entities == [{"name": "Caroline", "type": "PERSON", "score": 0.9}]


def test_extract_batch_survives_a_failing_chunk():
    """One bad chunk must not abort a whole ingest."""
    schema = SimpleNamespace(
        entity_spec={"person": {"description": "p", "threshold": 0.35}},
        relation_spec={},
        label_to_type={"person": "PERSON"},
        svo_triples=[],
    )
    schema_miner = MagicMock()
    schema_miner.build_batch.return_value = [schema, schema]

    mock_model = MagicMock()
    mock_model.inference.side_effect = [
        RuntimeError("boom"),
        ([[{"text": "Caroline", "label": "person", "score": 0.9}]], [[]]),
    ]
    extractor = GLiNER2EntityExtractor(model_loader=lambda: mock_model, schema_miner=schema_miner)

    results = extractor.extract_batch(["bad chunk", "good chunk"])

    assert results[0] == ExtractionResult()
    assert results[1].entities[0]["name"] == "Caroline"


def test_base_entity_labels_cover_conversational_types():
    """spaCy has no tag for the types dialogue is actually made of."""
    for label in ("occupation", "animal", "activity", "emotion"):
        assert label in _BASE_ENTITY_LABELS


class TestSharedModelApiContract:
    """frame_srl reuses the extractor's model, so both must speak ONE API.

    Regression: the model swap left frame_srl calling GLiNER2's
    create_schema()/extract(). It threw AFTER chunks+entities were written, so
    each document reported failure while leaving a half-built graph — three eval
    runs scored 0.73-0.78 against graphs with ZERO relations and ZERO frames.
    """

    def test_fill_roles_uses_inference_not_gliner2_schema_api(self) -> None:
        from unittest.mock import MagicMock

        from graphknows.ingestion.extraction.relations import frame_srl

        model = MagicMock()
        # A relex model has NO create_schema; touching it must raise.
        del model.create_schema
        model.inference.return_value = (
            [[{"text": "Caroline", "label": "Employee", "score": 0.9}]],
            [[]],
        )
        roles, conf = frame_srl._fill_roles(
            "Caroline works as a nurse.", "Being_employed", model, "Caroline", ["Caroline"]
        )
        assert model.inference.called, "must call inference(), not the GLiNER2 schema API"
        assert conf == pytest.approx(0.9)

    def test_fill_roles_handles_entities_only_return(self) -> None:
        """return_relations=False returns the entities list ALONE, not a tuple."""
        from unittest.mock import MagicMock

        from graphknows.ingestion.extraction.relations import frame_srl

        model = MagicMock()
        model.inference.return_value = [[{"text": "Caroline", "label": "Employee", "score": 0.9}]]
        roles, _conf = frame_srl._fill_roles(
            "Caroline works as a nurse.", "Being_employed", model, "Caroline", ["Caroline"]
        )
        assert roles.get("Employee") == ["Caroline"]

    def test_relex_handles_entities_only_return_when_no_predicates_mined(self) -> None:
        """A chunk whose SVO pass mined nothing takes the entities-only branch."""
        schema = SimpleNamespace(
            entity_spec={"person": {"description": "p", "threshold": 0.35}},
            relation_spec={},  # no mined predicates -> return_relations=False
            label_to_type={"person": "PERSON"},
            svo_triples=[],
        )
        schema_miner = MagicMock()
        schema_miner.build.return_value = schema
        mock_model = MagicMock()
        mock_model.inference.return_value = [
            [{"text": "Caroline", "label": "person", "score": 0.9}]
        ]
        extractor = GLiNER2EntityExtractor(
            model_loader=lambda: mock_model, schema_miner=schema_miner
        )

        result = extractor.extract("Caroline said hello.")

        assert result.entities == [{"name": "Caroline", "type": "PERSON", "score": 0.9}]
        assert result.relations == []


class TestSurfaceCleaning:
    """Entity surfaces must not carry enclosing quotes.

    Regression: relex returns the span as written, so a quoted title arrives as
    '"Becoming Nicole"'. The store derives ENTITY.name_norm from this string;
    ArcadeDB strips the quotes persisting name_norm but not name, the fields
    diverge, and a later MERGE cannot find its own record -> DuplicatedKeyException
    on ENTITY[name_norm] -> the caller's WHOLE DOCUMENT fails to ingest.
    """

    def test_clean_surface_strips_enclosing_quotes(self) -> None:
        from graphknows.ingestion.extraction.entities.extractor import _clean_surface

        assert _clean_surface('"Becoming Nicole"') == "Becoming Nicole"
        assert _clean_surface("'Charlotte's Web'") == "Charlotte's Web"
        assert _clean_surface("“Brave”") == "Brave"
        assert _clean_surface("(the studio)") == "the studio"
        assert _clean_surface("nurse.") == "nurse"
        assert _clean_surface("  spaced   out  ") == "spaced out"
        assert _clean_surface('""') == ""

    def test_clean_surface_keeps_internal_punctuation(self) -> None:
        from graphknows.ingestion.extraction.entities.extractor import _clean_surface

        assert _clean_surface("Charlotte's Web") == "Charlotte's Web"
        assert _clean_surface("me-time") == "me-time"

    def test_quoted_entity_is_cleaned_before_it_becomes_a_merge_key(self) -> None:
        extractor, _schema, _m = _mock_extractor(
            [], relex_entities=[{"text": '"Becoming Nicole"', "label": "work_of_art", "score": 0.9}]
        )

        assert extractor.extract('She read "Becoming Nicole".').entities[0]["name"] == (
            "Becoming Nicole"
        )

    def test_quoted_relation_arguments_are_cleaned(self) -> None:
        extractor, _schema, _m = _mock_extractor(
            [],
            relex_relations=[
                _rel("Melanie", "READ", '"Becoming Nicole"', 0.9, tail_type="work_of_art")
            ],
        )

        r = extractor.extract('Melanie read "Becoming Nicole".').relations[0]
        assert r["tail"] == "Becoming Nicole"


class TestSharedModelThreadSafety:
    """The extraction model is shared state; inference() must be serialised.

    Regression: ONE model instance serves every concurrently-ingesting
    conversation AND the frame-SRL pass, all dispatched via asyncio.to_thread.
    Concurrent inference() raised a bare, message-less exception that reached the
    caller as "Error executing tool memory_ingest: " with nothing after the colon,
    losing a document that ingests cleanly on its own.
    """

    def test_inference_is_serialised_across_threads(self) -> None:
        import threading
        from unittest.mock import MagicMock

        from graphknows.ingestion.extraction.entities.gliner_model import _SerialInference

        inner = MagicMock()
        overlap = {"max": 0, "cur": 0}
        guard = threading.Lock()

        def _slow(*_a, **_k):
            with guard:
                overlap["cur"] += 1
                overlap["max"] = max(overlap["max"], overlap["cur"])
            # long enough that unserialised callers would demonstrably overlap
            threading.Event().wait(0.02)
            with guard:
                overlap["cur"] -= 1
            return ([[]], [[]])

        inner.inference.side_effect = _slow
        model = _SerialInference(inner)

        threads = [
            threading.Thread(target=lambda: model.inference(texts=["x"], labels=[]))
            for _ in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert inner.inference.call_count == 6
        assert overlap["max"] == 1, "inference() ran concurrently — the model is not reentrant"

    def test_non_inference_attributes_pass_through(self) -> None:
        from unittest.mock import MagicMock

        from graphknows.ingestion.extraction.entities.gliner_model import _SerialInference

        inner = MagicMock()
        inner.some_attr = 42
        assert _SerialInference(inner).some_attr == 42


class TestExtractorLifetime:
    """The extractor owns a ~2.5GB model: ONE per process, not one per call.

    Regression: Memory.add_memory() built a fresh STMService — and therefore a
    fresh extractor and model — on EVERY ingest call, while Memory itself is
    cached per namespace (up to 64). A single-process run rebuilt the model once
    per DOCUMENT, climbing to ~12.5GB RSS and dying ~28 documents into a
    10-conversation run, and paying ~15s per document to reload a model it had.
    Not a leak — an object-lifetime mistake.
    """

    def test_builder_returns_the_same_instance(self) -> None:
        from graphknows.ingestion.extraction.entities import (
            build_entity_extractor,
            reset_entity_extractor,
        )
        from graphknows.settings import GraphKnowsSettings

        reset_entity_extractor()
        try:
            s = GraphKnowsSettings()
            assert build_entity_extractor(s) is build_entity_extractor(s)
        finally:
            reset_entity_extractor()

    def test_shared_across_namespaces(self) -> None:
        """Memory is cached PER NAMESPACE; a per-namespace model multiplies the
        footprint by the number of conversations in flight."""
        from graphknows.ingestion.extraction.entities import (
            build_entity_extractor,
            reset_entity_extractor,
        )
        from graphknows.settings import GraphKnowsSettings

        reset_entity_extractor()
        try:
            first = build_entity_extractor(GraphKnowsSettings())
            second = build_entity_extractor(GraphKnowsSettings())
            assert first is second
        finally:
            reset_entity_extractor()

    def test_reset_releases_it(self) -> None:
        from graphknows.ingestion.extraction.entities import (
            build_entity_extractor,
            reset_entity_extractor,
        )
        from graphknows.settings import GraphKnowsSettings

        reset_entity_extractor()
        first = build_entity_extractor(GraphKnowsSettings())
        reset_entity_extractor()
        assert build_entity_extractor(GraphKnowsSettings()) is not first
        reset_entity_extractor()

    def test_concurrent_first_callers_build_once(self) -> None:
        """Double-checked locking: a race here would build two 2.5GB models."""
        import threading

        from graphknows.ingestion.extraction.entities import (
            build_entity_extractor,
            reset_entity_extractor,
        )
        from graphknows.settings import GraphKnowsSettings

        reset_entity_extractor()
        try:
            got: list = []
            threads = [
                threading.Thread(
                    target=lambda: got.append(build_entity_extractor(GraphKnowsSettings()))
                )
                for _ in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert len({id(x) for x in got}) == 1, "concurrent callers built >1 model"
        finally:
            reset_entity_extractor()
