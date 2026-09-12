"""Unit tests for the ontology-labelled relex extractor.

Two changes shaped this suite. The model swap (GLiNER2 ->
knowledgator/gliner-relex-large-v1.0) made relations facts rather than votes:
relex returns span-grounded typed head/tail pairs, so its output IS the fact and
none of the old surface-matching corroboration survives.

Then the SVO miner went. It had been doing two jobs — emitting structural triples
AND supplying the predicate vocabulary relex was asked to extract against — and
measured on a live conv-30 graph it produced 28 of 43 relation edges, 25 of which
were ``<speaker> HAS <common noun>`` over a four-word predicate vocabulary. The
relation label space now comes from the ontology properties nearest the chunk
embedding (``extra_relation_labels``), which is the only relation source left.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graphknows.ingestion.extraction.entities.extractor import (
    _BASE_ENTITY_LABELS,
    ExtractionResult,
    GLiNER2EntityExtractor,
    _SpacySchemaMiner,
)

# The ontology property hints ingest._label_hints selects for a chunk:
# {property label -> definition}, nearest-first by cosine against the chunk
# embedding. schema.org labels, because that is what the bundled digest speaks.
_ONTOLOGY_PROPERTIES = {
    "worksFor": "Organizations that the person works for.",
    "knows": "The most generic bi-directional social/work relation.",
}


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


def _mock_extractor(relex_relations=None, relex_entities=None):
    """Extractor with a mocked relex model and a hand-built dynamic schema."""
    schema = SimpleNamespace(
        entity_spec={"person": {"description": "dynamic person", "threshold": 0.35}},
        label_to_type={"person": "PERSON"},
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


def _rel(head, relation, tail, score, *, head_type="person", tail_type="occupation"):
    return {
        "head": {"text": head, "type": head_type},
        "tail": {"text": tail, "type": tail_type},
        "relation": relation,
        "score": score,
    }


def test_spacy_schema_miner_mines_the_entity_schema():
    """Real spaCy: the entity schema comes from NER. No relation mining happens."""
    miner = _SpacySchemaMiner()
    schema = miner.build("Caroline: I work as a nurse at Mercy Hospital.")

    assert schema.entity_spec, "spaCy NER should mine at least one entity label"
    assert all(k == k.lower() for k in schema.entity_spec), "schema keys are lowercased"
    assert not hasattr(schema, "svo_triples"), "the SVO miner is gone"
    assert not hasattr(schema, "relation_spec"), "relation labels come from the ontology"


# --- the relation label space is the ontology's ------------------------------


def test_relation_spec_handed_to_relex_is_the_ontology_property_labels():
    """ "the GLiNER relex model with the schema.org labels based on the closest
    ontology module embedding" — the labels ingest._label_hints selected for this
    chunk ARE the relation spec, verbatim and in order.
    """
    extractor, _schema, mock_model = _mock_extractor()

    extractor.extract(
        "Caroline: I work as a nurse.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
    )

    kwargs = mock_model.inference.call_args.kwargs
    assert kwargs["relations"] == ["worksFor", "knows"]
    assert kwargs["return_relations"] is True


def test_a_relation_is_named_after_the_ontology_property_that_matched():
    extractor, _schema, _m = _mock_extractor(
        relex_relations=[_rel("Caroline", "worksFor", "Mercy Hospital", 0.91)]
    )

    result = extractor.extract(
        "Caroline: I work as a nurse at Mercy Hospital.",
        extra_relation_labels=dict(_ONTOLOGY_PROPERTIES),
    )

    r = result.relations[0]
    assert (r["head"], r["relation"], r["tail"]) == ("Caroline", "worksFor", "Mercy Hospital")
    assert r["source"] == "relex"
    assert r["score"] == pytest.approx(0.91)


def test_chunk_with_no_ontology_property_asks_for_no_relations_and_says_why(caplog):
    """An unlogged empty result is indistinguishable from a broken extractor.

    ``{}`` means the selection RAN and nothing matched — that is the warning.
    """
    extractor, _schema, mock_model = _mock_extractor()
    mock_model.inference.return_value = [[{"text": "Caroline", "label": "person", "score": 0.9}]]

    with caplog.at_level(logging.WARNING):
        result = extractor.extract("Caroline: I work as a nurse.", extra_relation_labels={})

    kwargs = mock_model.inference.call_args.kwargs
    assert kwargs["relations"] == []
    assert kwargs["return_relations"] is False
    assert result.relations == []
    assert [e["name"] for e in result.entities] == ["Caroline"], "entities still extract"
    assert "no ontology property" in caplog.text


def test_the_entity_only_pass_does_not_warn_about_a_missing_ontology_property(caplog):
    """``None`` means the caller never asked — that is not the ontology finding nothing.

    The ingest handler is two-pass: pass 1 extracts entities with NO relation
    labels, because the property vocabulary is selected FROM the types pass 1
    finds. Warning there fires once per chunk by design and drowns the case the
    warning exists for. The two must stay distinguishable — "the gate ran and
    found nothing" is a warning, "the gate was not asked to run" is not.
    """
    extractor, _schema, mock_model = _mock_extractor()
    mock_model.inference.return_value = [[{"text": "Caroline", "label": "person", "score": 0.9}]]

    with caplog.at_level(logging.DEBUG):
        extractor.extract("Caroline: I work as a nurse.", extra_relation_labels=None)

    assert "no ontology property" not in caplog.text
    assert "entity-only pass" in caplog.text, "still traceable, just not a warning"


def test_relex_types_its_own_arguments():
    """relex types head and tail natively — richer than the NER/supersense fallback."""
    extractor, _schema, _m = _mock_extractor(
        relex_relations=[_rel("Caroline", "worksFor", "nurse", 0.9)]
    )

    r = extractor.extract(
        "Caroline: I work as a nurse.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
    ).relations[0]

    assert r["head_type"] == "PERSON"  # label_to_type maps the observed schema key
    assert r["tail_type"] == "OCCUPATION"  # base-inventory label, upper-cased


class TestEntityTypeVocabulary:
    """ENTITY.type is the graph's OWN taxonomy — never the ontology's class names.

    Ontology class labels are offered to relex as candidate span labels, and relex
    echoes back whichever it picked. Those echoes were being stored verbatim as
    ENTITY.type. Censused on the full conv-30 ingest (369 turns, database
    ``mem_lg30v4``): 1256 entities over 41 distinct types against a 763/10
    baseline, led by EMOTION 275, ACTIVITY 202, RELATIONSHIP 113, POSSESSION 106,
    GOAL 96 — every one of them a class label of the ontology that run configured
    (``ONTOLOGY_CLASS`` carries ``…/ontology/personal#Emotion`` URIs), and all
    1256 MENTIONS-linked, i.e. minted on the ENTITY path rather than as relation
    endpoints. The schema.org default leaks the same way, more quietly:
    BANKACCOUNT, HOWTO, WHOLESALESTORE, DISCUSSIONFORUMPOSTING.

    A type nothing else in the graph speaks is worse than no type: ENTITY.type is
    written first-write-wins, it conditions relation-property selection and the
    ontology class matcher, and two vocabularies in one column make it
    unqueryable. Untyped is the recoverable outcome — ``coalesce`` lets a later
    NER write fill it in.
    """

    def test_an_ontology_class_hint_does_not_become_the_entity_type(self):
        extractor, _schema, mock_model = _mock_extractor(
            relex_entities=[{"text": "the studio", "label": "WholesaleStore", "score": 0.9}]
        )

        (ent,) = extractor.extract(
            "Gina: I opened the studio.", extra_entity_labels=["WholesaleStore"]
        ).entities

        labels = mock_model.inference.call_args.kwargs["labels"]
        assert "WholesaleStore" in labels, "the hint still steers what relex looks for"
        assert ent["name"] == "studio", "the span it found is still an entity"
        assert ent["type"] == "", "an ontology class name is not a graph type"

    def test_an_observed_ner_label_still_types_the_entity(self):
        extractor, _schema, _m = _mock_extractor()

        (ent,) = extractor.extract("Caroline: I am here.").entities

        assert ent["type"] == "PERSON"

    def test_a_base_inventory_label_still_types_the_entity(self):
        extractor, _schema, _m = _mock_extractor(
            relex_entities=[{"text": "nurse", "label": "occupation", "score": 0.9}]
        )

        (ent,) = extractor.extract("Caroline: I work as a nurse.").entities

        assert ent["type"] == "OCCUPATION"

    def test_relation_endpoints_keep_the_ontology_label(self):
        """The endpoint type is a GATE input, not a stored vocabulary decision.

        ``ingest._range_admits`` maps it onto a profile class to decide whether
        ``owns`` may range over it, so narrowing it here would switch off the one
        violation that gate exists to catch.
        """
        extractor, _schema, _m = _mock_extractor(
            relex_relations=[
                _rel("Caroline", "owns", "the studio", 0.9, tail_type="WholesaleStore")
            ]
        )

        r = extractor.extract(
            "Caroline: I own the studio.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
        ).relations[0]

        assert r["tail_type"] == "WHOLESALESTORE"


def test_subthreshold_relex_relation_is_dropped():
    extractor, _schema, _m = _mock_extractor(
        relex_relations=[_rel("Caroline", "worksFor", "nurse", 0.2)]
    )

    assert (
        extractor.extract(
            "Caroline: I work as a nurse.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
        ).relations
        == []
    )


def test_pronoun_arguments_are_dropped():
    """relex has no speaker resolution, so a bare pronoun has no recoverable referent.

    Keeping them would MERGE every speaker's facts onto one ENTITY node called "I".
    """
    extractor, _schema, _m = _mock_extractor(relex_relations=[_rel("I", "worksFor", "nurse", 0.99)])

    assert (
        extractor.extract(
            "Caroline: I work as a nurse.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
        ).relations
        == []
    )


def test_pronoun_entities_are_dropped():
    extractor, _schema, _m = _mock_extractor(
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
    extractor, _schema, mock_model = _mock_extractor()

    extractor.extract("Caroline: I work as a nurse.")

    kwargs = mock_model.inference.call_args.kwargs
    assert "person" in kwargs["labels"]  # the observed label
    for base in ("occupation", "organization", "event"):
        assert base in kwargs["labels"], f"base label {base!r} must reach relex"
    assert len(kwargs["labels"]) == len(set(kwargs["labels"])), "labels must be deduped"


def test_relex_receives_the_pair_and_relation_thresholds():
    extractor, _schema, mock_model = _mock_extractor()

    extractor.extract(
        "Caroline: I work as a nurse.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
    )

    kwargs = mock_model.inference.call_args.kwargs
    assert kwargs["adjacency_threshold"] > 0, "unbounded entity pairing invents relations"
    assert kwargs["relation_threshold"] > 0


# --- deixis: the speaker is the missing end of every first-person fact -------


class TestDeixisResolvedRelationSurface:
    """Relations are read off a surface where "I"/"my" have become the speaker.

    LoCoMo speakers state facts about themselves in the first person ("I opened
    an online clothing store", "My collection has twenty pieces"). relex needs
    two entity SPANS to emit a relation and a pronoun is not a span, so the
    SUBJECT of nearly every stated fact is invisible and the relation has only
    one end. Measured on 40 real conv-30 turns, same model/labels/threshold:
    3 relations from the raw text, 19 from the resolved surface.

    The stored chunk text is NOT rewritten — the repo guarantees it stays the
    original, and character offsets carry alias candidates and overlap
    resolution. So entities come off the original text and relations off the
    surface; they decouple because relation heads/tails are matched by NAME.
    """

    def _extractor(self, relex_relations=None):
        mock_model = MagicMock()
        mock_model.inference.return_value = (
            [[{"text": "Gina", "label": "person", "score": 0.9}]],
            [relex_relations or []],
        )
        return GLiNER2EntityExtractor(
            model_loader=lambda: mock_model, schema_miner=_SpacySchemaMiner()
        ), mock_model

    def test_relations_are_read_off_the_resolved_surface(self):
        extractor, mock_model = self._extractor()

        extractor.extract(
            "Gina: I opened an online clothing store.",
            extra_relation_labels=dict(_ONTOLOGY_PROPERTIES),
        )

        calls = mock_model.inference.call_args_list
        assert len(calls) == 2, "one pass for entities on the original, one for relations"
        relation_call = next(c for c in calls if c.kwargs["return_relations"])
        assert "Gina opened an online clothing store" in relation_call.kwargs["texts"][0]

    def test_entities_are_still_read_off_the_unmodified_text(self):
        """Offsets, alias candidates and overlap resolution anchor on the original."""
        extractor, mock_model = self._extractor()
        text = "Gina: I opened an online clothing store."

        extractor.extract(text, extra_relation_labels=dict(_ONTOLOGY_PROPERTIES))

        entity_call = next(
            c for c in mock_model.inference.call_args_list if not c.kwargs["return_relations"]
        )
        assert entity_call.kwargs["texts"] == [text]

    def test_no_known_speaker_falls_back_to_one_pass_on_the_raw_text(self, caplog):
        extractor, mock_model = self._extractor()

        with caplog.at_level(logging.DEBUG):
            extractor.extract(
                "The store opened in Denver.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
            )

        assert mock_model.inference.call_count == 1
        assert mock_model.inference.call_args.kwargs["texts"] == ["The store opened in Denver."]
        assert "no speaker" in caplog.text, "a silent fallback is indistinguishable from a bug"

    def test_text_without_deixis_costs_no_second_pass(self):
        extractor, mock_model = self._extractor()

        extractor.extract(
            "Gina: Melanie opened the store.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
        )

        assert mock_model.inference.call_count == 1

    def test_a_first_person_fact_reaches_the_graph_with_the_speaker_as_its_head(self):
        extractor, _m = self._extractor(
            relex_relations=[_rel("Gina", "worksFor", "clothing store", 0.5, tail_type="facility")]
        )

        result = extractor.extract(
            "Gina: I opened an online clothing store.",
            extra_relation_labels=dict(_ONTOLOGY_PROPERTIES),
        )

        assert [(r["head"], r["relation"], r["tail"]) for r in result.relations] == [
            ("Gina", "worksFor", "clothing store")
        ]


# --- the threshold ----------------------------------------------------------


def test_relation_threshold_admits_a_correct_ontology_labelled_relation():
    """0.7 came from the model card and discards every correct extraction.

    Measured: relex scores the textbook "Caroline works for Mercy Hospital" under
    the injected label ``worksFor`` at 0.437. The card's floor is calibrated for
    the model's own in-distribution label vocabulary, not for injected ontology
    property names, which score markedly lower.
    """
    from graphknows.ingestion.extraction.entities.extractor import _RELATION_THRESHOLD

    assert _RELATION_THRESHOLD <= 0.437, "the measured score of a CORRECT relation"
    assert _RELATION_THRESHOLD >= 0.3, "at 0.1 sub-span noise appears; at 0.0 it explodes"

    extractor, _schema, _m = _mock_extractor(
        relex_relations=[_rel("Caroline", "worksFor", "Mercy Hospital", 0.437)]
    )
    result = extractor.extract(
        "Caroline works for Mercy Hospital.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
    )
    assert [r["relation"] for r in result.relations] == ["worksFor"]


# --- precision guards -------------------------------------------------------


class TestRelationEndpointGuards:
    """More recall makes precision the problem, and the label set is CLOSED.

    Offered 5 ontology properties on two friends chatting, relex is FORCED to
    choose and emitted both ``Gina spouse Jon`` and ``Gina children Jon``. Neither
    is true. Two structural drops are cheap and safe, and every drop is counted
    and logged — an unlogged swallow makes "nothing matched" and "broken"
    indistinguishable from outside.
    """

    def test_a_self_relation_is_dropped(self):
        """The resolved surface makes "Gina knows Gina" easy to produce."""
        extractor, _schema, _m = _mock_extractor(
            relex_relations=[_rel("Gina", "knows", "gina", 0.9, tail_type="person")]
        )

        assert (
            extractor.extract(
                "Gina: I know myself.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
            ).relations
            == []
        )

    def test_a_determiner_does_not_hide_a_self_relation(self):
        """``the fair -[attendee]-> fair``: two spans of ONE entity, measured live.

        The determiner is not part of the entity — it already forks the ENTITY
        merge key on the entity path (``_strip_determiner``), and on the relation
        path it also walks the two ends past the same-entity check.
        """
        extractor, _schema, _m = _mock_extractor(
            relex_relations=[_rel("the fair", "attendee", "fair", 0.9, head_type="event")]
        )

        assert (
            extractor.extract(
                "Gina: The fair was great.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
            ).relations
            == []
        )

    def test_an_endpoint_that_is_not_an_entity_name_is_dropped(self):
        """``is_graph_entity_name`` is the ONE junk predicate; it applies to both ends."""
        extractor, _schema, _m = _mock_extractor(
            relex_relations=[
                _rel("Gina! I", "knows", "Jon", 0.9, tail_type="person"),
                _rel("Gina", "knows", "23 July 2023", 0.9, tail_type="date"),
                _rel("Gina", "knows", "Jon", 0.9, tail_type="person"),
            ]
        )

        result = extractor.extract(
            "Gina: Thanks, Jon!", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
        )

        assert [(r["head"], r["tail"]) for r in result.relations] == [("Gina", "Jon")]

    def test_a_speaker_initial_clause_reduces_to_the_speaker(self):
        """The substitution defeats the clause-fragment rule, which keys on PRONOUNS.

        ``is_graph_entity_name`` drops "I'm over the moon because" because its
        first token is a pronoun. Rewrite the pronoun to "Gina" and the same span
        sails through — measured on the 40-turn probe, "Gina is over the moon
        because" and "Jon is staying positive. Jon" both reached the graph as
        relation endpoints. The referent of such a span is the speaker, exactly as
        for the "Caroline: Thanks, Melanie" shape this reduction already handles.
        """
        from graphknows.ingestion.extraction.entities.extractor import _reduce_speaker_span

        speakers = frozenset({"gina", "jon"})
        assert _reduce_speaker_span("Gina is over the moon because", speakers) == "Gina"
        assert _reduce_speaker_span("Jon is staying positive. Jon", speakers) == "Jon"
        # A proper-name sequence is a name, not a clause, and must survive.
        assert _reduce_speaker_span("Gina Torres", speakers) == "Gina Torres"
        assert _reduce_speaker_span("Gina", speakers) == "Gina"
        # A bare genitive is the same person. Measured live it forked the node
        # ("Jon's owns biz" beside "Jon owns biz") and hid a self-relation
        # outright: "Jon's -[spouse]-> Jon" passed the head!=tail check.
        assert _reduce_speaker_span("Jon's", speakers) == "Jon"
        # ...but only when the genitive IS the whole span: a genitive phrase
        # names something else and keeps its owner.
        assert _reduce_speaker_span("Jon's studio", speakers) == "Jon's studio"
        # Non-dialogue text knows no speakers, so this is the identity function.
        assert _reduce_speaker_span("Gina is over the moon", frozenset()) == (
            "Gina is over the moon"
        )

    def test_every_drop_is_counted_and_logged(self, caplog):
        extractor, _schema, _m = _mock_extractor(
            relex_relations=[
                _rel("Gina", "knows", "Gina", 0.9, tail_type="person"),
                _rel("Gina", "knows", "23 July 2023", 0.9, tail_type="date"),
            ]
        )

        with caplog.at_level(logging.INFO):
            extractor.extract(
                "Gina: Thanks, Jon!", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
            )

        assert "2 of 2" in caplog.text
        assert "self-relation" in caplog.text
        assert "tail is not an entity name" in caplog.text


# --- the SVO miner is gone --------------------------------------------------


class TestSvoMinerIsGone:
    """Nothing but relex may mint a relation.

    Measured on a live conv-30 graph: the relation layer was 43 edges over 763
    entities, SVO owned 28 of them, and 25 of those 28 were
    ``<speaker> HAS <common noun>`` (``Jon HAS back``, ``Jon HAS corner``,
    ``Jon HAS side``) — its entire predicate vocabulary was
    ``{HAS: 25, BE_TO: 1, IS_A: 1, LOSE_AT: 1}``. relex, over 15 edges, used 9
    distinct predicates.
    """

    def _real_miner_extractor(self):
        mock_model = MagicMock()
        mock_model.inference.return_value = ([[]], [[]])
        return GLiNER2EntityExtractor(
            model_loader=lambda: mock_model, schema_miner=_SpacySchemaMiner()
        ), mock_model

    def test_a_possessive_no_longer_mints_a_has_relation(self):
        extractor, _m = self._real_miner_extractor()

        result = extractor.extract(
            "Jon: My back hurts and Caroline's studio is nice.",
            extra_relation_labels=dict(_ONTOLOGY_PROPERTIES),
        )

        assert result.relations == []

    def test_a_copular_clause_no_longer_mints_an_is_a_relation(self):
        extractor, _m = self._real_miner_extractor()

        result = extractor.extract(
            "Caroline: I am a nurse.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES)
        )

        assert result.relations == []

    def test_mined_verb_predicates_never_reach_the_relation_spec(self):
        """WORK_AS/VISIT_WITH were SVO's open-vocabulary labels; the ontology's are all
        relex is offered now."""
        extractor, mock_model = self._real_miner_extractor()

        extractor.extract(
            "Caroline: I work as a nurse and I visited Rome with David.",
            extra_relation_labels=dict(_ONTOLOGY_PROPERTIES),
        )

        assert mock_model.inference.call_args.kwargs["relations"] == ["worksFor", "knows"]


# --- schema / lifecycle -----------------------------------------------------


def test_explicit_relation_schema_overrides_the_ontology_hints():
    """An explicitly passed taxonomy schema (escape hatch) reaches relex."""
    extractor, _schema, mock_model = _mock_extractor()
    extractor._relation_schema = {"created": "one entity created another"}

    extractor.extract("OpenAI built ChatGPT.", extra_relation_labels=dict(_ONTOLOGY_PROPERTIES))

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
    schema = SimpleNamespace(entity_spec={}, label_to_type={}, speakers=frozenset())
    schema_miner = MagicMock()
    schema_miner.build.return_value = schema
    mock_model = MagicMock()
    # relex.inference() -> (entities_per_text, relations_per_text)
    mock_model.inference.return_value = (
        [[{"text": "dance studio", "label": "facility", "score": 0.9}]],
        [[]],
    )
    extractor = GLiNER2EntityExtractor(model_loader=lambda: mock_model, schema_miner=schema_miner)

    result = extractor.extract("No named entities here.")
    assert [e["name"] for e in result.entities] == ["dance studio"]

    mock_model.inference.assert_called_once()
    # The base inventory is what it falls back to.
    assert "facility" in mock_model.inference.call_args.kwargs["labels"]


def test_extract_returns_empty_for_blank_text():
    schema = SimpleNamespace(entity_spec={}, label_to_type={}, speakers=frozenset())
    schema_miner = MagicMock()
    schema_miner.build.return_value = schema
    mock_model = MagicMock()
    extractor = GLiNER2EntityExtractor(model_loader=lambda: mock_model, schema_miner=schema_miner)

    assert extractor.extract("   ") == ExtractionResult()
    mock_model.inference.assert_not_called()


def test_extract_filters_low_confidence_and_blank_entities():
    extractor, _schema, _m = _mock_extractor(
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
        label_to_type={"person": "PERSON"},
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
        label_to_type={"person": "PERSON"},
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
    for label in ("occupation", "animal", "food"):
        assert label in _BASE_ENTITY_LABELS


def test_base_entity_labels_ask_for_no_state_or_process_types():
    """A zero-shot extractor returns whatever part of speech the label names.

    "activity" and "emotion" were in this inventory. Asked for an activity the
    model returns every VERB, asked for an emotion every ADJECTIVE — neither is
    a thing, and neither can be a graph node. Measured on LoCoMo conv-30 they
    produced 906 of 2105 ENTITY nodes (43%), and the most-mentioned "entities"
    in the entire graph were `keep` x56, `going` x32, `make` x29, `really` x30.
    Those then reached the generator's fact-sheet through the frame layer.

    No downstream name filter can repair this: the span really is what the label
    asked for. The gate is not asking.
    """
    for label in ("activity", "emotion", "action", "feeling", "state"):
        assert label not in _BASE_ENTITY_LABELS


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
            relex_entities=[{"text": '"Becoming Nicole"', "label": "work_of_art", "score": 0.9}]
        )

        assert extractor.extract('She read "Becoming Nicole".').entities[0]["name"] == (
            "Becoming Nicole"
        )

    def test_quoted_relation_arguments_are_cleaned(self) -> None:
        extractor, _schema, _m = _mock_extractor(
            relex_relations=[
                _rel("Melanie", "read", '"Becoming Nicole"', 0.9, tail_type="work_of_art")
            ],
        )

        r = extractor.extract(
            'Melanie read "Becoming Nicole".',
            extra_relation_labels={"read": "The Book/Article read by a person."},
        ).relations[0]
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
            build_decoder,
            reset_entity_extractor,
        )
        from graphknows.settings import GraphKnowsSettings

        reset_entity_extractor()
        try:
            s = GraphKnowsSettings()
            assert build_decoder(s) is build_decoder(s)
        finally:
            reset_entity_extractor()

    def test_shared_across_namespaces(self) -> None:
        """Memory is cached PER NAMESPACE; a per-namespace model multiplies the
        footprint by the number of conversations in flight."""
        from graphknows.ingestion.extraction.entities import (
            build_decoder,
            reset_entity_extractor,
        )
        from graphknows.settings import GraphKnowsSettings

        reset_entity_extractor()
        try:
            first = build_decoder(GraphKnowsSettings())
            second = build_decoder(GraphKnowsSettings())
            assert first is second
        finally:
            reset_entity_extractor()

    def test_reset_releases_it(self) -> None:
        from graphknows.ingestion.extraction.entities import (
            build_decoder,
            reset_entity_extractor,
        )
        from graphknows.settings import GraphKnowsSettings

        reset_entity_extractor()
        first = build_decoder(GraphKnowsSettings())
        reset_entity_extractor()
        assert build_decoder(GraphKnowsSettings()) is not first
        reset_entity_extractor()

    def test_concurrent_first_callers_build_once(self) -> None:
        """Double-checked locking: a race here would build two 2.5GB models."""
        import threading

        from graphknows.ingestion.extraction.entities import (
            build_decoder,
            reset_entity_extractor,
        )
        from graphknows.settings import GraphKnowsSettings

        reset_entity_extractor()
        try:
            got: list = []
            threads = [
                threading.Thread(target=lambda: got.append(build_decoder(GraphKnowsSettings())))
                for _ in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert len({id(x) for x in got}) == 1, "concurrent callers built >1 model"
        finally:
            reset_entity_extractor()
