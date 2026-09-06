"""The typing pass exists to hand the model a class DEFINITION, per line.

Two claims, both measured on the ontology_kg prototype and both invisible to the
main extractor: a CCO definition types ``research`` as **Act** where a bare label
list types it OCCUPATION, and the line — not the five-turn window — is the unit,
because the line still carries the ``Speaker:`` prefix that resolves "I".

No model is loaded here. The point under test is what reaches the model and what
comes back out, not what the weights do with it.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from graphknows.ingestion.extraction.entities import typing_model
from graphknows.settings import get_settings
from graphknows.symbolic.ontology.loader import OntologyTerm


class _FakeModel:
    """Records every call and replays a scripted answer per line."""

    def __init__(self, answers: dict[str, dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self._answers = answers or {}

    def extract_entities(self, text: str, schema: dict[str, str], **kw: Any) -> dict[str, Any]:
        self.calls.append((text, schema))
        return self._answers.get(text, {"entities": {}})


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch):
    def _install(answers: dict[str, dict[str, Any]] | None = None) -> _FakeModel:
        fake = _FakeModel(answers)
        monkeypatch.setattr(typing_model, "_model", fake)
        return fake

    return _install


SCHEMA = {"Person": "A human being.", "Act": "Something an agent does."}


def test_each_line_is_typed_separately_with_its_speaker_prefix_intact(model) -> None:
    fake = model()

    typing_model.type_lines(["Mel: I paint.", "", "Caroline: I do research."], SCHEMA)

    assert [c[0] for c in fake.calls] == ["Mel: I paint.", "Caroline: I do research."]


def test_the_model_receives_the_definition_not_just_the_label(model) -> None:
    fake = model()

    typing_model.type_lines(["Mel: I paint."], SCHEMA)

    assert fake.calls[0][1]["Act"] == "Something an agent does."


def test_the_models_own_confidence_survives_to_the_caller(model) -> None:
    model({"Mel: I paint.": {"entities": {"Act": [{"text": "paint", "confidence": 0.82}]}}})

    typed = typing_model.type_lines(["Mel: I paint."], SCHEMA)

    assert typed == {"paint": ("Act", 0.82)}


def test_a_bare_string_span_still_types_rather_than_crashing(model) -> None:
    # Older/plainer return shapes carry no confidence; the surface still counts.
    model({"Mel: I paint.": {"entities": {"Act": ["paint"]}}})

    assert typing_model.type_lines(["Mel: I paint."], SCHEMA)["paint"][0] == "Act"


def test_the_first_class_to_claim_a_surface_keeps_it(model) -> None:
    """A later line re-typing the same string is the model contradicting itself."""
    model(
        {
            "Mel: I paint.": {"entities": {"Act": [{"text": "paint", "confidence": 0.9}]}},
            "Mel: a paint.": {"entities": {"Person": [{"text": "paint", "confidence": 0.5}]}},
        }
    )

    typed = typing_model.type_lines(["Mel: I paint.", "Mel: a paint."], SCHEMA)

    assert typed["paint"] == ("Act", 0.9)


def test_one_bad_line_does_not_lose_the_rest_of_the_chunk(monkeypatch) -> None:
    class _Flaky(_FakeModel):
        def extract_entities(self, text, schema, **kw):
            if "boom" in text:
                raise RuntimeError("model exploded")
            return {"entities": {"Act": [{"text": "paint", "confidence": 0.7}]}}

    monkeypatch.setattr(typing_model, "_model", _Flaky())

    assert typing_model.type_lines(["boom", "Mel: I paint."], SCHEMA) == {"paint": ("Act", 0.7)}


def test_an_empty_schema_never_reaches_the_model(model) -> None:
    fake = model()
    assert typing_model.type_lines(["Mel: I paint."], {}) == {}
    assert fake.calls == []


class TestBuildSchema:
    def _terms(self) -> Any:
        class _Idx:
            classes: ClassVar[list[OntologyTerm]] = [
                OntologyTerm(uri="u:Act", label="Act", definition="x" * 300, kind="class"),
                OntologyTerm(uri="u:Family", label="Family", definition="", kind="class"),
            ]

        return _Idx()

    def test_a_definition_is_truncated_to_the_encoder_budget(self) -> None:
        schema = typing_model.build_schema(self._terms(), ["Act"])
        assert len(schema["Act"]) == typing_model.MAX_DEFINITION

    def test_a_class_with_no_definition_falls_back_to_its_own_label(self) -> None:
        assert typing_model.build_schema(self._terms(), ["Family"])["Family"] == "Family"

    def test_a_label_the_ontology_does_not_have_is_skipped(self) -> None:
        assert typing_model.build_schema(self._terms(), ["Nonexistent"]) == {}


def test_the_pass_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPHKNOWS_DEFINITION_TYPING", raising=False)
    get_settings.cache_clear()
    assert typing_model.typing_enabled() is False
    monkeypatch.setenv("GRAPHKNOWS_DEFINITION_TYPING", "1")
    get_settings.cache_clear()
    assert typing_model.typing_enabled() is True
