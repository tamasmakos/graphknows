"""A missing spaCy model must fail naming the model and its install command."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "module_path, attr",
    [
        ("graphknows.symbolic.ontology.skos", "_lemmatise"),
    ],
)
def test_missing_model_names_model_and_install_command(
    monkeypatch: pytest.MonkeyPatch, module_path: str, attr: str
) -> None:
    monkeypatch.setenv("GRAPHKNOWS_SPACY_MODEL", "gk_no_such_model")

    import importlib

    from graphknows.exceptions import MissingModelError

    module = importlib.import_module(module_path)

    with pytest.raises(MissingModelError) as exc_info:
        if attr == "_lemmatise":
            module._lemmatise({"dog"})
        else:
            getattr(module, attr)()

    message = str(exc_info.value)
    assert "gk_no_such_model" in message
    assert "python -m spacy download gk_no_such_model" in message


def test_load_spacy_model_raises_missing_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHKNOWS_SPACY_MODEL", "gk_no_such_model")

    from graphknows.exceptions import MissingModelError
    from graphknows.nlp import load_spacy_model

    with pytest.raises(MissingModelError) as exc_info:
        load_spacy_model("gk_no_such_model")

    message = str(exc_info.value)
    assert "gk_no_such_model" in message
    assert "python -m spacy download gk_no_such_model" in message
