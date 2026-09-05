"""build_stm_service wires one decoder through, and the sidecar seam is gone.

The relation sidecar was injected as ``rel_extractor=`` at three points — the
factory, STMService and _STMIngestHandler. A leftover parameter anywhere along
that chain would keep the sidecar constructible (and its branch reachable)
after the mode fork moved into ``build_decoder``.
"""

from __future__ import annotations

import inspect

from graphknows.ingestion import build_stm_service
from graphknows.ingestion.stm.ingest import _STMIngestHandler
from graphknows.ingestion.stm.service import STMService


def test_constructors_take_no_rel_extractor():
    assert "rel_extractor" not in inspect.signature(STMService.__init__).parameters
    assert "rel_extractor" not in inspect.signature(_STMIngestHandler.__init__).parameters


def test_no_sidecar_call_survives_in_the_ingest_handler():
    source = inspect.getsource(_STMIngestHandler)
    assert "_rel_extractor" not in source
    assert "extract_relations" not in source


def test_build_stm_service_passes_the_decoder_and_nothing_else(monkeypatch):
    import graphknows.ingestion.extraction.entities as entities
    import graphknows.ingestion.stm.service as service
    import graphknows.storage as storage
    import graphknows.symbolic.ontology as ontology

    decoder = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(storage, "build_graph_store", lambda settings, ns: "store")
    monkeypatch.setattr(entities, "build_decoder", lambda settings: decoder)
    monkeypatch.setattr(ontology, "load_ontology_index", lambda source: None)
    monkeypatch.setattr(
        service,
        "STMService",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs),
    )

    build_stm_service(_Settings(), "ns")

    assert captured["args"] == ("store", decoder)
    assert "rel_extractor" not in captured["kwargs"]


class _Settings:
    """The only settings build_stm_service reads once its collaborators are stubbed."""

    embed_dimensions = 8
    enable_frames = False
    ontology_source = ""
