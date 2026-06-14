"""Smoke tests for graphknows.config.builders store factories."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from graphknows.config import GraphKnowsSettings
from graphknows.config.builders import build_memgraph_store, build_neo4j_store, build_pgvector_store
from graphknows.config.builders.stores import (
    build_corpus_workflow,
    build_relation_extractor,
    build_stm_service,
)


def test_build_neo4j_store_smoke():
    settings = GraphKnowsSettings()
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_driver.return_value = MagicMock()
        store = build_neo4j_store(settings)
        assert store is not None
        assert store._relation_extractor is None


def test_build_neo4j_store_llm_assisted_wires_rdf_relation_extractor(monkeypatch):
    monkeypatch.setenv("RELATION_EXTRACTION_MODE", "llm_assisted")
    settings = GraphKnowsSettings()
    settings.ontology_file = "graphknows/ingestion/ontology/assets/politics/prov-o.ttl"
    with (
        patch("neo4j.GraphDatabase.driver") as mock_driver,
        patch("graphknows.config.builders.stores.build_relation_extractor") as mock_build_rel,
    ):
        mock_driver.return_value = MagicMock()
        mock_build_rel.return_value = MagicMock()
        store = build_neo4j_store(settings)

    assert store._relation_extractor is mock_build_rel.return_value
    mock_build_rel.assert_called_once_with(settings)


def test_build_relation_extractor_imports_dspy_adapter_lazily():
    settings = GraphKnowsSettings()
    settings.ontology_file = "ontology.ttl"
    with patch(
        "graphknows.config.adapters.dspy_relation_extractor.DspyRelationExtractorAdapter"
    ) as mock_adapter:
        mock_adapter.return_value = MagicMock()
        extractor = build_relation_extractor(settings)

    assert extractor is mock_adapter.return_value
    assert mock_adapter.call_args.kwargs["config"]["ontology"]["file"] == "ontology.ttl"


def test_build_pgvector_store_smoke():
    settings = GraphKnowsSettings()
    store = build_pgvector_store(settings)
    assert store is not None
    assert store.dsn == settings.pgvector_dsn


def test_build_memgraph_store_smoke():
    settings = GraphKnowsSettings()
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_driver.return_value = MagicMock()
        store = build_memgraph_store(settings)
        assert store is not None


def test_build_stm_service_default_uses_gliner2():
    settings = GraphKnowsSettings()
    with (
        patch("graphknows.config.builders.stores.build_memgraph_store", return_value=MagicMock()),
        patch("graphknows.config.builders.stores.build_pgvector_store", return_value=MagicMock()),
        patch("graphknows.memory.stm._async_bridge._run"),
        patch("graphknows.memory.extractor.GLiNER2EntityExtractor") as mock_gliner,
        patch("graphknows.ingestion.ontology.rel_extractor.OntologyRelExtractor") as mock_rel,
    ):
        service = build_stm_service(settings)

    assert service._extractor is mock_gliner.return_value
    assert service._rel_extractor is None
    mock_rel.assert_not_called()


def test_build_stm_service_llm_assisted_wires_relation_extractor(monkeypatch):
    monkeypatch.setenv("RELATION_EXTRACTION_MODE", "llm_assisted")
    settings = GraphKnowsSettings()
    with (
        patch("graphknows.config.builders.stores.build_memgraph_store", return_value=MagicMock()),
        patch("graphknows.config.builders.stores.build_pgvector_store", return_value=MagicMock()),
        patch("graphknows.memory.stm._async_bridge._run"),
        patch("graphknows.memory.extractor.GLiNER2EntityExtractor") as mock_gliner,
        patch("graphknows.config.builders.stores.build_relation_extractor") as mock_rel,
    ):
        mock_rel.return_value = MagicMock()
        service = build_stm_service(settings)

    assert service._extractor is mock_gliner.return_value
    assert service._rel_extractor is mock_rel.return_value
    mock_rel.assert_called_once_with(settings)


def test_build_stm_service_off_disables_extraction(monkeypatch):
    monkeypatch.setenv("RELATION_EXTRACTION_MODE", "off")
    settings = GraphKnowsSettings()
    with (
        patch("graphknows.config.builders.stores.build_memgraph_store", return_value=MagicMock()),
        patch("graphknows.config.builders.stores.build_pgvector_store", return_value=MagicMock()),
        patch("graphknows.memory.stm._async_bridge._run"),
        patch("graphknows.memory.extractor.GLiNER2EntityExtractor") as mock_gliner,
        patch("graphknows.ingestion.ontology.rel_extractor.OntologyRelExtractor") as mock_rel,
    ):
        service = build_stm_service(settings)

    assert service._extractor is None
    mock_gliner.assert_not_called()
    mock_rel.assert_not_called()


def test_corpus_workflow_default_passes_gliner2_mode():
    settings = GraphKnowsSettings()
    with (
        patch("graphknows.config.builders.stores.build_memgraph_store", return_value=MagicMock()),
        patch("graphknows.config.builders.stores.build_pgvector_store", return_value=MagicMock()),
        patch("graphknows.config.builders.stores.build_neo4j_store", return_value=MagicMock()),
        patch("graphknows.memory.extractor.GLiNER2EntityExtractor"),
        patch("graphknows.ingestion.ontology.rel_extractor.OntologyRelExtractor") as mock_rel,
    ):
        workflow = build_corpus_workflow(settings)

    assert workflow._relation_extraction_mode == "gliner2"
    assert workflow._rel_extractor is None
    mock_rel.assert_not_called()
