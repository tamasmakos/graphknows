"""Shared store factory functions."""

from __future__ import annotations

import os
from typing import Any

from graphknows.config import GraphKnowsSettings
from graphknows.config.adapters.ports.relation_extractor import IRelationExtractor
from graphknows.memory.stores.memgraph import MemgraphStore
from graphknows.memory.stores.neo4j import Neo4jStore
from graphknows.memory.stores.pgvector import PgVectorStore


def build_relation_extractor(settings: GraphKnowsSettings) -> IRelationExtractor:
    """Build the relation extractor used by graph stores."""
    from graphknows.config.adapters.dspy_relation_extractor import DspyRelationExtractorAdapter

    ontology_file = getattr(settings, "ontology_file", "") or os.environ.get(
        "GRAPHKNOWS_ONTOLOGY_FILE", ""
    )
    return DspyRelationExtractorAdapter(
        config={
            "llm": {
                "model": settings.extraction_model or settings.llm_model,
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
            },
            "infra": {"openrouter_api_key": settings.openrouter_api_key or ""},
            "ontology": {"file": ontology_file},
        }
    )


def build_neo4j_store(settings: GraphKnowsSettings) -> Neo4jStore:
    """Build a Neo4jStore from settings."""
    relation_extractor = (
        build_relation_extractor(settings)
        if settings.relation_extraction_mode == "llm_assisted"
        else None
    )
    return Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        relation_extractor=relation_extractor,
    )


def build_pgvector_store(settings: GraphKnowsSettings) -> PgVectorStore:
    """Build a PgVectorStore from settings."""
    return PgVectorStore(dsn=settings.pgvector_dsn)


def build_memgraph_store(settings: GraphKnowsSettings) -> MemgraphStore:
    """Build a MemgraphStore from settings."""
    return MemgraphStore(uri=settings.memgraph_uri)


# --- Service factories ---


def _build_entity_extractor(mode: str) -> Any:
    """Return a GLiNER2EntityExtractor, or None when extraction is disabled."""
    if mode == "off":
        return None
    from graphknows.memory.extractor import GLiNER2EntityExtractor

    return GLiNER2EntityExtractor()


def build_stm_service(settings: GraphKnowsSettings) -> Any:
    """Build an STMService from settings."""
    from graphknows.memory.stm._async_bridge import _run
    from graphknows.memory.stm.service import STMService
    from graphknows.memory.stores.embedder import embed_one

    mg = build_memgraph_store(settings)
    pg = build_pgvector_store(settings)
    _run(pg.connect())

    mode = settings.relation_extraction_mode
    extractor = _build_entity_extractor(mode)
    rel_extractor = build_relation_extractor(settings) if mode == "llm_assisted" else None
    return STMService(mg, pg, extractor, embed_fn=embed_one, rel_extractor=rel_extractor)


def build_promotion_pipeline(settings: GraphKnowsSettings) -> Any:
    """Build an LTMPromotionPipeline from settings."""
    from graphknows.memory.flush import LTMPromotionPipeline

    mg = build_memgraph_store(settings)
    pg = build_pgvector_store(settings)
    neo = build_neo4j_store(settings)
    key = settings.openrouter_api_key or ""
    return LTMPromotionPipeline(
        mg,
        pg,
        neo,
        openrouter_key=key,
        relation_extraction_mode=settings.relation_extraction_mode,
    )


def build_corpus_workflow(settings: GraphKnowsSettings) -> Any:
    """Build a CorpusIngestWorkflow from settings."""
    from graphknows.memory.stores.embedder import embed_one
    from graphknows.memory.workflows.corpus_ingest import CorpusIngestWorkflow

    mg = build_memgraph_store(settings)
    pg = build_pgvector_store(settings)
    flush_pg = build_pgvector_store(settings)
    neo = build_neo4j_store(settings)
    mode = settings.relation_extraction_mode
    extractor = _build_entity_extractor(mode)
    key = settings.openrouter_api_key or ""
    return CorpusIngestWorkflow(
        mg_store=mg,
        pg_store=pg,
        neo_store=neo,
        extractor=extractor,
        embed_fn=embed_one,
        openrouter_key=key,
        rel_extractor=None,
        relation_extraction_mode=mode,
        flush_pg_store=flush_pg,
    )
