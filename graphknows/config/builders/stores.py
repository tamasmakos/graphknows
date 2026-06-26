"""Shared store factory functions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from graphknows.config import GraphKnowsSettings
from graphknows.config.adapters.ports.relation_extractor import IRelationExtractor

if TYPE_CHECKING:
    from graphknows.memory.stores.arcadedb.client import ArcadeDBClient
    from graphknows.memory.stores.arcadedb.ltm_store import ArcadeDBLTMStore
    from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore
    from graphknows.memory.stores.arcadedb.vector_store import ArcadeDBVectorStore


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
                # Relation extraction needs headroom: reasoning models spend the
                # budget on reasoning_content and truncate the structured triplet
                # output at 2048, producing malformed/empty JSON.
                "max_tokens": max(settings.llm_max_tokens, 4096),
            },
            "infra": {"openrouter_api_key": settings.openrouter_api_key or ""},
            "ontology": {"file": ontology_file},
        }
    )


def build_arcadedb_client(settings: GraphKnowsSettings) -> ArcadeDBClient:
    """Build an ArcadeDBClient from settings (not yet connected)."""
    from graphknows.memory.stores.arcadedb.client import ArcadeDBClient

    return ArcadeDBClient(
        base_url=settings.arcadedb_url,
        user=settings.arcadedb_user,
        password=settings.arcadedb_password,
    )


def build_arcadedb_stm_store(settings: GraphKnowsSettings) -> ArcadeDBSTMStore:
    """Build an ArcadeDBSTMStore (client not yet connected — call ensure_schema after connect)."""
    from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore

    client = build_arcadedb_client(settings)
    return ArcadeDBSTMStore(client=client)


def build_arcadedb_ltm_store(settings: GraphKnowsSettings) -> ArcadeDBLTMStore:
    """Build an ArcadeDBLTMStore from settings."""
    from graphknows.memory.stores.arcadedb.ltm_store import ArcadeDBLTMStore

    relation_extractor = (
        build_relation_extractor(settings)
        if settings.relation_extraction_mode == "llm_assisted"
        else None
    )
    client = build_arcadedb_client(settings)
    return ArcadeDBLTMStore(client=client, relation_extractor=relation_extractor)


def build_arcadedb_vector_store(settings: GraphKnowsSettings) -> ArcadeDBVectorStore:
    """Build an ArcadeDBVectorStore from settings."""
    from graphknows.memory.stores.arcadedb.vector_store import ArcadeDBVectorStore

    client = build_arcadedb_client(settings)
    return ArcadeDBVectorStore(client=client)


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
    from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore
    from graphknows.memory.stores.arcadedb.vector_store import ArcadeDBVectorStore
    from graphknows.memory.stores.embedder import embed_one

    # Share ONE client across the graph + vector stores. Separate connections see
    # each other's freshly-MERGEd CHUNK nodes only after an index-visibility lag,
    # which lets the dual chunk-write paths create duplicate nodes under load.
    client = build_arcadedb_client(settings)
    mg = ArcadeDBSTMStore(client=client)
    pg = ArcadeDBVectorStore(client=client)
    _run(mg._c.connect())
    _run(mg.ensure_schema())
    _run(pg.connect())  # connects client + ensures schema

    mode = settings.relation_extraction_mode
    extractor = _build_entity_extractor(mode)
    rel_extractor = build_relation_extractor(settings) if mode == "llm_assisted" else None
    return STMService(
        mg,
        pg,
        extractor,
        embed_fn=embed_one,
        rel_extractor=rel_extractor,
        spacy_relations=settings.spacy_relations,
    )


def build_promotion_pipeline(settings: GraphKnowsSettings) -> Any:
    """Build an LTMPromotionPipeline from settings."""
    from graphknows.memory.flush import LTMPromotionPipeline
    from graphknows.memory.stores.arcadedb.ltm_store import ArcadeDBLTMStore
    from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore
    from graphknows.memory.stores.arcadedb.vector_store import ArcadeDBVectorStore

    # Share ONE client across the flush stores. The graph-write path
    # (neo.write_session_subgraph) and the vector-promote path (pg.promote_to_ltm)
    # both MERGE CHUNK by id; on separate connections a node created by one is not
    # immediately visible to the other's CHUNK(id) unique index, so the MERGE
    # stops deduping and writes duplicate, half-un-embedded chunk nodes.
    client = build_arcadedb_client(settings)
    mg = ArcadeDBSTMStore(client=client)
    pg = ArcadeDBVectorStore(client=client)
    relation_extractor = (
        build_relation_extractor(settings)
        if settings.relation_extraction_mode == "llm_assisted"
        else None
    )
    neo = ArcadeDBLTMStore(client=client, relation_extractor=relation_extractor)
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

    mg = build_arcadedb_stm_store(settings)
    pg = build_arcadedb_vector_store(settings)
    flush_pg = build_arcadedb_vector_store(settings)
    neo = build_arcadedb_ltm_store(settings)
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
