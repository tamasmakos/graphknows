"""Shared store factory functions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from graphknows.config import GraphKnowsSettings
from graphknows.ports.relation_extractor import IRelationExtractor

if TYPE_CHECKING:
    from graphknows.memory.stores.arcadedb.client import ArcadeDBClient
    from graphknows.memory.stores.arcadedb.ltm_store import ArcadeDBLTMStore
    from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore
    from graphknows.memory.stores.arcadedb.vector_store import ArcadeDBVectorStore


def build_relation_extractor(settings: GraphKnowsSettings) -> IRelationExtractor:
    """Build the relation extractor used by graph stores."""
    from graphknows.config.adapters.dspy_relation_extractor import DspyRelationExtractorAdapter

    ontology_file = settings.ontology_file or os.environ.get("GRAPHKNOWS_ONTOLOGY_FILE", "")
    return DspyRelationExtractorAdapter(
        config={
            "llm": {
                "model": settings.extraction_model or settings.llm_model,
                "api_key": settings.llm_api_key or "",
                "api_base": settings.llm_api_base or "",
                "temperature": settings.llm_temperature,
                # Relation extraction needs headroom: reasoning models spend the
                # budget on reasoning_content and truncate the structured triplet
                # output at 2048, producing malformed/empty JSON.
                "max_tokens": max(settings.llm_max_tokens, 4096),
            },
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
    from graphknows.config.profiles import get_profile
    from graphknows.memory.stores.arcadedb.ltm_store import ArcadeDBLTMStore

    profile = get_profile(settings)
    relation_extractor = build_relation_extractor(settings) if profile.extraction_uses_llm else None
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
    """Build an STMService from settings (unconnected — call ``await service.connect()``).

    The service is returned without an open connection so the ArcadeDB client is
    bound to the caller's event loop when ``connect()`` is awaited, not to a
    background loop at build time.
    """
    from graphknows.memory.stm.service import STMService
    from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore
    from graphknows.memory.stores.arcadedb.vector_store import ArcadeDBVectorStore
    from graphknows.memory.stores.embedder import embed_one

    # Share ONE client across the graph + vector stores. Separate connections see
    # each other's freshly-MERGEd CHUNK nodes only after an index-visibility lag,
    # which lets the dual chunk-write paths create duplicate nodes under load.
    client = build_arcadedb_client(settings)
    stm = ArcadeDBSTMStore(client=client)
    vec = ArcadeDBVectorStore(client=client)

    from graphknows.config.profiles import get_profile

    profile = get_profile(settings)
    extractor = _build_entity_extractor(
        "llm_assisted" if profile.extraction_uses_llm else "gliner2"
    )
    rel_extractor = build_relation_extractor(settings) if profile.extraction_uses_llm else None
    return STMService(
        stm,
        vec,
        extractor,
        embed_fn=embed_one,
        rel_extractor=rel_extractor,
    )


def build_promotion_pipeline(settings: GraphKnowsSettings) -> Any:
    """Build an LTMPromotionPipeline from settings."""
    from graphknows.config.profiles import get_profile
    from graphknows.memory.flush import LTMPromotionPipeline
    from graphknows.memory.stores.arcadedb.ltm_store import ArcadeDBLTMStore
    from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore
    from graphknows.memory.stores.arcadedb.vector_store import ArcadeDBVectorStore

    # Share ONE client across the flush stores. The graph-write path
    # (ltm.write_session_subgraph) and the vector-promote path (vec.promote_to_ltm)
    # both MERGE CHUNK by id; on separate connections a node created by one is not
    # immediately visible to the other's CHUNK(id) unique index, so the MERGE
    # stops deduping and writes duplicate, half-un-embedded chunk nodes.

    profile = get_profile(settings)
    client = build_arcadedb_client(settings)
    stm = ArcadeDBSTMStore(client=client)
    vec = ArcadeDBVectorStore(client=client)
    relation_extractor = build_relation_extractor(settings) if profile.extraction_uses_llm else None
    ltm = ArcadeDBLTMStore(client=client, relation_extractor=relation_extractor)
    from graphknows.config.builders.channels import build_channels

    return LTMPromotionPipeline(
        stm,
        vec,
        ltm,
        llm_api_key=settings.llm_api_key or "",
        llm_api_base=settings.llm_api_base or "",
        llm_model=settings.summarization_model or settings.llm_model,
        extraction_uses_llm=profile.extraction_uses_llm,
        channels=build_channels(settings),
    )


def build_corpus_workflow(settings: GraphKnowsSettings) -> Any:
    """Build a CorpusIngestWorkflow from settings."""
    from graphknows.config.profiles import get_profile
    from graphknows.memory.stores.embedder import embed_one
    from graphknows.memory.workflows.corpus_ingest import CorpusIngestWorkflow

    profile = get_profile(settings)
    stm = build_arcadedb_stm_store(settings)
    vec = build_arcadedb_vector_store(settings)
    flush_vec = build_arcadedb_vector_store(settings)
    ltm = build_arcadedb_ltm_store(settings)
    extractor = _build_entity_extractor(
        "llm_assisted" if profile.extraction_uses_llm else "gliner2"
    )
    from graphknows.config.builders.channels import build_channels

    return CorpusIngestWorkflow(
        stm_store=stm,
        vec_store=vec,
        ltm_store=ltm,
        extractor=extractor,
        embed_fn=embed_one,
        llm_api_key=settings.llm_api_key or "",
        llm_api_base=settings.llm_api_base or "",
        rel_extractor=None,
        extraction_uses_llm=profile.extraction_uses_llm,
        flush_vec_store=flush_vec,
        channels=build_channels(settings),
    )
