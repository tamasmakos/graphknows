"""Promotion — the STM->LTM flush, expressed as an ordered step pipeline.

``build_promotion_pipeline(settings, namespace)`` assembles the mode-appropriate
step list; ``PromotionPipeline`` runs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphknows.ingestion.promotion.base import PromotionContext, PromotionStep
from graphknows.ingestion.promotion.pipeline import (
    LTMPromotionPipeline,
    default_promotion_steps,
)

if TYPE_CHECKING:
    from graphknows.settings import GraphKnowsSettings


def build_promotion_pipeline(
    settings: GraphKnowsSettings, namespace: str = ""
) -> LTMPromotionPipeline:
    """Build the mode-appropriate promotion pipeline with connected-on-demand stores.

    The step list is assembled by the pipeline from the mode (see
    :func:`default_promotion_steps`): llm_assisted adds relation extraction,
    community detection, topic summary, and community-metadata write-back; llm_free
    omits them.
    """
    from graphknows.channels.registry import default_channels
    from graphknows.ingestion.extraction.relations import build_relation_extractor
    from graphknows.settings import MemoryMode
    from graphknows.storage import build_arcadedb_client, db_names
    from graphknows.storage.arcadedb.ltm_store import ArcadeDBLTMStore
    from graphknows.storage.arcadedb.stm_store import ArcadeDBSTMStore
    from graphknows.storage.arcadedb.vector_store import ArcadeDBVectorStore

    uses_llm = settings.mode is MemoryMode.llm_assisted

    # Share ONE client across the flush stores. The graph-write path and the
    # vector-promote path both MERGE CHUNK by id; on separate connections a node
    # created by one is not immediately visible to the other's CHUNK(id) unique
    # index, so the MERGE stops deduping and writes duplicate chunk nodes.
    stm_db, ltm_db = db_names(namespace)
    client = build_arcadedb_client(settings)
    stm = ArcadeDBSTMStore(client=client, db=stm_db)
    vec = ArcadeDBVectorStore(client=client, stm_db=stm_db, ltm_db=ltm_db)
    relation_extractor = build_relation_extractor(settings) if uses_llm else None
    ltm = ArcadeDBLTMStore(client=client, relation_extractor=relation_extractor, db=ltm_db)

    return LTMPromotionPipeline(
        stm,
        vec,
        ltm,
        extraction_uses_llm=uses_llm,
        llm_api_key=settings.llm_api_key or "",
        llm_api_base=settings.llm_api_base or "",
        llm_model=settings.summarization_model or settings.llm_model,
        channels=default_channels(settings),
    )


__all__ = [
    "LTMPromotionPipeline",
    "PromotionContext",
    "PromotionStep",
    "build_promotion_pipeline",
    "default_promotion_steps",
]
