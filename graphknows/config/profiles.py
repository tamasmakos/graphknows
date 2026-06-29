"""Memory profile seam: maps MemoryMode to capability decisions.

Architectural rule
------------------
- ``llm_free`` mode: taxonomy drives extraction (GLiNER entities + spaCy
  dependency-pattern relations). Retrieval is deterministic.
- ``llm_assisted`` mode: ontology drives extraction (DSPy LLM relation
  extraction constrained to a configured RDF ontology). Retrieval is agentic.

This module is the **single source of truth** for mode-to-capability mapping.
All other modules MUST derive their branching from ``get_profile(settings)``
rather than inspecting ``settings.mode`` directly.

Frozen contracts (downstream agents depend on these — do NOT change):
- ``MemoryProfile`` fields:
    - ``extraction_uses_llm: bool``
    - ``retrieval: Literal["deterministic", "agentic"]``
    - ``schema_kind: Literal["taxonomy", "ontology"]``
- ``get_profile(settings: GraphKnowsSettings) -> MemoryProfile``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from graphknows.config import GraphKnowsSettings


@dataclass(frozen=True)
class MemoryProfile:
    """Immutable capability profile derived from the active MemoryMode.

    Attributes:
        extraction_uses_llm: True when the LLM extraction path is active
            (``llm_assisted``); False for the deterministic path
            (``llm_free``). Use this to gate DSPy relation extraction.
        retrieval: Which retriever implementation to use.
            ``"deterministic"`` selects ``DETRetriever``;
            ``"agentic"`` selects ``AgenticRetriever``.
        schema_kind: Taxonomy (flat label set, llm_free) or ontology
            (RDF-constrained relation types, llm_assisted).
    """

    extraction_uses_llm: bool
    retrieval: Literal["deterministic", "agentic"]
    schema_kind: Literal["taxonomy", "ontology"]


def get_profile(settings: GraphKnowsSettings) -> MemoryProfile:
    """Return the MemoryProfile for the given settings.

    Selection is driven solely by ``settings.mode``.

    Args:
        settings: The active ``GraphKnowsSettings`` instance.

    Returns:
        A frozen ``MemoryProfile`` describing the active capability set.
    """
    from graphknows.config import MemoryMode

    if settings.mode is MemoryMode.llm_assisted:
        return MemoryProfile(
            extraction_uses_llm=True,
            retrieval="agentic",
            schema_kind="ontology",
        )
    # Default: llm_free
    return MemoryProfile(
        extraction_uses_llm=False,
        retrieval="deterministic",
        schema_kind="taxonomy",
    )
