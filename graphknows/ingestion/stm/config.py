"""STMConfig: configuration dataclass for the STM service layer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class STMConfig:
    """Configuration for STMService.

    Attributes:
        chunk_size: Target character count per text chunk.
        overlap: Character overlap between adjacent chunks.
        entity_confidence: Minimum GLiNER confidence threshold for entities.
        ann_top_k: Number of ANN candidates to fetch before RRF reranking.
        rrf_weights: Multiplicative weights per source type used in RRF fusion.
    """

    chunk_size: int = 512
    overlap: int = 64
    entity_confidence: float = 0.5
    ann_top_k: int = 10
    rrf_weights: dict[str, float] = field(
        default_factory=lambda: {
            "chunk": 1.0,
            "turn": 0.8,
            "entity": 0.9,
        }
    )
