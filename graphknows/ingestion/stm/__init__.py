"""graphknows.ingestion.stm — Short-Term Memory service layer.

Re-exports the primary public API for STM operations.
"""

from __future__ import annotations

from graphknows.ingestion.stm.config import STMConfig
from graphknows.ingestion.stm.service import STMService

__all__ = ["STMConfig", "STMService"]
