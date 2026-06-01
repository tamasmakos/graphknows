"""graphknows.memory.stm — Short-Term Memory service layer.

Re-exports the primary public API for STM operations.
"""

from __future__ import annotations

from graphknows.memory.config import STMConfig
from graphknows.memory.stm.service import STMService

__all__ = ["STMConfig", "STMService"]
