"""DialogWriter: record agent/user conversation turns into STM interaction graph.

DEPRECATED: This module is a compatibility shim. Use graphknows.memory.stm.dialog directly.
"""

from __future__ import annotations

import warnings

# Re-export all public names from the new canonical location.
from graphknows.memory.stm.dialog import (
    DialogCapture,
    DialogResult,
    Participant,
    ToolCall,
    Turn,
)

__all__ = ["DialogCapture", "DialogResult", "DialogWriter", "Participant", "ToolCall", "Turn"]


class DialogWriter(DialogCapture):
    """Compatibility shim for DialogCapture.

    .. deprecated::
        Use :class:`graphknows.memory.stm.dialog.DialogCapture` instead.
    """

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        warnings.warn(
            "DialogWriter is deprecated; use graphknows.memory.stm.dialog.DialogCapture instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
