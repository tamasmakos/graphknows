"""Claude Code hooks — a stdlib-only leaf that talks to the service over the client SDK.

The integration ships two data files beside the code: the guidance skill telling an
agent when to record and when to retrieve (FR-038), and the hook block to merge into a
Claude Code ``settings.json``. Both are read from the installed package, so an installed
wheel can hand them to an installer without a checkout.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, cast

_DATA = files(__name__)


def guidance() -> str:
    """The shipped skill saying when to record and when to retrieve."""
    return _DATA.joinpath("SKILL.md").read_text(encoding="utf-8")


def settings_block() -> dict[str, Any]:
    """The hook block wiring each Claude Code event to its verb."""
    return cast(
        "dict[str, Any]", json.loads(_DATA.joinpath("settings.json").read_text(encoding="utf-8"))
    )


__all__ = ["guidance", "settings_block"]
