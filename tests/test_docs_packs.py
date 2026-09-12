"""The packs, hook and merge-gate docs must match the code they describe."""

from __future__ import annotations

import json
from pathlib import Path

from graphknows.integrations.claude_code import settings_block
from graphknows.packs.protocol import DomainPack

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_packs_doc_names_every_protocol_member() -> None:
    """`docs/packs.md` documents the whole `DomainPack` surface, and is indexed."""
    text = (REPO_ROOT / "docs" / "packs.md").read_text(encoding="utf-8")
    for member in (name for name in vars(DomainPack) if not name.startswith("_")):
        assert member in text, f"docs/packs.md does not document DomainPack.{member}"
    for symbol in ("load_packs", "PackConflictError", "CodePack", "AgentPack"):
        assert symbol in text, f"docs/packs.md does not name {symbol}"

    for index in ("docs/README.md", "README.md"):
        assert "packs.md" in (REPO_ROOT / index).read_text(encoding="utf-8"), (
            f"{index} does not link docs/packs.md"
        )


def test_hook_doc_names_every_shipped_event_and_verb() -> None:
    """`docs/integrations.md` documents every event/verb pair the shipped block wires."""
    text = (REPO_ROOT / "docs" / "integrations.md").read_text(encoding="utf-8")
    for event, matchers in settings_block()["hooks"].items():
        assert event in text, f"docs/integrations.md does not document the {event} hook"
        for matcher in matchers:
            for hook in matcher["hooks"]:
                verb = hook["command"].rsplit(maxsplit=1)[-1]
                assert f"`{verb}`" in text, (
                    f"docs/integrations.md does not document the {verb} verb"
                )


def test_merge_gate_doc_names_registered_evaluation_verbs() -> None:
    """CONTRIBUTING.md's merge gate names verbs `python -m evaluation` dispatches."""
    text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    dispatch = (REPO_ROOT / "evaluation" / "__main__.py").read_text(encoding="utf-8")

    assert "## The merge gate" in text, "CONTRIBUTING.md does not document the merge gate"
    for verb in ("panel", "deadweight"):
        assert f"python -m evaluation {verb}" in text, (
            f"CONTRIBUTING.md's merge gate does not name `{verb}`"
        )
        assert json.dumps(verb) in dispatch, (
            f"`python -m evaluation` no longer dispatches {verb} — CONTRIBUTING.md is stale"
        )
    assert "make gate" in text, "CONTRIBUTING.md's merge gate does not name `make gate`"
