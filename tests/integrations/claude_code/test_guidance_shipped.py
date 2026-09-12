"""The guidance skill and the hook settings block ship with the package (FR-038).

They are read through ``importlib.resources``, so this passes only while the files are
actually part of the installed package rather than of a checkout.
"""

from __future__ import annotations

from pathlib import Path

import graphknows.integrations.claude_code as claude_code
from graphknows.integrations.claude_code import guidance, settings_block

_VERB_FOR_EVENT = {
    "SessionStart": "context",
    "UserPromptSubmit": "recall",
    "PreToolUse": "preview",
    "PostToolUse": "observe",
    "Stop": "remember",
    "PreCompact": "catchup",
}


def test_guidance_says_when_to_record_and_when_to_retrieve() -> None:
    text = guidance().lower()
    assert "## retrieve" in text
    assert "## record" in text


def test_settings_block_wires_every_hook_verb() -> None:
    """Each event runs its own verb: the mapping, not just the vocabulary (FR-031..FR-035)."""
    events = settings_block()["hooks"]
    wired = {
        event: hook["command"]
        for event, matchers in events.items()
        for matcher in matchers
        for hook in matcher["hooks"]
    }
    assert {event: command.rsplit(maxsplit=1)[-1] for event, command in wired.items()} == (
        _VERB_FOR_EVENT
    )


def test_every_hook_command_is_runnable() -> None:
    """The command a user copies must name a module that exists."""
    module = "graphknows.integrations.claude_code"
    for matchers in settings_block()["hooks"].values():
        for matcher in matchers:
            for hook in matcher["hooks"]:
                assert hook["command"].startswith(f"python -m {module} ")
    assert (Path(claude_code.__file__).parent / "__main__.py").is_file()


def test_tool_events_carry_a_matcher() -> None:
    """A tool hook without a matcher fires on every tool, including reads of unrelated files."""
    events = settings_block()["hooks"]
    assert events["PreToolUse"][0]["matcher"] == "Edit|Write"
    assert events["PostToolUse"][0]["matcher"] == "Read|Grep"
