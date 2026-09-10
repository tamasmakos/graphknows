"""The guidance skill and the hook settings block ship with the package (FR-038).

They are read through ``importlib.resources``, so this passes only while the files are
actually part of the installed package rather than of a checkout.
"""

from __future__ import annotations

from graphknows.integrations.claude_code import guidance, settings_block

_VERBS = {"context", "recall", "preview", "observe", "catchup"}


def test_guidance_says_when_to_record_and_when_to_retrieve() -> None:
    text = guidance().lower()
    assert "## retrieve" in text
    assert "## record" in text


def test_settings_block_wires_every_hook_verb() -> None:
    events = settings_block()["hooks"]
    commands = [
        hook["command"]
        for matchers in events.values()
        for matcher in matchers
        for hook in matcher["hooks"]
    ]
    assert len(commands) == len(events)
    assert {command.rsplit(maxsplit=1)[-1] for command in commands} == _VERBS
