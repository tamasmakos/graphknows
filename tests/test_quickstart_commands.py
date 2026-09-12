"""Every command the quickstart prints must exist and dispatch.

The quickstart is the acceptance for each phase, so a verb it names that no
dispatcher registers — a renamed scenario, a moved entry point — is a broken
acceptance. The check is textual on purpose: the dispatch tables are read as
source rather than imported, so asserting the docs costs no ML import.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = REPO_ROOT / ".claude" / "specs" / "004-neurosymbolic-memory-core" / "quickstart.md"

# `python -m <module>` -> the source registering the verbs that module dispatches.
_DISPATCHERS = {
    "evaluation": "evaluation/__main__.py",
    "evaluation scenario": "evaluation/scenarios.py",
    "graphknows.cli.memory": "graphknows/cli/memory.py",
    "graphknows.integrations.claude_code.hooks": "graphknows/integrations/claude_code/hooks.py",
}
# Tools outside this repository: their existence is the image's business, not ours.
_EXTERNAL = {"git", "echo", "lint-imports"}

_PLACEHOLDER = re.compile(r"<[^>]*>")
_BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def _commands(text: str) -> list[list[str]]:
    """The tokens of every shell command in *text*'s bash blocks, placeholders dropped.

    A `<...>` span is an operator placeholder for the reader to fill in, never
    part of a verb, so it is stripped before tokenising. Continuations, pipes
    and `&&` chains each carry their own command.
    """
    commands = []
    for block in _BASH_BLOCK.findall(text):
        script = _PLACEHOLDER.sub("", block).replace("\\\n", " ")
        for line in re.split(r"[\n|]|&&", script):
            tokens = line.split("#", maxsplit=1)[0].split()
            if tokens:
                commands.append(tokens)
    return commands


def _assert_dispatchable(tokens: list[str]) -> None:
    """Fail unless the repository dispatches *tokens*, or the image owns them."""
    head, rest = tokens[0], tokens[1:]
    if head in _EXTERNAL:
        return
    if head == "make":
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        for target in rest:
            assert re.search(rf"^{re.escape(target)}:", makefile, re.MULTILINE), (
                f"quickstart runs `make {target}`, which the Makefile does not define"
            )
        return
    if head == "pytest":
        for path in (token for token in rest if not token.startswith("-")):
            assert (REPO_ROOT / path).exists(), f"quickstart runs pytest on absent {path}"
        return
    assert head == "python" and rest[:1] == ["-m"], f"quickstart runs unrecognised `{head}`"

    module, verbs = rest[1], [token for token in rest[2:] if not token.startswith("-")]
    if module == "evaluation" and verbs[:1] == ["scenario"]:
        module, verbs = "evaluation scenario", verbs[1:]
    assert module in _DISPATCHERS, f"quickstart runs `python -m {module}`, which has no dispatcher"
    if not verbs:
        return
    dispatch = (REPO_ROOT / _DISPATCHERS[module]).read_text(encoding="utf-8")
    assert f'"{verbs[0]}"' in dispatch, f"`python -m {module}` does not dispatch {verbs[0]!r}"


def test_every_quickstart_command_is_dispatchable() -> None:
    """No phase of the quickstart names a verb this repository cannot run."""
    commands = _commands(QUICKSTART.read_text(encoding="utf-8"))
    assert commands, f"{QUICKSTART.name} has no bash blocks left to check"
    for tokens in commands:
        _assert_dispatchable(tokens)
