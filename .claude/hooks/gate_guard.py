"""PreToolUse guard: agents cannot skip the git hooks or weaken the gate configs.

The quality gate for this repo is a pair of git hooks (pre-commit: static checks;
pre-push: the full suite). Git offers no way to make a local hook mandatory —
``--no-verify`` always works. That is fine for a human at a terminal and fatal
for an agent, because an agent that hits a red gate will reach for the flag.

The asymmetry this file creates:

  * an agent's git commands arrive through the Bash tool, which routes through
    this hook, so ``--no-verify`` is denied before git ever sees it;
  * a human typing the same command in their own shell is untouched, because
    this is a Claude Code hook and not a git hook.

Bypassing is therefore possible, manual, and attributable — exactly the property
wanted. Deliberate: there is no environment-variable escape hatch. An env var is
a string, and an agent can set a string.

The second half guards the gate's own configuration. The realistic failure is
not an agent skipping a check; it is an agent "fixing" a red build by loosening
``ruff.toml``, lowering a coverage floor, or adding a ``disable_error_code``.
Pure-gate files are denied outright. ``pyproject.toml`` is escalated to a prompt
instead: it holds the mypy knobs *and* the dependency list, so a hard deny would
block ordinary work.

The file-write check runs on BOTH tool families. Guarding only Edit/Write leaves
``sed -i``, ``cp``, ``tee`` and shell redirection as a trivial way to rewrite the
very configs this exists to protect — the lock has to cover the shell too, or it
is decoration of exactly the kind this repo already had too much of.

Known ceiling: shell coverage is pattern-based, so a determined rewrite (a
here-doc into python, a renamed temp file) still gets through. It stops the
realistic case — an agent reaching for the quickest way to turn a build green —
not an adversary. The human-visible git history is the backstop.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import PurePath

# Files whose entire purpose is to define a gate. Editing one is either a
# deliberate policy change (the human makes it) or a gate being weakened.
LOCKED_FILES = frozenset(
    {
        "ruff.toml",
        ".pre-commit-config.yaml",
        ".importlinter",
        "pytest.ini",
        "sonar-project.properties",
    }
)

# Mixed config: gate knobs live beside dependencies, so prompt rather than deny.
PROMPTED_FILES = frozenset({"pyproject.toml"})

_NO_VERIFY = re.compile(r"(?:^|\s)(--no-verify|-n)(?=\s|$|=)")
_GIT_VERB = re.compile(r"\bgit\b[^|;&]*?\b(commit|push)\b")
_HOOKS_PATH = re.compile(r"\bgit\s+config\b[^|;&]*\bcore\.hooksPath\b", re.IGNORECASE)
# Anchored to the start of a command segment: these are env-var prefixes and
# subcommands, not words. An unanchored match also fires on the word appearing
# inside a commit message, which is a legitimate thing to write.
_HOOK_DISABLE = re.compile(
    r"^\s*(?:SKIP\s*=|PRE_COMMIT_ALLOW_NO_CONFIG\s*=)|\bpre-commit\s+uninstall\b",
    re.IGNORECASE,
)
# Shell writes that would edit a protected file without going through Edit/Write.
_SHELL_WRITE = re.compile(
    r"(?:\bsed\b[^|;&]*-i|\b(?:cp|mv|tee|dd|install)\b|>>?\s*[\"']?[\w./\\-]*)",
)


def _decide(decision: str, reason: str) -> None:
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


def _locked_reason(name: str) -> str:
    return (
        f"Blocked: {name} defines a quality gate. Agents do not edit gate "
        "configs — weakening a check to make a build green is the failure this "
        "rule exists to prevent. Report what the gate flagged instead."
    )


def _classify_path(file_path: str) -> tuple[str, str] | None:
    """Return (decision, reason) for a path, or None when it is unprotected."""
    if not file_path:
        return None
    path = PurePath(file_path.replace("\\", "/"))
    name = path.name
    parts = set(path.parts)

    if name in LOCKED_FILES:
        return ("deny", _locked_reason(name))
    if ".claude" in parts and "hooks" in parts:
        return ("deny", "Blocked: .claude/hooks/ is the enforcement layer itself.")
    if name in PROMPTED_FILES:
        return (
            "ask",
            f"{name} holds gate configuration (mypy error codes, tool settings) "
            "alongside dependencies. Confirm this edit is not loosening a check.",
        )
    return None


def _check_bash(command: str) -> None:
    # `git commit -n` is --no-verify; the flag only matters on commit/push, so
    # an unrelated `-n` (grep -n, sort -n) never trips this.
    for segment in re.split(r"[|;&]{1,2}", command):
        if _GIT_VERB.search(segment) and _NO_VERIFY.search(segment):
            _decide(
                "deny",
                "Blocked: --no-verify skips the repo's quality gate (pre-commit: "
                "lint/types/security; pre-push: full suite + architecture). If the "
                "gate is red, fix what it found or report it — do not bypass it. "
                "Only the human may bypass, from their own terminal.",
            )
        if _HOOK_DISABLE.search(segment):
            _decide("deny", "Blocked: this disables or skips the pre-commit hooks.")
    if _HOOKS_PATH.search(command):
        _decide("deny", "Blocked: re-pointing core.hooksPath disables the repo's git hooks.")

    # A shell write aimed at a protected path is the same act as an Edit.
    if not _SHELL_WRITE.search(command):
        return
    for token in re.findall(r"[\w./\\-]+", command):
        verdict = _classify_path(token)
        if verdict is not None:
            decision, reason = verdict
            if decision == "deny":
                _decide("deny", f"{reason} (Attempted via shell rather than Edit.)")
            _decide(decision, reason)


def main() -> None:
    """Deny, prompt for, or allow the pending tool call."""
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return  # Never block on a malformed payload — fail open, not closed.

    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}

    if tool in ("Bash", "PowerShell"):
        _check_bash(str(tool_input.get("command", "")))
    elif tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        raw = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        verdict = _classify_path(str(raw))
        if verdict is not None:
            _decide(*verdict)


if __name__ == "__main__":
    main()
