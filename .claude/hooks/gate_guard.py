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
``ruff.toml``, lowering the coverage floor, or adding a ``disable_error_code``.
Pure-gate files are denied outright. ``pyproject.toml`` is escalated to a prompt
instead: it holds the mypy knobs *and* the dependency list, so a hard deny would
block ordinary work.

The file-write check runs on BOTH tool families. Guarding only Edit/Write leaves
``sed -i``, ``cp``, ``tee`` and shell redirection as a trivial way to rewrite the
very configs this exists to protect — the lock has to cover the shell too, or it
is decoration of exactly the kind this repo already had too much of.

Two deliberate limits, stated rather than implied:

* Only write *targets* are checked, not every token in a command that happens to
  contain a redirect. The stricter version denied read-only commands that
  mentioned a locked path next to a ``2>&1``. A gate with false positives is a
  gate that gets switched off.
* Shell coverage is pattern-based, so an indirect write (a here-doc into python,
  a helper script) still gets through. This stops an agent reaching for the
  quickest way to turn a build green; it is not an adversarial sandbox. Git
  history is the backstop, which is why the gate configs are version controlled.
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
        # The pre-push gate itself: which checks run, the coverage floor, and
        # whether the Sonar verdict blocks. Lives in scripts/, but it is gate
        # policy, so it carries the same lock as the declarative configs.
        "gate.sh",
        "sonar-publish.sh",
        "install-hooks.sh",
    }
)

# Mixed config: gate knobs live beside dependencies, so prompt rather than deny.
PROMPTED_FILES = frozenset({"pyproject.toml"})

# The git invocation itself: optional env-var prefixes, then `git`, then the
# verb, then its arguments. Anchored, because an unanchored "a git verb appears
# somewhere in this segment" test also matches prose ABOUT git — a heredoc
# mentioning `git` and `pre-commit` next to an unrelated `grep -n` was denied by
# the previous version. The flag must belong to the command, not the sentence.
_GIT_INVOCATION = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*git\s+(?:-\S+\s+)*(commit|push)\b(?P<args>.*)$"
)
_NO_VERIFY = re.compile(r"(?:^|\s)(--no-verify|-n)(?=\s|$|=)")
_HOOKS_PATH = re.compile(r"\bgit\s+config\b[^|;&]*\bcore\.hooksPath\b", re.IGNORECASE)
# Anchored to the start of a command segment: these are env-var prefixes and
# subcommands, not words. Unanchored, it also fires on the name appearing inside
# a commit message, which is a legitimate thing to write.
_HOOK_DISABLE = re.compile(
    r"^\s*(?:SKIP\s*=|PRE_COMMIT_ALLOW_NO_CONFIG\s*=)|\bpre-commit\s+uninstall\b",
    re.IGNORECASE,
)
_REDIRECT_TARGET = re.compile(r">>?\s*[\"']?([\w./\\-]+)")
_WRITE_CMD = re.compile(r"\b(?:cp|mv|tee|dd|install|truncate)\b([^|;&]*)")
_SED_INPLACE = re.compile(r"\bsed\b[^|;&]*?-i[^|;&]*")


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
    if ".claude" in parts and name == "settings.json":
        # settings.json REGISTERS this guard. Protecting the implementation but
        # not the registration protects nothing: delete the hook entry and every
        # other rule here stops running.
        return ("deny", "Blocked: .claude/settings.json registers the gate guard.")
    if name in PROMPTED_FILES:
        return (
            "ask",
            f"{name} holds gate configuration (mypy error codes, tool settings) "
            "alongside dependencies. Confirm this edit is not loosening a check.",
        )
    return None


def _write_targets(command: str) -> list[str]:
    """Paths this command would write to. A read-only mention is not a target."""
    targets: list[str] = [m.group(1) for m in _REDIRECT_TARGET.finditer(command)]
    for match in _WRITE_CMD.finditer(command):
        targets += [t for t in re.findall(r"[\w./\\-]+", match.group(1)) if t[:1] != "-"]
    for match in _SED_INPLACE.finditer(command):
        targets += [t for t in re.findall(r"[\w./\\-]+", match.group(0)) if t[:1] != "-"]
    return targets


def _check_bash(command: str) -> None:
    # `git commit -n` is --no-verify; the flag only matters on commit/push, so
    # an unrelated `-n` (grep -n, sort -n) never trips this.
    for segment in re.split(r"[|;&]{1,2}", command):
        invocation = _GIT_INVOCATION.match(segment)
        if invocation is not None and _NO_VERIFY.search(invocation.group("args")):
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
    for token in _write_targets(command):
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
