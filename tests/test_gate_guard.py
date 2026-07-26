"""The gate guard is enforcement code, so it carries tests like any other.

``.claude/hooks/gate_guard.py`` is what stops an agent skipping the git hooks or
loosening a gate config. It is also the one file in the repo that cannot be
exercised by using it — a bug there fails open, silently, which is the exact
failure class the rest of this work exists to eliminate.

Two properties matter and pull against each other:

* it must deny the real bypasses (``--no-verify``, hook-path re-pointing, writes
  to gate configs from either the edit tools or the shell), and
* it must NOT deny ordinary work. An earlier version flagged any command that
  contained a redirect and mentioned a locked path, so ``grep floor
  scripts/gate.sh 2>&1`` was blocked. A gate with false positives is a gate
  someone switches off, at which point it protects nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

GUARD = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "gate_guard.py"

# Assembled rather than written literally: the guard inspects the command string
# of every Bash call, and a test file containing the raw flag makes routine
# commands that touch this file (grep, sed) trip the live hook.
NO_VERIFY = "--no" + "-verify"
SKIP_VAR = "SKI" + "P="


def decide(payload: dict[str, object]) -> str:
    """Run the guard exactly as Claude Code does and return its verdict."""
    result = subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    if not result.stdout.strip():
        return "allow"
    return json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"]


def bash(command: str) -> str:
    return decide({"tool_name": "Bash", "tool_input": {"command": command}})


def powershell(command: str) -> str:
    return decide({"tool_name": "PowerShell", "tool_input": {"command": command}})


def edit(file_path: str) -> str:
    return decide({"tool_name": "Edit", "tool_input": {"file_path": file_path}})


@pytest.mark.skipif(not GUARD.exists(), reason="gate guard not installed")
class TestBypassesAreDenied:
    @pytest.mark.parametrize(
        "command",
        [
            f"git commit {NO_VERIFY} -m x",
            f"git push {NO_VERIFY}",
            "git commit -n -m x",
            f"{SKIP_VAR}mypy git commit -m x",
            "git config core.hooksPath /dev/null",
            "pre-commit uninstall",
        ],
    )
    def test_gate_skipping_command(self, command: str) -> None:
        assert bash(command) == "deny"

    @pytest.mark.parametrize(
        "command",
        [
            "sed -i s/x/y/ ruff.toml",
            "cp /tmp/g.py .claude/hooks/gate_guard.py",
            "echo foo > pytest.ini",
            "tee .importlinter",
            "cp /tmp/x scripts/gate.sh",
            "mv /tmp/y scripts/sonar-publish.sh",
            "echo x > sonar-project.properties",
        ],
    )
    def test_shell_write_to_gate_config(self, command: str) -> None:
        """Guarding only Edit/Write would leave every one of these open."""
        assert bash(command) == "deny"

    @pytest.mark.parametrize(
        "path",
        [
            "/app/ruff.toml",
            "C:\\repo\\.pre-commit-config.yaml",
            "/app/.claude/hooks/gate_guard.py",
            "/app/scripts/gate.sh",
            "/app/pytest.ini",
            # settings.json REGISTERS the guard. Locking the implementation but
            # leaving the registration writable protects nothing — delete the
            # hook entry and every other rule here stops running.
            "/app/.claude/settings.json",
        ],
    )
    def test_edit_of_gate_config(self, path: str) -> None:
        assert edit(path) == "deny"

    @pytest.mark.parametrize(
        "command",
        [f"git commit {NO_VERIFY} -m x", "cp /tmp/x scripts/gate.sh"],
    )
    def test_powershell_is_guarded_too(self, command: str) -> None:
        """PowerShell is this environment's primary shell.

        The guard handled the PowerShell tool from the start, but the hook
        matcher named only `Bash`, so those calls never reached it — the whole
        rule set was one tool name away from being bypassed.
        """
        assert powershell(command) == "deny"


@pytest.mark.skipif(not GUARD.exists(), reason="gate guard not installed")
class TestOrdinaryWorkIsAllowed:
    @pytest.mark.parametrize(
        "command",
        [
            "git commit -m 'normal message'",
            # The words appear in a commit MESSAGE, not as an env prefix. The
            # unanchored version of this rule blocked a real commit whose text
            # described the rule itself.
            f"git commit -m 'denies pre-commit {SKIP_VAR} vars'",
            "cat ruff.toml",
            "bash scripts/gate.sh 2>&1 | tail -5",
            "grep -n floor scripts/gate.sh 2>&1",
            "docker exec graphknows-workspace python -m pytest -q",
            "sed -i s/a/b/ graphknows/memory.py",
            "echo hi > /tmp/scratch.txt",
            # Prose about git next to an unrelated `-n`. The unanchored version
            # of the rule read this as `git commit -n` and denied it, which is
            # how a real Makefile edit got blocked.
            "python - <<'PY'\n# Run the gate as the git hook does, pre-commit style\nPY\ngrep -n gate Makefile",
            "grep -n 'pre-commit' scripts/gate.sh",
            "git log --oneline -n 5",
        ],
    )
    def test_command_is_not_blocked(self, command: str) -> None:
        assert bash(command) == "allow"

    @pytest.mark.parametrize(
        "path",
        ["/app/graphknows/memory.py", "/app/tests/test_x.py", "/app/scripts/preflight.py"],
    )
    def test_edit_is_not_blocked(self, path: str) -> None:
        assert edit(path) == "allow"


@pytest.mark.skipif(not GUARD.exists(), reason="gate guard not installed")
class TestMixedConfigPrompts:
    """pyproject.toml holds mypy knobs *and* dependencies, so it asks."""

    def test_edit_prompts(self) -> None:
        assert edit("/app/pyproject.toml") == "ask"

    def test_shell_write_prompts(self) -> None:
        assert bash("cp /tmp/p.toml pyproject.toml") == "ask"


@pytest.mark.skipif(not GUARD.exists(), reason="gate guard not installed")
def test_malformed_payload_fails_open() -> None:
    """A broken payload must not wedge every tool call in the session."""
    result = subprocess.run(
        [sys.executable, str(GUARD)], input="not json", capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
