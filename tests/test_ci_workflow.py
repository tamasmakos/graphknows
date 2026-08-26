"""CI workflow contract: what makes ci.yml callable and its output publishable.

These invariants exist because ci.yml is currently a leaf workflow — its `on:`
block only fires on pull_request/push/workflow_dispatch — so a release/tag
workflow has no way to reuse it and would have to duplicate every check:

  * `workflow_call` with a `skip_sonar` input is the seam a release workflow
    needs to invoke `gate` and `wheel` without duplicating them.
  * The Sonar step must be able to honour that input (via its own `if:`),
    otherwise a caller workflow with no PR context still tries to run Sonar.
  * `cancel-in-progress` currently cancels every non-main ref, tag refs
    included — a tag build superseded by another push to the same tag is
    exactly the release artifact that must survive, not be cancelled.
  * `wheel` builds `--wheel` only and never uploads `dist/`: CI proves a
    wheel *can* be built but produces nothing publishable, and the sdist the
    project ships (`[tool.hatch.build.targets.sdist]`) is never built at all.

Read as text with pathlib + re, not PyYAML: this project does not declare a
yaml dependency, and YAML 1.1 parses the bare `on:` key as the boolean
`True`, which makes structural assertions on it worse than a regex over text.

`.pip-audit-ignore` is exercised here too (test 5) because both scripts/gate.sh
and the `audit` job in this same file extract its entries with the identical
`grep -oE '^(PYSEC|GHSA)[A-Za-z0-9-]+'` — an entry shaped so that regex cannot
see it is a vulnerability silently un-ignored.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PIP_AUDIT_IGNORE = REPO_ROOT / ".pip-audit-ignore"

CI_TEXT = CI_WORKFLOW.read_text(encoding="utf-8")


def test_ci_is_callable_with_a_skip_sonar_input() -> None:
    """A release/tag workflow must be able to `uses: ./.github/workflows/ci.yml`
    instead of duplicating every check, which requires `workflow_call` under
    `on:` with a boolean `skip_sonar` input defaulting to false."""
    assert re.search(r"^  workflow_call:", CI_TEXT, re.MULTILINE), (
        f"{CI_WORKFLOW}: `on:` block has no `workflow_call:` trigger — "
        "ci.yml cannot be reused by a release/tag workflow"
    )
    assert re.search(r"^\s*skip_sonar:\s*$", CI_TEXT, re.MULTILINE), (
        f"{CI_WORKFLOW}: workflow_call block declares no `skip_sonar` input"
    )
    assert "type: boolean" in CI_TEXT, (
        f"{CI_WORKFLOW}: `skip_sonar` input is not declared `type: boolean`"
    )
    assert "default: false" in CI_TEXT, f"{CI_WORKFLOW}: `skip_sonar` input has no `default: false`"


def test_a_caller_can_turn_sonar_off() -> None:
    """The Sonar step's `if:` must reference `inputs.skip_sonar` alongside the
    existing pull_request gate, otherwise a caller workflow with no PR event
    (a tag push) cannot suppress Sonar and the step either never runs or runs
    unconditionally."""
    if_lines = [line for line in CI_TEXT.splitlines() if "if:" in line]
    assert any(
        "github.event_name == 'pull_request'" in line and "inputs.skip_sonar" in line
        for line in if_lines
    ), (
        f"{CI_WORKFLOW}: no `if:` line combines "
        "`github.event_name == 'pull_request'` with `inputs.skip_sonar` — "
        "a caller workflow has no way to disable the Sonar step"
    )


def test_tag_runs_are_never_cancelled() -> None:
    """`cancel-in-progress` must exempt tag refs the same way it exempts main:
    a tag run superseded by another push to the same tag is the release build,
    and losing it to cancellation loses the publishable artifacts."""
    match = re.search(r"^\s*cancel-in-progress:.*$", CI_TEXT, re.MULTILINE)
    assert match is not None, f"{CI_WORKFLOW}: no `cancel-in-progress:` line found"
    line = match.group(0)
    assert "startsWith(github.ref, 'refs/tags/')" in line, (
        f"{CI_WORKFLOW}: cancel-in-progress does not exempt tag refs "
        f"(startsWith(github.ref, 'refs/tags/')) — got: {line.strip()!r}"
    )
    assert "refs/heads/main" in line, (
        f"{CI_WORKFLOW}: cancel-in-progress lost its existing main exemption — "
        f"got: {line.strip()!r}"
    )


def test_wheel_job_produces_a_publishable_distribution() -> None:
    """The `wheel` job must build and upload both the wheel and the sdist the
    project ships (`[tool.hatch.build.targets.sdist]`), not just build a
    wheel and throw `dist/` away — otherwise CI proves a wheel *can* be built
    but produces nothing a release workflow can actually publish."""
    assert not re.search(r"uv build --wheel\b", CI_TEXT), (
        f"{CI_WORKFLOW}: still runs `uv build --wheel` — the wheel-only build "
        "this job must be replaced by a plain `uv build` (wheel + sdist)"
    )
    assert re.search(r"^\s*run: uv build\s*$", CI_TEXT, re.MULTILINE), (
        f"{CI_WORKFLOW}: no plain `uv build` invocation found (must build "
        "both the wheel and the sdist)"
    )
    assert "dist/*.tar.gz" in CI_TEXT or ".tar.gz" in CI_TEXT, (
        f"{CI_WORKFLOW}: the assertion step never checks the sdist "
        "(no `dist/*.tar.gz` / `.tar.gz` reference)"
    )
    assert "actions/upload-artifact" in CI_TEXT, (
        f"{CI_WORKFLOW}: `wheel` job never uploads `dist/` — "
        "no `actions/upload-artifact` step found"
    )
    assert re.search(r"path:.*dist", CI_TEXT), (
        f"{CI_WORKFLOW}: no `path:` entry pointing at `dist` for the upload-artifact step"
    )


def test_vulnerability_ledger_exists_and_is_machine_readable() -> None:
    """`.pip-audit-ignore` must exist and every entry's first token must match
    the exact shape `scripts/gate.sh` and the `audit` job extract with
    `grep -oE '^(PYSEC|GHSA)[A-Za-z0-9-]+'` — an entry that grep cannot see is
    a vulnerability silently un-ignored, and an empty ledger drowns the real
    signal in a huge audit diff."""
    assert PIP_AUDIT_IGNORE.is_file(), (
        f"{PIP_AUDIT_IGNORE}: missing — scripts/gate.sh and the `audit` job both fail without it"
    )
    entry_pattern = re.compile(r"^(PYSEC|GHSA)[A-Za-z0-9-]+$")
    entries = 0
    for lineno, raw_line in enumerate(
        PIP_AUDIT_IGNORE.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        first_token = stripped.split()[0]
        assert entry_pattern.match(first_token), (
            f"{PIP_AUDIT_IGNORE}:{lineno}: {first_token!r} does not match "
            "^(PYSEC|GHSA)[A-Za-z0-9-]+$ — the grep both scripts/gate.sh and "
            "ci.yml use to build --ignore-vuln flags cannot see this entry"
        )
        entries += 1
    assert entries > 0, (
        f"{PIP_AUDIT_IGNORE}: no advisory entries found — auditing with an "
        "empty ignore list drowns the real signal"
    )
