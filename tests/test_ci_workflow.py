"""CI workflow contract: what makes ci.yml callable and its output publishable.

These invariants exist because ci.yml is currently a leaf workflow — its `on:`
block only fires on pull_request/push/workflow_dispatch — so a release/tag
workflow has no way to reuse it and would have to duplicate every check:

  * `workflow_call` is the seam a release workflow needs to invoke `gate` and
    `wheel` without duplicating them.
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


def _gate_job_text() -> str:
    """Slice the `gate` job's text out of CI_TEXT, from `^  gate:$` up to the
    next top-level job key (`^  \\w[\\w-]*:$`) or end of file, so assertions
    scoped to `gate` cannot be satisfied by a setting that lives in another
    job."""
    match = re.search(r"^  gate:$", CI_TEXT, re.MULTILINE)
    assert match is not None, f"{CI_WORKFLOW}: no `  gate:` job found"
    start = match.end()
    next_job = re.search(r"^  \w[\w-]*:$", CI_TEXT[start:], re.MULTILINE)
    end = start + next_job.start() if next_job else len(CI_TEXT)
    return CI_TEXT[start:end]


def test_gate_job_checks_out_full_history() -> None:
    """Diff coverage compares the diff against the pull request's base
    branch; a shallow clone has no `origin/<base>` to compare against, so the
    step fails loudly at best and silently reports a perfect score at worst.
    This setting was previously attributed to a now-deleted external-analysis
    step and was deleted with it — this test is what stops that happening
    again."""
    gate_text = _gate_job_text()
    assert "fetch-depth: 0" in gate_text, (
        f"{CI_WORKFLOW}: `gate` job checkout has no `fetch-depth: 0` — "
        "diff-cover has no origin/<base> to compare against"
    )


def test_diff_coverage_step_is_guarded_on_a_succeeded_suite_and_a_pull_request() -> None:
    """Unlike its neighbours this step must not run after a failed suite —
    there is nothing to report, reporting anyway is how the previous
    new-code number got pinned at zero, and a collection error leaves no
    coverage.xml so a permissive guard adds a missing-file error on top of
    the real failure."""
    gate_text = _gate_job_text()
    assert "diff-cover" in gate_text, f"{CI_WORKFLOW}: `gate` job has no `diff-cover` invocation"
    idx = gate_text.index("diff-cover")
    block_start = gate_text.rfind("\n      - ", 0, idx) + 1
    next_step = re.search(r"^      - ", gate_text[idx:], re.MULTILINE)
    block_end = idx + next_step.start() if next_step else len(gate_text)
    step_text = gate_text[block_start:block_end]

    if_match = re.search(r"^\s*if:.*$", step_text, re.MULTILINE)
    assert if_match is not None, f"{CI_WORKFLOW}: diff-cover step has no `if:` guard"
    if_line = if_match.group(0)
    assert "success()" in if_line, (
        f"{CI_WORKFLOW}: diff-cover step's `if:` does not contain `success()` — "
        f"got: {if_line.strip()!r}"
    )
    assert "github.event_name == 'pull_request'" in if_line, (
        f"{CI_WORKFLOW}: diff-cover step's `if:` does not contain "
        f"`github.event_name == 'pull_request'` — got: {if_line.strip()!r}"
    )
    assert "!cancelled()" not in if_line, (
        f"{CI_WORKFLOW}: diff-cover step's `if:` uses `!cancelled()`, which "
        f"runs it after a failed suite with no coverage.xml — got: {if_line.strip()!r}"
    )


def test_diff_coverage_threshold_is_a_distinct_constant_from_the_repo_floor() -> None:
    """The two thresholds are never expressed in terms of each other, so
    ratcheting one cannot silently move the other."""
    gate_text = _gate_job_text()
    repo_floor_match = re.search(r"--cov-fail-under=(\d+)", gate_text)
    diff_threshold_match = re.search(r'DIFF_COVERAGE_MIN:\s*"?(\d+)"?', gate_text)
    assert repo_floor_match is not None, (
        f"{CI_WORKFLOW}: no `--cov-fail-under=<N>` literal found in the gate job"
    )
    assert diff_threshold_match is not None, (
        f"{CI_WORKFLOW}: no `DIFF_COVERAGE_MIN: <N>` literal found in the gate job"
    )
    assert repo_floor_match.group(1) != diff_threshold_match.group(1), (
        f"{CI_WORKFLOW}: diff-coverage threshold "
        f"({diff_threshold_match.group(1)!r}) equals the whole-repo floor "
        f"({repo_floor_match.group(1)!r}) — they must be distinct literals"
    )

    idx = gate_text.index("diff-cover")
    block_start = gate_text.rfind("\n      - ", 0, idx) + 1
    next_step = re.search(r"^      - ", gate_text[idx:], re.MULTILINE)
    block_end = idx + next_step.start() if next_step else len(gate_text)
    step_text = gate_text[block_start:block_end]
    assert "cov-fail-under" not in step_text, (
        f"{CI_WORKFLOW}: diff-cover step references `cov-fail-under` — the "
        "diff-coverage threshold must not be expressed in terms of the "
        "whole-repo floor"
    )


def test_ci_is_callable_by_a_release_workflow() -> None:
    """A release/tag workflow must be able to `uses: ./.github/workflows/ci.yml`
    instead of duplicating every check, which requires `workflow_call` under
    `on:`."""
    assert re.search(r"^  workflow_call:", CI_TEXT, re.MULTILINE), (
        f"{CI_WORKFLOW}: `on:` block has no `workflow_call:` trigger — "
        "ci.yml cannot be reused by a release/tag workflow"
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
