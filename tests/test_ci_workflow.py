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

  * Every third-party `uses:` reference in ci.yml and release.yml must be
    pinned to a commit hash, not a mutable tag — the same rationale already
    recorded next to `pypa/gh-action-pypi-publish` in release.yml, extended
    to the rest of both workflows as one convention instead of one exception.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
PIP_AUDIT_IGNORE = REPO_ROOT / ".pip-audit-ignore"
PYPROJECT = REPO_ROOT / "pyproject.toml"

CI_TEXT = CI_WORKFLOW.read_text(encoding="utf-8")


def _job_text(name: str) -> str:
    """Slice a top-level job's text out of CI_TEXT, from `^  <name>:$` up to
    the next top-level job key (`^  \\w[\\w-]*:$`) or end of file, so
    assertions scoped to one job cannot be satisfied by a setting that lives
    in another job."""
    match = re.search(rf"^  {re.escape(name)}:$", CI_TEXT, re.MULTILINE)
    assert match is not None, f"{CI_WORKFLOW}: no `  {name}:` job found"
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
    gate_text = _job_text("gate")
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
    gate_text = _job_text("gate")
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
    gate_text = _job_text("gate")
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


@pytest.mark.parametrize("workflow_path", [CI_WORKFLOW, RELEASE_WORKFLOW])
def test_third_party_actions_are_pinned_to_a_commit_hash(workflow_path: Path) -> None:
    """A mutable tag (`@v7`, `@v10.0.1`, ...) can be repointed by the action's
    owner at any time, so what CI actually executes could change with no
    commit in this repository. `pypa/gh-action-pypi-publish` in release.yml
    already carries this rationale in its own trailing comment
    (`# v1.14.2`); this test is what spreads that convention to every other
    `uses:` reference in both workflows instead of leaving it as one
    exception."""
    workflow_text = workflow_path.read_text(encoding="utf-8")
    hash_pattern = re.compile(r"^[a-f0-9]{40}$")
    checked = 0
    for match in re.finditer(r"^\s*(?:- )?uses:\s*(\S+)(.*)$", workflow_text, re.MULTILINE):
        reference, rest_of_line = match.group(1), match.group(2)
        if reference.startswith("./"):
            continue  # local workflow reuse, e.g. `uses: ./.github/workflows/ci.yml` — no hash to pin
        checked += 1
        owner_repo, _, pin = reference.partition("@")
        assert hash_pattern.match(pin), (
            f"{workflow_path}: `uses: {reference}` is not pinned to a full "
            "40-character commit hash — a mutable tag can be repointed by "
            f"{owner_repo}'s owner at any time, changing what CI executes "
            "with no commit in this repository"
        )
        assert re.search(r"#\s*\S+", rest_of_line), (
            f"{workflow_path}: `uses: {reference}` has no trailing `# <version>` "
            "comment — the pinned hash is unreadable by a human without one"
        )
    assert checked > 0, (
        f"{workflow_path}: no third-party `uses:` references were checked — "
        "the matching regex is silently matching nothing"
    )


def test_integration_job_runs_against_arcadedb_service() -> None:
    """The eleven `integration`-marked tests currently never run in any
    workflow — `gate`'s pytest step excludes them (`-m "not integration"`)
    and nothing else invokes them, so a live-storage regression can merge
    unseen. A dedicated `integration` job with a live ArcadeDB service
    container, gated by `GRAPHKNOWS_REQUIRE_ARCADEDB=1` so a container that
    never started fails instead of reporting a suite that skipped past every
    test as green, is what FR-008/SC-005 require."""
    assert re.search(r"^  integration:$", CI_TEXT, re.MULTILINE), (
        f"{CI_WORKFLOW}: no top-level `integration:` job found"
    )
    integration_text = _job_text("integration")

    assert re.search(r"^\s*services:\s*$", integration_text, re.MULTILINE), (
        f"{CI_WORKFLOW}: `integration` job has no `services:` block — "
        "no ArcadeDB service container is provisioned"
    )
    assert re.search(r"^\s*arcadedb:\s*$", integration_text, re.MULTILINE), (
        f"{CI_WORKFLOW}: `integration` job's `services:` block has no `arcadedb:` entry"
    )
    assert "arcadedata/arcadedb" in integration_text, (
        f"{CI_WORKFLOW}: `integration` job's arcadedb service does not use "
        "the arcadedata/arcadedb image"
    )
    assert "--health-cmd" in integration_text, (
        f"{CI_WORKFLOW}: `integration` job's arcadedb service has no health "
        "check — GitHub Actions would start the job's own steps without "
        "waiting for ArcadeDB to be ready"
    )

    run_match = re.search(r"^\s*run:.*pytest.*$", integration_text, re.MULTILINE)
    assert run_match is not None, f"{CI_WORKFLOW}: `integration` job has no `pytest` invocation"
    run_line = run_match.group(0)
    assert re.search(r"-m\s+integration\b", run_line), (
        f"{CI_WORKFLOW}: `integration` job's pytest step does not select "
        f"`-m integration` — got: {run_line.strip()!r}"
    )
    assert "not integration" not in run_line, (
        f"{CI_WORKFLOW}: `integration` job's pytest step still excludes "
        f"integration tests — got: {run_line.strip()!r}"
    )

    assert re.search(r'GRAPHKNOWS_REQUIRE_ARCADEDB:\s*"?1"?', integration_text), (
        f"{CI_WORKFLOW}: `integration` job does not set "
        "GRAPHKNOWS_REQUIRE_ARCADEDB=1 — a container that never started "
        "would skip instead of fail"
    )

    # The job must run on every pull request and every push to main. It
    # carries no job-level `if:` of its own, so it inherits the workflow's
    # `on:` block (`pull_request: branches: [main]` / `push: branches:
    # [main]`) unmodified rather than narrowing it to a subset of events.
    assert not re.search(r"^\s*if:", integration_text, re.MULTILINE), (
        f"{CI_WORKFLOW}: `integration` job carries an `if:` guard that could "
        "narrow it below every pull request and every push to main"
    )
    assert re.search(r"^  pull_request:$", CI_TEXT, re.MULTILINE), (
        f"{CI_WORKFLOW}: no workflow-level `pull_request:` trigger — the "
        "`integration` job would not run on pull requests"
    )
    assert re.search(r"^\s*branches:\s*\[main\]\s*$", CI_TEXT, re.MULTILINE), (
        f"{CI_WORKFLOW}: no `branches: [main]` push trigger — the "
        "`integration` job would not run on pushes to main"
    )


def test_smoke_job_declares_the_interpreters_the_classifiers_name() -> None:
    """`requires-python = ">=3.11"` and the classifier list advertise which
    interpreters this package supports; the `smoke` job is what actually
    installs and imports the built wheel on each one. If the two lists
    diverge, either CI is silently skipping an advertised interpreter or the
    classifiers are advertising one nothing ever tests."""
    smoke_text = _job_text("smoke")

    python_version_line = re.search(r"^\s*python-version:.*$", smoke_text, re.MULTILINE)
    assert python_version_line is not None, (
        f"{CI_WORKFLOW}: `smoke` job has no `python-version:` matrix line"
    )
    smoke_versions = set(re.findall(r'"(\d+\.\d+)"', python_version_line.group(0)))

    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    classifiers = pyproject["project"]["classifiers"]
    classifier_versions = {
        version
        for classifier in classifiers
        for version in re.findall(r"Programming Language :: Python :: (\d+\.\d+)", classifier)
    }

    assert smoke_versions and classifier_versions, (
        f"{CI_WORKFLOW} smoke matrix versions {smoke_versions!r} and "
        f"{PYPROJECT} classifier versions {classifier_versions!r}: neither may be empty"
    )
    assert smoke_versions == classifier_versions, (
        f"{CI_WORKFLOW} smoke matrix declares {sorted(smoke_versions)} but "
        f"{PYPROJECT} classifiers advertise {sorted(classifier_versions)} — "
        "the interpreters CI tests and the interpreters the package claims to "
        "support must be the same set"
    )


def _uv_sync_invocations(dockerfile: str) -> list[str]:
    """Each real `uv sync` command in *dockerfile*, continuations joined.

    Comment lines are skipped: this Dockerfile explains its own sync steps in
    prose that says "uv sync" too, and counting those would make the check
    assert against the comments rather than the commands.
    """
    lines = dockerfile.splitlines()
    invocations: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.lstrip().startswith("#") and "uv sync" in line:
            command = [line]
            while command[-1].rstrip().endswith("\\") and index + 1 < len(lines):
                index += 1
                command.append(lines[index])
            invocations.append("\n".join(command))
        index += 1
    return invocations


def test_build_extras_use_ontology_not_assisted() -> None:
    """The image and CI must sync an extra that exists.

    `assisted` was deleted when dspy and litellm became core (ADR 0003), and
    `uv sync --extra assisted --locked` fails outright on an undeclared extra —
    so a leftover flag here is a build that cannot start, found at image-build
    time rather than at review time. The LLM decoder needs no extra now; the RDF
    loader still does, which is why the replacement is `ontology` and not
    nothing.
    """
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    declared = set(
        tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"].get(
            "optional-dependencies", {}
        )
    )
    assert "assisted" not in declared, (
        "this test guards the removal of the `assisted` extra, but pyproject "
        "declares it again — remove the test or the extra, not both"
    )

    for name, text in (("Dockerfile", dockerfile), (CI_WORKFLOW.name, CI_TEXT)):
        for extra in re.findall(r"--extra\s+([A-Za-z0-9._-]+)", text):
            assert extra in declared, (
                f"{name}: syncs `--extra {extra}`, which pyproject.toml does not "
                f"declare (declared: {sorted(declared)}) — `uv sync --locked` fails on it"
            )

    syncs = _uv_sync_invocations(dockerfile)
    assert len(syncs) == 2, f"Dockerfile: expected two `uv sync` invocations, found {len(syncs)}"
    for sync in syncs:
        assert "--extra ontology" in sync, (
            f"Dockerfile: a `uv sync` does not carry `--extra ontology`, so the dev "
            f"image loses the RDF loader:\n{sync}"
        )

    image_extras = re.search(r'ENV GRAPHKNOWS_IMAGE_EXTRAS="([^"]*)"', dockerfile)
    assert image_extras is not None, "Dockerfile: no GRAPHKNOWS_IMAGE_EXTRAS declaration"
    claimed = {e for e in image_extras.group(1).split(",") if e}
    assert claimed <= declared, (
        f"Dockerfile: GRAPHKNOWS_IMAGE_EXTRAS claims {sorted(claimed - declared)}, which is "
        "not a declared extra — scripts/preflight.py would refuse to start the container"
    )
    assert "ontology" in claimed, (
        f"Dockerfile: GRAPHKNOWS_IMAGE_EXTRAS is {sorted(claimed)} but the image syncs "
        "`--extra ontology`; preflight can only verify what the image claims"
    )
