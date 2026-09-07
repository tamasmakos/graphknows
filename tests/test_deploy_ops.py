"""Deployment operational basics (FR-042): memory limits, backup/restore, SBOM+provenance.

Read as text with pathlib + re, the way tests/test_ci_workflow.py reads ci.yml and
release.yml — deploy/compose.yaml is exercised structurally elsewhere
(tests/test_offline.py), and docs/deployment.md has no structure to parse at all,
so a text check is the natural fit for all three rather than a special case for
one of them.

  * `mem_limit` (not `deploy.resources.limits.memory`) is the assertion for
    deploy/compose.yaml: that field is honoured by plain `docker compose up`
    without swarm mode, which is how deploy/Dockerfile's stack — and
    release.yml's `stack` job — actually run it.
  * docs/deployment.md must show both directions of the backup/restore
    procedure against the volume actually declared in compose, `arcadedb_data`
    — a doc that only shows how to back up is half a procedure.
  * release.yml's SBOM step is a new `uses:` reference, so
    test_ci_workflow.py's existing `test_third_party_actions_are_pinned_to_a_commit_hash`
    (parametrized over both workflow files) already covers pinning it. The
    provenance step is deliberately a plain `run:` (self-attested sha256
    digests, not a Sigstore-signed attestation) — see
    test_release_records_provenance_covering_the_wheel_and_bundle for why.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "deploy" / "compose.yaml"
DEPLOYMENT_DOC = REPO_ROOT / "docs" / "deployment.md"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"

COMPOSE_TEXT = COMPOSE_FILE.read_text(encoding="utf-8")
DEPLOYMENT_TEXT = DEPLOYMENT_DOC.read_text(encoding="utf-8")
RELEASE_TEXT = RELEASE_WORKFLOW.read_text(encoding="utf-8")


def _service_text(name: str) -> str:
    """Slice one top-level service's block out of COMPOSE_TEXT, from
    `^  <name>:$` up to the next service key or the `volumes:` top-level
    key — so an assertion scoped to one service cannot be satisfied by a
    setting that lives on the other."""
    match = re.search(rf"^  {re.escape(name)}:$", COMPOSE_TEXT, re.MULTILINE)
    assert match is not None, f"{COMPOSE_FILE}: no `  {name}:` service found"
    start = match.end()
    next_key = re.search(r"^  \w[\w-]*:$", COMPOSE_TEXT[start:], re.MULTILINE)
    end = start + next_key.start() if next_key else len(COMPOSE_TEXT)
    return COMPOSE_TEXT[start:end]


@pytest.mark.parametrize("service", ["arcadedb", "app"])
def test_service_declares_a_memory_limit(service: str) -> None:
    """Both services need a bound: an unbounded arcadedb or app container can
    take down the host it shares with the other, and `mem_limit` (not
    `deploy.resources.limits.memory`, which is swarm-oriented) is what plain
    `docker compose up` actually enforces."""
    service_text = _service_text(service)
    match = re.search(
        r'^\s*mem_limit:\s*"?\$\{[A-Za-z_]+:-\S+?\}"?\s*$', service_text, re.MULTILINE
    )
    assert match is not None, (
        f"{COMPOSE_FILE}: `{service}` service has no `mem_limit:` line with a "
        "configurable default (${VAR:-default})"
    )


def test_arcadedb_memory_limit_leaves_room_above_the_jvm_heap() -> None:
    """The container limit must exceed the JVM's own -Xmx: a limit equal to or
    below the heap ceiling gets the process OOM-killed by the kernel instead
    of by the JVM, discarding whatever the healthcheck would have reported."""
    arcadedb_text = _service_text("arcadedb")
    xmx_match = re.search(r"-Xmx\$\{GRAPHKNOWS_ARCADEDB_XMX:-(\d+)G\}", arcadedb_text)
    limit_match = re.search(
        r"mem_limit:\s*\"?\$\{GRAPHKNOWS_ARCADEDB_MEM_LIMIT:-(\d+)G\}", arcadedb_text
    )
    assert xmx_match is not None, f"{COMPOSE_FILE}: no GRAPHKNOWS_ARCADEDB_XMX default found"
    assert limit_match is not None, (
        f"{COMPOSE_FILE}: no GRAPHKNOWS_ARCADEDB_MEM_LIMIT default found"
    )
    assert int(limit_match.group(1)) > int(xmx_match.group(1)), (
        f"{COMPOSE_FILE}: arcadedb mem_limit default ({limit_match.group(1)}G) does not "
        f"exceed the -Xmx default ({xmx_match.group(1)}G) — the container would be killed "
        "at or before the JVM's own heap ceiling"
    )


def test_deployment_doc_covers_both_backup_and_restore_of_the_arcadedb_volume() -> None:
    """A procedure that only shows how to make an archive, with no matching
    command to load it back in, is not a restore procedure — this asserts
    both halves exist and both name the actual volume from compose.yaml."""
    assert "arcadedb_data" in DEPLOYMENT_TEXT, (
        f"{DEPLOYMENT_DOC}: never names the `arcadedb_data` volume declared in {COMPOSE_FILE}"
    )
    heading_match = re.search(r"^#+\s*Backup and restore\s*$", DEPLOYMENT_TEXT, re.MULTILINE)
    assert heading_match is not None, f"{DEPLOYMENT_DOC}: no 'Backup and restore' heading"
    section = DEPLOYMENT_TEXT[heading_match.end() :]
    next_heading = re.search(r"^#+\s", section, re.MULTILINE)
    if next_heading:
        section = section[: next_heading.start()]

    assert re.search(r"tar c[a-z]*f", section), (
        f"{DEPLOYMENT_DOC}: 'Backup and restore' section has no `tar c..f` archive command"
    )
    assert re.search(r"tar x[a-z]*f", section), (
        f"{DEPLOYMENT_DOC}: 'Backup and restore' section has no `tar x..f` extract command"
    )
    assert "docker compose" in section and "arcadedb" in section, (
        f"{DEPLOYMENT_DOC}: 'Backup and restore' section never operates the arcadedb service "
        "through docker compose"
    )


def _step_text(name: str) -> str:
    """Slice one `- name: <name>` step's YAML block out of RELEASE_TEXT, up
    to the next step or end of file."""
    match = re.search(rf"^\s*- name: {re.escape(name)}\s*$", RELEASE_TEXT, re.MULTILINE)
    assert match is not None, f"{RELEASE_WORKFLOW}: no step named {name!r} found"
    start = match.end()
    next_step = re.search(r"^\s*- (name: |uses: )", RELEASE_TEXT[start:], re.MULTILINE)
    end = start + next_step.start() if next_step else len(RELEASE_TEXT)
    return RELEASE_TEXT[start:end]


def test_release_generates_an_sbom_for_the_release_assets() -> None:
    """The wheel and bundle attached to a release must ship a machine-readable
    bill of materials scanned from what is actually released (`dist/`), not
    the working tree — a recipient with no repository access has no other
    way to audit what went into the artifact."""
    step_text = _step_text("generate a software bill of materials")
    assert "sbom-action" in step_text, (
        f"{RELEASE_WORKFLOW}: SBOM step does not use an sbom-generating action"
    )
    assert re.search(r"format:\s*spdx-json", step_text), (
        f"{RELEASE_WORKFLOW}: SBOM step does not request spdx-json output"
    )
    output_match = re.search(r"output-file:\s*(\S+\.spdx\.json)", step_text)
    assert output_match is not None, f"{RELEASE_WORKFLOW}: SBOM step has no `output-file:`"
    sbom_path = output_match.group(1)

    upload_step = _step_text("attach the wheel, bundle, SBOM and provenance record to the release")
    assert re.search(r"gh release upload.*" + re.escape(sbom_path), upload_step, re.DOTALL), (
        f"{RELEASE_WORKFLOW}: SBOM file {sbom_path!r} is generated but never uploaded "
        "with `gh release upload`"
    )


def test_release_records_provenance_covering_the_wheel_and_bundle() -> None:
    """The wheel and the deployment bundle — the two files a recipient
    actually downloads — must both be digested into the provenance record,
    not just one of them; a provenance record covering only the wheel leaves
    the bundle's Dockerfile/compose/scripts payload unattested.

    This is a self-attested record (sha256 digests + commit/run/ref), not a
    Sigstore-signed attestation: test_publishes_wheel_as_private_release_asset
    (tests/test_release_workflow.py) already pins that `id-token` — the OIDC
    permission any signed attestation needs — never reappears in this
    workflow now that FR-007 removed the PyPI Trusted Publisher that used to
    need one. A signed attestation step would fail that test outright, so
    this file must not require one either.
    """
    assert "id-token" not in RELEASE_TEXT, (
        f"{RELEASE_WORKFLOW}: contains `id-token` — a provenance step here must be "
        "self-attested, not OIDC-signed (see test_publishes_wheel_as_private_release_asset)"
    )

    step_text = _step_text("record provenance for the release assets")
    assert "sha256" in step_text, (
        f"{RELEASE_WORKFLOW}: provenance step never computes a sha256 digest"
    )
    assert re.search(r"dist/\*\.whl", step_text), (
        f"{RELEASE_WORKFLOW}: provenance step does not cover the wheel (dist/*.whl)"
    )
    assert re.search(r'"\$BUNDLE"', step_text), (
        f"{RELEASE_WORKFLOW}: provenance step does not cover the deployment bundle ($BUNDLE)"
    )
    output_match = re.search(r'open\("(dist/\S+\.json)"', step_text)
    assert output_match is not None, (
        f"{RELEASE_WORKFLOW}: provenance step never writes a JSON provenance file"
    )
    provenance_path = output_match.group(1)
    upload_step = _step_text("attach the wheel, bundle, SBOM and provenance record to the release")
    assert re.search(r"gh release upload.*" + re.escape(provenance_path), upload_step, re.DOTALL), (
        f"{RELEASE_WORKFLOW}: provenance file {provenance_path!r} is generated but never "
        "uploaded with `gh release upload`"
    )
