"""Offline-deployment invariants for `deploy/`.

The release image makes a runtime promise — ingest and search with no outbound
connection — that nothing else in the tree can check, because the image is built
on the client's host and never in CI. What *is* checkable here is the recipe:
whether the build is reproducible (base pinned by digest, wheel by exact
filename), whether everything the entrypoint needs is actually in the image, and
whether the offline switches are thrown at a point in the build where they do not
break the one step that is allowed to use the network.

The Dockerfile is read as an ordered list of instructions rather than as a blob
of text, because two of those questions are about *order*: `HF_HUB_OFFLINE=1`
before the bake would fail the build, and after it is the whole point.

The compose file is read as text split into service blocks rather than through
PyYAML — the project declares no yaml dependency, the same reason
`tests/test_release_workflow.py` reads workflows with `re` — because the only
questions asked of it are which service carries which key, and whether a name
appears at all.

One node here is not about the recipe: `test_llm_free_ingest_and_recall_attempt_no_egress`
runs the `llm_free` write and read paths behind a monkeypatched
`socket.socket.connect` and asserts they dialled nothing but the database. It is
the closest this suite gets to the runtime promise itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import itertools
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DOCKERFILE = REPO_ROOT / "deploy" / "Dockerfile"
DEPLOY_COMPOSE = REPO_ROOT / "deploy" / "compose.yaml"
DEPLOY_ACCEPTANCE = REPO_ROOT / "deploy" / "acceptance.sh"
ENTRYPOINT = REPO_ROOT / "scripts" / "docker-entrypoint.sh"
VERSION_FILE = REPO_ROOT / "graphknows" / "_version.py"

Instruction = tuple[str, str]


def _instructions(dockerfile: Path) -> list[Instruction]:
    """Parse a Dockerfile into ordered (INSTRUCTION, argument) pairs.

    Backslash continuations are joined first, so a multi-line `RUN` or `ENV` is
    one pair and an index comparison between two of them means what it reads as.
    """
    joined = re.sub(r"\\r?\n\s*", " ", dockerfile.read_text(encoding="utf-8"))
    pairs: list[Instruction] = []
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keyword, _, argument = line.partition(" ")
        pairs.append((keyword.upper(), argument.strip()))
    return pairs


def _env(instructions: list[Instruction]) -> dict[str, str]:
    """Every variable the image declares, last assignment winning."""
    declared: dict[str, str] = {}
    for keyword, argument in instructions:
        if keyword == "ENV":
            declared.update(
                (name, value.strip('"'))
                for name, value in re.findall(r'(\w+)=("[^"]*"|\S+)', argument)
            )
    return declared


def _index_of(instructions: list[Instruction], keyword: str, needle: str) -> int:
    """Position of the one *keyword* instruction mentioning *needle*."""
    hits = [i for i, (kw, arg) in enumerate(instructions) if kw == keyword and needle in arg]
    assert len(hits) == 1, (
        f"{DEPLOY_DOCKERFILE}: expected one {keyword} mentioning {needle!r}, found {len(hits)}"
    )
    return hits[0]


def _package_version() -> str:
    match = re.search(r'__version__ = "([^"]+)"', VERSION_FILE.read_text(encoding="utf-8"))
    assert match, f"{VERSION_FILE}: no __version__ literal"
    return match.group(1)


def test_release_image_is_pinned_offline_and_self_contained() -> None:
    assert DEPLOY_DOCKERFILE.is_file(), f"{DEPLOY_DOCKERFILE}: does not exist"
    instructions = _instructions(DEPLOY_DOCKERFILE)

    keyword, base = instructions[0]
    assert keyword == "FROM", f"{DEPLOY_DOCKERFILE}: first instruction is {keyword}, not FROM"
    assert re.fullmatch(r"python:3\.13-slim@sha256:[0-9a-f]{64}", base), (
        f"{DEPLOY_DOCKERFILE}: base image is {base!r} — python:3.13-slim pinned by digest, "
        "not by tag: the client builds this image, so two builds of one release must agree"
    )

    # Exact filename, never a glob: `[ontology]` is a glob character class, so
    # `graphknows-*.whl[ontology]` expands to nothing and pip silently installs
    # neither the wheel nor the extra (research.md R9).
    wheel = f"graphknows-{_package_version()}-py3-none-any.whl[ontology]"
    install = instructions[_index_of(instructions, "RUN", "pip install")][1]
    assert wheel in install, (
        f"{DEPLOY_DOCKERFILE}: the pip install does not name {wheel!r} — the image installs the "
        "release wheel of this exact version, with the ontology extra"
    )
    assert "*" not in install, f"{DEPLOY_DOCKERFILE}: globbed pip install: {install!r}"

    # Not in the wheel — it ships only the `graphknows` package (research.md R1).
    copied = " ".join(arg for keyword, arg in instructions if keyword == "COPY")
    for script in ("bake_models.py", "preflight.py", "docker-entrypoint.sh"):
        assert f"scripts/{script}" in copied, (
            f"{DEPLOY_DOCKERFILE}: scripts/{script} is never COPYed from the build context, "
            "and it is not in the wheel — the entrypoint would fail on a missing path"
        )

    expected = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "GRAPHKNOWS_REQUIRE_BAKED": "1",
    }
    declared = _env(instructions)
    for name, value in expected.items():
        assert declared.get(name) == value, (
            f"{DEPLOY_DOCKERFILE}: {name} is {declared.get(name)!r}, expected {value!r} "
            "(contracts/deployment-environment.md)"
        )
    for name in ("HF_HOME", "NLTK_DATA"):
        assert declared.get(name, "").startswith("/"), (
            f"{DEPLOY_DOCKERFILE}: {name} must name the absolute path the bake wrote to, "
            f"got {declared.get(name)!r}"
        )

    # Order is the load-bearing part: the bake is the one networked step, so the
    # caches must be pointed at before it and the offline switches thrown after.
    bake = _index_of(instructions, "RUN", "bake_models.py")
    assert _index_of(instructions, "ENV", "HF_HOME") < bake, (
        f"{DEPLOY_DOCKERFILE}: HF_HOME is set after the bake, so the weights land somewhere "
        "the runtime does not read"
    )
    assert _index_of(instructions, "ENV", "HF_HUB_OFFLINE") > bake, (
        f"{DEPLOY_DOCKERFILE}: the offline variables are set before the bake, which is the one "
        "step permitted to use the network — the build would fail with an empty cache"
    )


# The six the release image owns (contracts/deployment-environment.md). The compose
# file may name none of them: whichever value it set would win over the image's.
OFFLINE_VARIABLES = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_HOME",
    "NLTK_DATA",
    "GRAPHKNOWS_REQUIRE_BAKED",
)


def _uncommented(path: Path) -> str:
    """*path* without its comment lines — a variable named in prose is not a setting."""
    return "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def _index_of_line(lines: list[str], needle: str) -> int:
    """Index of the first line in *lines* containing *needle*."""
    return next(i for i, line in enumerate(lines) if needle in line)


def _service_blocks(compose: str) -> dict[str, str]:
    """Every service in *compose*, mapped to the block indented under its name."""
    services = re.split(r"\n(?=\S)", compose.partition("\nservices:\n")[2])[0]
    parts = re.split(r"^  ([\w.-]+):$", services, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_compose_overrides_no_offline_variable() -> None:
    assert DEPLOY_COMPOSE.is_file(), f"{DEPLOY_COMPOSE}: does not exist"
    compose = _uncommented(DEPLOY_COMPOSE)

    app = _service_blocks(compose).get("app")
    assert app is not None, (
        f"{DEPLOY_COMPOSE}: no `app` service — the release stack is the memory service "
        f"and its database, found {sorted(_service_blocks(compose))}"
    )
    # The client builds locally, so there is no registry to pull the app from
    # (research.md R8) — an `image:` here would name one that does not exist.
    assert "build:" in app, f"{DEPLOY_COMPOSE}: the app service declares no `build:`"
    assert "image:" not in app, (
        f"{DEPLOY_COMPOSE}: the app service names an `image:` — the client builds the "
        "release image from the wheel and there is nothing to pull it from"
    )

    for name in OFFLINE_VARIABLES:
        assert name not in compose, (
            f"{DEPLOY_COMPOSE}: sets {name}, which deploy/Dockerfile owns — a compose "
            "value wins over the image's and would silently undo the offline guarantee "
            "(contracts/deployment-environment.md)"
        )


def test_entrypoint_checks_baked_models_only_when_required() -> None:
    """The boot check is gated, shallow, and runs before the handoff.

    All three properties are load-bearing and none is visible from a green
    container: ungated it would break the dev image's lazy cold start (FR-006),
    `--check` would deep-load every model for a ~5x ingest tax (research.md R2),
    and after `exec` it would never run at all.
    """
    assert ENTRYPOINT.is_file(), f"{ENTRYPOINT}: does not exist"
    lines = _uncommented(ENTRYPOINT).splitlines()

    checks = [line for line in lines if "bake_models.py" in line]
    assert len(checks) == 1, (
        f"{ENTRYPOINT}: expected one bake_models.py check, found {len(checks)} — on an "
        "air-gapped host an empty cache must fail at boot, not mid-ingest (FR-005)"
    )
    check = checks[0]
    assert "GRAPHKNOWS_REQUIRE_BAKED" in check, (
        f"{ENTRYPOINT}: the check is ungated: {check.strip()!r} — the dev container sets no "
        "such variable and would pay a full bake on every cold start (FR-006)"
    )
    assert "--present" in check and "--check" not in check, (
        f"{ENTRYPOINT}: the check runs {check.strip()!r} — `--present` is files-on-disk; "
        "`--check` deep-loads every model and costs ~5x an ingest (research.md R2)"
    )
    assert lines.index(check) < _index_of_line(lines, 'exec "$@"'), (
        f'{ENTRYPOINT}: the check sits after `exec "$@"`, which replaces this shell — '
        "nothing below that line ever runs"
    )


# The three ways a host actually stops outbound traffic. The acceptance run's own
# output cannot tell them apart from no blocking at all — it looks identical on a
# networked laptop — so naming the method in the script is the only thing that
# lets a reviewer read a green run as evidence rather than as a smoke test.
EGRESS_METHODS = ("internal: true", "--network none", "firewall")


def _header_comment(script: Path) -> str:
    """The comment block at the top of *script*, shebang and body excluded.

    Read as the leading run of `#` lines rather than as the whole file, so a
    method named down in the code — an echo, a variable, a `docker network`
    invocation — does not count as having documented it where a reviewer looks.
    """
    lines = script.read_text(encoding="utf-8").splitlines()
    header = itertools.takewhile(lambda line: line.startswith("#"), lines[1:])
    return "\n".join(header)


def test_acceptance_script_names_how_egress_is_blocked() -> None:
    assert DEPLOY_ACCEPTANCE.is_file(), (
        f"{DEPLOY_ACCEPTANCE}: does not exist — SC-001…SC-003 have no runnable form"
    )
    header = _header_comment(DEPLOY_ACCEPTANCE)
    assert any(method in header for method in EGRESS_METHODS), (
        f"{DEPLOY_ACCEPTANCE}: its header comment names none of {list(EGRESS_METHODS)}, so a "
        "reviewer cannot tell whether a passing run was made on a blocked host or a networked "
        "one — and the two produce the same output"
    )


# --- The runtime promise: an llm_free ingest and recall dial only the database ---

# Ordinary prose on purpose: two entities in one sentence is enough to send the
# write path through the same extraction a client's document takes.
EGRESS_DOCUMENT = "Ada Lovelace wrote the first algorithm for the Analytical Engine in London."

# A namespace of its own (`mem_offline_egress`) rather than the shared default,
# so a developer with a live ArcadeDB does not find test prose in the golden
# layer. Dropped in the `finally` below.
EGRESS_NAMESPACE = "offline_egress"


class _EgressBlockedError(AssertionError):
    """Raised in place of a connect() to an address outside the allowlist."""


@dataclass
class _Dials:
    """Every (host, port) the guard was asked to dial, split by the allowlist.

    Both halves are read. `blocked` failing the test is the point; `allowed`
    proves the paths under test actually reached the database, because a test
    asserting an absence has to show it was watching something.
    """

    allowed: list[tuple[str, int]] = field(default_factory=list)
    blocked: list[tuple[str, int]] = field(default_factory=list)


def _database_addresses(url: str) -> frozenset[str]:
    """Every address the host in *url* answers to, plus the name itself.

    Under compose the database is `http://arcadedb:2480`, which is not loopback
    (research.md R5): an allowlist of `127.0.0.1`/`::1` alone would block the one
    connection the offline stack is supposed to make, and fail this node for a
    reason that has nothing to do with egress.
    """
    split = urlsplit(url)
    host = split.hostname or ""
    if not host:
        return frozenset()
    try:
        infos = socket.getaddrinfo(host, split.port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        # Nothing resolves the name here — no compose network, or no resolver at
        # all. The name is then the whole allowance, which is all an unresolved
        # host could ever be handed to connect() anyway.
        return frozenset({host})
    return frozenset({host, *(str(info[4][0]) for info in infos)})


def _dialled(address: object) -> tuple[str, int] | None:
    """The (host, port) a connect() argument names, or None when nothing leaves the machine.

    AF_INET hands connect() a 2-tuple and AF_INET6 a 4-tuple, both host-first;
    AF_UNIX hands it a path, which has no host to judge.
    """
    if isinstance(address, tuple) and len(address) >= 2 and isinstance(address[0], str):
        return address[0], int(address[1])
    return None


def _is_allowed(host: str, allowed: frozenset[str]) -> bool:
    """True for the database's own addresses and for every flavour of loopback."""
    if host in allowed:
        return True
    try:
        # `%eth0` — a scoped IPv6 address; ip_address rejects the suffix, and the
        # scope has no bearing on whether the address is local.
        return ipaddress.ip_address(host.partition("%")[0]).is_loopback
    except ValueError:
        return False  # a name the resolver never turned into an address: not ours


class _EgressGuard:
    """An allowlist over every outbound connection the process attempts.

    Stdlib monkeypatching rather than a `pytest-socket` dependency (research.md
    R5). Dials are recorded as well as refused: the ingest path catches broadly
    at its I/O boundaries, so a raise alone could be swallowed before the test
    ever sees it.
    """

    def __init__(self, allowed: frozenset[str]) -> None:
        self.allowed = allowed
        self.dials = _Dials()

    def check(self, address: object) -> None:
        """Record where a connect() is headed, and refuse it if it leaves the allowlist."""
        dial = _dialled(address)
        if dial is None:
            return
        if _is_allowed(dial[0], self.allowed):
            self.dials.allowed.append(dial)
            return
        self.dials.blocked.append(dial)
        raise _EgressBlockedError(
            f"egress to {dial[0]}:{dial[1]} blocked; allowed: {sorted(self.allowed)} plus loopback"
        )

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch both ways a connection is opened, for the duration of one test."""
        real_connect = socket.socket.connect

        def guarded_connect(sock: socket.socket, address: Any) -> Any:
            self.check(address)
            return real_connect(sock, address)

        monkeypatch.setattr(socket.socket, "connect", guarded_connect)

        # The release image is Linux, where asyncio's selector loop dials through
        # the socket object patched above. On Windows the default proactor loop
        # dials with ConnectEx on the socket's handle instead and never calls
        # `socket.socket.connect`, so patching only that would leave this node
        # watching an empty road on the platform it is developed on.
        windows_events = getattr(asyncio, "windows_events", None)
        if windows_events is None:
            return
        real_iocp_connect = windows_events.IocpProactor.connect

        def guarded_iocp_connect(proactor: Any, conn: socket.socket, address: Any) -> Any:
            self.check(address)
            return real_iocp_connect(proactor, conn, address)

        monkeypatch.setattr(windows_events.IocpProactor, "connect", guarded_iocp_connect)


async def test_llm_free_ingest_and_recall_attempt_no_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The llm_free write and read paths connect to the database and to nothing else."""
    # Three of the four switches deploy/Dockerfile bakes in
    # (contracts/deployment-environment.md). They are part of what is under test,
    # not a way around it: with them set a cached model loads from disk, and
    # without them huggingface_hub revalidates each cached file against the hub —
    # a request the release image never makes and this node would report as the
    # client's egress.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("GRAPHKNOWS_MODE", "llm_free")

    from graphknows.memory import Memory
    from graphknows.settings import GraphKnowsSettings, MemoryMode

    settings = GraphKnowsSettings()
    assert settings.mode is MemoryMode.llm_free, (
        f"the mode under test resolved to {settings.mode!r}: llm_assisted calls a network "
        "LLM by design, and would fail this node for a reason it is not asking about"
    )

    # The database host is resolved HERE, before the block goes in: connect() is
    # handed addresses the resolver has already produced, so an allowlist built
    # after the guard could never match one. The honest consequence (research.md
    # R5) is that this test is kinder than the field — on a genuinely DNS-less
    # host the lookup fails out here rather than inside the guarded block, so a
    # green run is not proof that the deployed stack never needs a resolver.
    guard = _EgressGuard(_database_addresses(settings.arcadedb_url))
    guard.install(monkeypatch)

    memory = Memory(settings, namespace=EGRESS_NAMESPACE)
    try:
        # Neither call has to succeed — with no database up they both end in a
        # refused connection to an allowed address, which is what "attempt" in
        # the node's name means. Under test is where they dialled, which the
        # guard records either way; an _EgressBlockedError swallowed in here is
        # read back off `guard.dials` below.
        with contextlib.suppress(Exception):
            await memory.ingest_memory(text=EGRESS_DOCUMENT, session_id="egress-guard")
        with contextlib.suppress(Exception):
            await memory.recall_memory("Analytical Engine", session_id="egress-guard")
    finally:
        with contextlib.suppress(Exception):
            await memory.drop_namespace()
        with contextlib.suppress(Exception):
            await memory.close()

    assert not guard.dials.blocked, (
        f"the llm_free path dialled {sorted(set(guard.dials.blocked))}, outside an allowlist "
        f"of loopback plus {settings.arcadedb_url} — on an air-gapped host that is a hang or "
        "a stack trace, not a slower answer"
    )
    database_port = urlsplit(settings.arcadedb_url).port or 2480
    assert any(port == database_port for _host, port in guard.dials.allowed), (
        f"nothing dialled the database on port {database_port}, so this node proved nothing: "
        "ingest and recall gave up before reaching it, and an absence of egress is not "
        "evidence when the path under test never ran"
    )


# --- The doc that has to say what the image cannot: where the guarantee stops ---

DEPLOYMENT_DOC = REPO_ROOT / "docs" / "deployment.md"
DOCS_INDEX = REPO_ROOT / "docs" / "README.md"
SETTINGS = REPO_ROOT / "graphknows" / "settings.py"

# Each requirement is checked against one `##` section rather than the whole
# page: an operator reads the section a heading sent them to, and three facts
# scattered across three sections answer nobody. Needles are matched casefolded.
GUARANTEE_BOUNDARIES = (
    # FR-020 — the build reaching Hugging Face is the build working.
    ("FR-020", ("build", "network", "running")),
    # FR-021 — both escapes named where they can be found and avoided.
    ("FR-021", ("llm_assisted", "GRAPHKNOWS_EMBED_API_BASE")),
    # FR-022 — a schema change is a rebuild from the original sources.
    ("FR-022", ("schema version", "re-ingest", "no migration")),
)


def _sections(doc: str) -> list[str]:
    """Each `##` heading of *doc* with the body under it, casefolded."""
    return [section.casefold() for section in re.split(r"\n(?=## )", doc)]


def test_deployment_doc_states_the_guarantee_boundaries() -> None:
    assert DEPLOYMENT_DOC.is_file(), (
        f"{DEPLOYMENT_DOC}: does not exist — FR-020…FR-022 are stated nowhere a client reads"
    )
    doc = DEPLOYMENT_DOC.read_text(encoding="utf-8")
    assert "deployment.md" in DOCS_INDEX.read_text(encoding="utf-8"), (
        f"{DOCS_INDEX}: does not link deployment.md — a page missing from the index is a "
        "page nobody finds"
    )

    sections = _sections(doc)
    for requirement, needles in GUARANTEE_BOUNDARIES:
        stated = any(
            all(needle.casefold() in section for needle in needles) for section in sections
        )
        assert stated, (
            f"{DEPLOYMENT_DOC}: no single section says all of {list(needles)} ({requirement})"
        )

    # Every knob the page names must exist. A setting spelled wrong here is worse
    # than an unmentioned one: the operator sets it, sees no effect, and reads
    # the silence as the guarantee holding.
    declared = "\n".join(
        path.read_text(encoding="utf-8") for path in (SETTINGS, DEPLOY_DOCKERFILE, ENTRYPOINT)
    )
    for name in sorted(set(re.findall(r"GRAPHKNOWS_[A-Z_]+", doc))):
        assert name in declared, (
            f"{DEPLOYMENT_DOC}: names {name}, which neither graphknows/settings.py nor the "
            "deploy recipe declares — an operator would set it and change nothing"
        )
