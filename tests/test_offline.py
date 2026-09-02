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
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DOCKERFILE = REPO_ROOT / "deploy" / "Dockerfile"
DEPLOY_COMPOSE = REPO_ROOT / "deploy" / "compose.yaml"
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
