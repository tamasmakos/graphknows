# GraphKnows dev workspace image. Docker is dev-only: the PyPI wheel is the
# product, there is no shipped runtime image, so this is one stage.
#
#   docker build -t graphknows:dev .                                  # CPU
#   docker build --build-arg TORCH_BACKEND=cu126 -t graphknows:dev .  # GPU
#
# Measured: 10.1GB (`docker images graphknows:dev`). The old four-stage `dev`
# target additionally baked every model into the image on top of the same
# torch/transformers/spaCy install this stage carries, so this is not larger.

FROM python:3.13-slim

# uv from its own official image, pinned — never installed through the
# system Python's pip, which floats to whatever uv last published.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

# Build outside /app: docker-compose bind-mounts the working tree over /app,
# which would hide anything built there.
WORKDIR /build

# uv otherwise creates .venv inside the project directory (CWD-relative), and
# that path is exactly what the compose bind mount hides at container start,
# leaving the container with no interpreter. Set before both sync passes.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# CPU wheel by default; --build-arg TORCH_BACKEND=cu126 for the GPU eval
# workstation. This replaces the old TORCH_INDEX arg: syncing from the lock
# resolves from uv.lock, and pointing it at an alternate index URL would
# conflict with --locked below, whereas UV_TORCH_BACKEND is uv's own
# selector for this and is exactly what .github/workflows/ci.yml sets for CI.
ARG TORCH_BACKEND=cpu
ENV UV_TORCH_BACKEND=${TORCH_BACKEND}

# Manifests first, so a source-only change does not invalidate the (very
# slow) dependency layer. README.md is required: pyproject declares it as the
# package readme, so hatchling fails to build the wheel without it.
COPY pyproject.toml uv.lock README.md ./

# Syncing from the lock — NOT the old lock-free `.[assisted]` resolve, which
# resolved fresh from pyproject.toml's ranges and ignored uv.lock entirely,
# disagreeing with what CI and every developer resolve from (#196).
# `--locked` fails the build outright on drift instead of silently resolving
# something else. Groups installed here and why: `dev` because
# scripts/gate.sh and the container-exec make targets run
# ruff/mypy/import-linter/bandit/pytest/pip-audit inside this container;
# `eval` because `python -m evaluation` runs here too; `--extra ontology`
# because the RDF/OWL loader must work in the dev container. The LLM decoder
# needs no extra any more — dspy and litellm are core (ADR 0003), so
# GRAPHKNOWS_MODE=llm_assisted works from the base sync alone.
# --no-install-project on this pass: only dependencies, not the package
# itself, so this layer is unaffected by source edits.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --locked --no-install-project \
        --extra ontology --group dev --group eval

COPY graphknows ./graphknows

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --locked \
        --extra ontology --group dev --group eval

# Both installs below MUST come after the last `uv sync`: sync prunes
# anything not in the lock, so installing here first would be wiped.

# torch through the interface TORCH_BACKEND actually governs. UV_TORCH_BACKEND
# steers `uv pip`, not `uv sync`: the syncs above install torch from PyPI
# whatever the arg says, and PyPI's linux wheel bundles CUDA 13 kernels — so a
# cpu build carried ~2.5GB it never runs, and a cu128 build ran on CPU because
# torch.cuda cannot initialise a CUDA-13 wheel under a 12.x driver. Measured
# on the eval box: relex/verifier/frames on 6 CPU threads, ~27 s per chunk.
# `--reinstall` because the version is already satisfied; the unsuffixed
# `nvidia-*` 13.x packages the PyPI wheel pulled in are not pruned by `uv pip`,
# so they are removed by name. Pin matches uv.lock.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv pip \
    install --reinstall --torch-backend ${TORCH_BACKEND} "torch==2.12.0" \
    && uv pip freeze | grep -E '^nvidia-[a-z0-9-]+==13\.' | cut -d= -f1 | xargs -r uv pip uninstall

# spaCy model, installed INTO the venv the app runs (a plain `spacy download`
# targets the system python, which the app never uses). Not part of uv.lock,
# so it goes through uv's pip-compatible interface rather than `uv sync`.
RUN uv pip \
    install --no-cache \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"

# The `srl` extra: transformer frame identification, which the eval measures.
# `--no-deps` is required: it pins protobuf<6 where this project resolves
# 7.x, and drags in pytorch-lightning for training code inference never runs.
# `--no-deps` also drops the rest of both packages' closure not already
# satisfied by the `uv sync` above: gdown, reached through nlpaug's
# unconditional `import gdown` (nlpaug/util/file/download.py), plus gdown's
# own beautifulsoup4/soupsieve/PySocks. Pins match uv.lock.
RUN uv pip \
    install --no-cache --no-deps frame-semantic-transformer nlpaug \
        gdown==6.1.0 beautifulsoup4==4.15.0 soupsieve==2.9.2 PySocks==1.7.1

# Records which optional extras THIS image was built to carry. `ontology` comes
# from the `uv sync --extra ontology` above; `srl` from the out-of-band install
# just above. scripts/preflight.py refuses to start the container if a claimed
# extra's distributions aren't actually installed, so a build that drops the
# step above announces itself instead of silently shipping less. The LLM stack
# is not listed because it is no longer an extra — it is core.
ENV GRAPHKNOWS_IMAGE_EXTRAS="ontology"

RUN useradd -m -u 1000 graphknows

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    # Model weights live in a compose NAMED VOLUME mounted at /opt/models —
    # a different lifetime from the BuildKit `uv` cache mount above. Deleting
    # the wrong one either re-downloads gigabytes of models or does nothing
    # useful, so keep the two straight.
    HF_HOME=/opt/models/huggingface \
    NLTK_DATA=/opt/models/nltk_data

WORKDIR /app

# /opt/models is a compose named volume, mounted empty and root-owned at
# container start — a different lifetime from the BuildKit `uv` cache mount
# above. Create + chown it now so the non-root user below can write model
# downloads into it on first load.
RUN mkdir -p /opt/models && chown graphknows:graphknows /opt/models

# The compose bind mount normally supplies these, but nothing else does when
# the image runs standalone (`docker run graphknows:dev`). pyproject.toml is
# what preflight.py reads its declared dependencies/extras from; harmlessly
# shadowed by the bind mount's own copy under compose.
COPY --chown=graphknows:graphknows scripts/docker-entrypoint.sh scripts/preflight.py ./scripts/
COPY --chown=graphknows:graphknows pyproject.toml ./

USER graphknows

# Invoked via `sh <script>` rather than as an executable: the +x bit does not
# survive a checkout on every host, and a non-executable entrypoint fails
# with an opaque "exec format" error.
ENTRYPOINT ["/bin/sh", "/app/scripts/docker-entrypoint.sh"]
