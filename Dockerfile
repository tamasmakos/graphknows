# GraphKnows image.
#
#   docker build -t graphknows .                         # CPU (default, ~4GB)
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
#                -t graphknows:cuda .                    # GPU (~10GB)
#   docker build --target dev -t graphknows:dev .        # + developer tooling
#
# Stages: builder -> models -> runtime -> dev.
# `runtime` is what ships. Nothing a customer does not need may enter it — no
# uv, no git, no test/lint toolchain. Developer tooling lives in `dev`.
#
# Every model is baked in `models` and copied into `runtime`, so a container
# serves requests with no per-call download and works with the network cut.
# Verify with:  scripts/docker-smoke.sh

# ─── builder ───────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir uv

# Manifests first, so a source-only change does not invalidate the (very slow)
# dependency layer. README.md is required: pyproject declares it as the package
# readme, so hatchling fails to build the wheel without it.
COPY pyproject.toml uv.lock README.md ./

# torch dominates image size: the PyPI wheel bundles CUDA kernels (~2.5GB) that
# a CPU deployment never executes. Install it FIRST from the selected index so
# the subsequent resolve sees the constraint already satisfied and does not pull
# the PyPI build over the top. Override TORCH_INDEX to ship a GPU image.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache \
        --index-url ${TORCH_INDEX} torch

COPY graphknows ./graphknows

# `.[assisted]` — NOT `.[all]`, which is not a declared extra. uv only *warns*
# on an unknown extra, so `.[all]` silently installed nothing and shipped an
# image where GRAPHKNOWS_MODE=llm_assisted raised ModuleNotFoundError.
# Runtime deps only: no --group dev/eval, no jupyterlab. Those are `dev`.
RUN uv pip install --python /opt/venv/bin/python --no-cache ".[assisted]"

# Record which lock this venv was resolved from. scripts/preflight.py compares it
# at container start against the uv.lock in the (bind-mounted) source and refuses
# to boot on a mismatch — the guard for an image silently going stale under the
# working tree mounted over it.
RUN sha256sum uv.lock | cut -d' ' -f1 > /opt/venv/.lock-hash

# spaCy model, installed INTO the venv the app runs (a plain `spacy download`
# targets the system python, which the app never uses).
RUN uv pip install --python /opt/venv/bin/python --no-cache \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"

# ─── models ────────────────────────────────────────────────────────────────────
# Separate stage so the download (multi-GB, slow) caches independently of the
# dependency layers and is not re-run on every source edit.
FROM builder AS models

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/models/huggingface \
    NLTK_DATA=/opt/models/nltk_data \
    PYTHONPATH=/build

COPY scripts/bake_models.py ./scripts/bake_models.py

# Resolves every model id from the code that owns it, so this cannot drift from
# the app. Fails the build if any download fails — better here than on a
# customer's first ingest.
RUN python scripts/bake_models.py

# ─── runtime ───────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

RUN useradd -m -u 1000 graphknows

COPY --from=builder --chown=graphknows:graphknows /opt/venv /opt/venv
COPY --from=models  --chown=graphknows:graphknows /opt/models /opt/models
COPY --chown=graphknows:graphknows graphknows ./graphknows
COPY --chown=graphknows:graphknows frontend ./frontend
COPY --chown=graphknows:graphknows scripts/bake_models.py ./scripts/bake_models.py
COPY --chown=graphknows:graphknows scripts/preflight.py ./scripts/preflight.py
COPY --chown=graphknows:graphknows scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Point the ML libraries at the baked caches.
    HF_HOME=/opt/models/huggingface \
    NLTK_DATA=/opt/models/nltk_data \
    # Offline by default. Every default model is baked, and without this the hub
    # client still makes a revision check on each model load: that is per-call
    # latency on a cold worker, a hard runtime dependency on huggingface.co being
    # reachable, and an outright failure in an air-gapped deployment. It does not
    # degrade gracefully either — with the network cut but the hub client live,
    # SentenceTransformer raises rather than falling back to its cache.
    #
    # To run a model that is NOT baked (a different GRAPHKNOWS_EMBED_MODEL, say),
    # set HF_HUB_OFFLINE=0 and accept the first-use download.
    HF_HUB_OFFLINE=1 \
    # Silence the tokenizers fork warning under uvicorn workers.
    TOKENIZERS_PARALLELISM=false

USER graphknows

# Fail fast and loudly if the image is missing a model, instead of surfacing it
# as a download stall on the first customer request.
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD python /app/scripts/bake_models.py --check || exit 1

# Invoked via `sh <script>` rather than as an executable: the +x bit does not
# survive a checkout on every host, and a non-executable entrypoint fails with an
# opaque "exec format" error. Wraps every CMD below, including the dev shell.
ENTRYPOINT ["/bin/sh", "/app/scripts/docker-entrypoint.sh"]

CMD ["graphknows-mcp"]

# ─── web ───────────────────────────────────────────────────────────────────────
# Memory Mirror UI. A stage rather than a second Dockerfile: the old
# Dockerfile.web duplicated the install and had already drifted (it omitted the
# groups that were smuggling nltk in, so the web image lacked deps the mcp image
# had — from the same commit). One build, one dependency set, two entrypoints.
FROM runtime AS web

ENV GRAPHKNOWS_WEB_HOST="0.0.0.0" \
    GRAPHKNOWS_WEB_PORT="8800" \
    GRAPHKNOWS_FRONTEND_DIR="/app/frontend"

EXPOSE 8800

# Timeout raised to 15s: the probe cold-starts a Python interpreter each run and
# was timing out under CPU load (the endpoint itself returns 200), flapping
# unhealthy.
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8800/healthz')" || exit 1

CMD ["graphknows-web"]

# ─── dev ───────────────────────────────────────────────────────────────────────
# The workspace image: developer tooling that MUST NOT leak into `runtime`.
# Build with `--target dev` (docker-compose does this for the workspace service).
FROM runtime AS dev

USER root

RUN pip install --no-cache-dir uv \
    && apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Test/lint/eval toolchain — deliberately absent from `runtime`.
COPY pyproject.toml uv.lock README.md ./
RUN uv pip install --python /opt/venv/bin/python --no-cache \
        --group dev --group eval jupyterlab

# SonarQube CLI — code-quality/secret scanning for local development and the
# git pre-commit hook. Installed to a system path so it is on PATH for every
# shell (login or not). Authenticate with SONARQUBE_CLI_TOKEN + SONARQUBE_CLI_ORG;
# `sonar auth login` is browser-based and cannot run headless.
RUN curl -fsSL https://raw.githubusercontent.com/SonarSource/sonarqube-cli/refs/heads/master/user-scripts/install.sh \
      | HOME=/opt bash \
    && ln -sf /opt/.local/share/sonarqube-cli/bin/sonar /usr/local/bin/sonar \
    && sonar --version

USER graphknows
