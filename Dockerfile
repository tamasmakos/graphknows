# ─── builder stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# uv for fast, reproducible installs
RUN pip install --no-cache-dir uv

# Copy workspace manifests first — maximise layer caching.
# README.md is required: pyproject declares `readme = "README.md"`, so the
# hatchling build backend fails ("Readme file does not exist") without it.
COPY pyproject.toml uv.lock README.md ./
COPY graphknows ./graphknows
COPY frontend ./frontend

# Install the batteries-included package (all extras) + dev/eval groups.
# Core deps are lean; the ML/NLP/transport stack lives behind extras (see
# pyproject [project.optional-dependencies]); [all] pulls them for the server.
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache \
        .[all] \
        --group dev \
        --group eval \
        jupyterlab

# ─── runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Install uv in the runtime so it's available in the workspace
RUN pip install --no-cache-dir uv

# Ensure `git` is available inside runtime containers (workspace, mcp, etc.)
# Keep apt lists cleaned to reduce image size.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Bring in only the venv and the application source
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/graphknows ./graphknows
COPY --from=builder /build/frontend ./frontend

# spaCy model for GLiNER2 schema mining — installed INTO the venv the app uses
# (a plain `spacy download` targets the system python, which the app never runs).
RUN uv pip install --python /opt/venv/bin/python \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_CONTROL=manual

# Non-root user for container security
RUN useradd -m -u 1000 graphknows
USER graphknows

CMD ["graphknows-mcp"]

# ---------------------------------------------------------------------------
# dev — the workspace image. Everything below is developer tooling and MUST NOT
# leak into the shipped `runtime` stage that backs the mcp and web services.
# Build with `--target dev` (docker-compose does this for the workspace service).
# ---------------------------------------------------------------------------
FROM runtime AS dev

USER root

# curl is only needed to fetch the SonarQube CLI installer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# SonarQube CLI — code-quality/secret scanning for local development and the
# git pre-commit hook. Installed to a system path so it is on PATH for every
# shell (login or not). Authenticate with SONARQUBE_CLI_TOKEN + SONARQUBE_CLI_ORG;
# `sonar auth login` is browser-based and cannot run headless.
RUN curl -fsSL https://raw.githubusercontent.com/SonarSource/sonarqube-cli/refs/heads/master/user-scripts/install.sh \
      | HOME=/opt bash \
    && ln -sf /opt/.local/share/sonarqube-cli/bin/sonar /usr/local/bin/sonar \
    && sonar --version

USER graphknows
