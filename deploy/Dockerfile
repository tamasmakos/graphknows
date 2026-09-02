# GraphKnows release image: built once with network, run with none.
#
#   docker build -f deploy/Dockerfile -t graphknows:2.0.0 .
#
# The build context is the release bundle — `deploy/`, `scripts/` and the wheel
# at its root — whose layout mirrors the repository, so the relative paths below
# resolve the same from a bundle and from a checkout that has built its wheel
# (research.md R1). Only the bake step reaches the network; everything after it
# runs offline. Pinned by digest and by exact wheel version so two builds of one
# release agree (research.md R8).

FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

# Cache locations first: the bake writes into them and the runtime reads from
# them. nltk.download() ignores NLTK_DATA, which is why bake_models.py passes
# download_dir explicitly.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/models/huggingface \
    NLTK_DATA=/opt/models/nltk_data

COPY graphknows-2.0.0-py3-none-any.whl /tmp/

# Exact filename, never a glob: `[ontology]` is a glob character class, so
# `graphknows-*.whl[ontology]` matches nothing and pip is handed the literal
# string — a silent build failure that reads as a typo (research.md R9). The
# version here is pinned to graphknows/_version.py by tests/test_offline.py.
# en_core_web_lg is not a wheel dependency and spacy.load() never downloads it,
# so the bake's spacy step needs it installed here.
RUN pip install --no-cache-dir \
    "/tmp/graphknows-2.0.0-py3-none-any.whl[ontology]" \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl" \
    && rm /tmp/graphknows-2.0.0-py3-none-any.whl

WORKDIR /app

# The wheel ships only the `graphknows` package, so these three are not in it
# and come from the build context instead (research.md R1).
COPY scripts/bake_models.py scripts/preflight.py scripts/docker-entrypoint.sh ./scripts/

RUN useradd -m -u 1000 graphknows \
    && mkdir -p /opt/models \
    && chown graphknows:graphknows /opt/models

USER graphknows

# The one step permitted to use the network, which is why the air-gap is a
# runtime guarantee and not a build-time one. Run as the runtime user so the
# weights land owned by it, with no multi-gigabyte chown layer afterwards.
RUN python scripts/bake_models.py

# Only after the bake: HF_HUB_OFFLINE=1 set any earlier would make the bake
# itself fail. GRAPHKNOWS_REQUIRE_BAKED is release-image-only — the dev
# container downloads lazily and its cold start breaks if this leaks there.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    GRAPHKNOWS_REQUIRE_BAKED=1

# `sh <script>`: the +x bit does not survive a checkout on every host, and a
# non-executable entrypoint fails with an opaque "exec format" error.
ENTRYPOINT ["/bin/sh", "/app/scripts/docker-entrypoint.sh"]
CMD ["graphknows-mcp"]
