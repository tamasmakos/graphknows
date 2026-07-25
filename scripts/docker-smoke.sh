#!/usr/bin/env bash
# docker-smoke.sh — the feedback loop for "the image is not deployable".
#
# Asserts, against the SHIPPED image (no bind-mount, so image defects can't be
# hidden by /app), the three symptoms under diagnosis:
#   A. every dependency a shipped capability imports is present in the image
#   B. every model/corpus a shipped capability loads is BAKED IN
#      (proved by running with the network cut: --network none + HF_HUB_OFFLINE)
#   C. first call has no cold-start download latency
#
# Usage:
#   scripts/docker-smoke.sh                    # assert against existing image
#   IMAGE=graphknows-mcp:latest scripts/docker-smoke.sh
#   scripts/docker-smoke.sh --build            # clean rebuild first (slow)
#
# Exit 0 = green. Non-zero = red, with FAIL lines naming each defect.

set -uo pipefail

IMAGE="${IMAGE:-graphknows-mcp:latest}"

if [[ "${1:-}" == "--build" ]]; then
  echo "=== rebuilding $IMAGE (runtime stage) ==="
  docker build --target runtime -t "$IMAGE" . || exit 1
fi

echo "=== image: $IMAGE ==="
docker image inspect "$IMAGE" --format 'size={{.Size}} bytes' || exit 1

# The probe runs INSIDE the container with no network at all. Anything that
# needs a download to work will fail loudly here instead of silently at 3am
# on a customer's first call.
read -r -d '' PROBE <<'PY'
import importlib, os, sys, time, traceback

fails, oks = [], []

def check(label, fn):
    t0 = time.perf_counter()
    try:
        fn()
    except BaseException as exc:
        fails.append(f"{label}: {type(exc).__name__}: {str(exc)[:300]}")
    else:
        oks.append(f"{label} ({time.perf_counter()-t0:.2f}s)")

# ---- A. imports the shipped code performs -------------------------------
# Each maps to a real import site in graphknows/ (grep-verified).
for mod, site in [
    ("sentence_transformers", "storage/embedder.py, ranking/rerank.py"),
    ("torch",                 "storage/embedder.py, ranking/rerank.py"),
    ("nltk",                  "channels/_framenet.py, ingestion/.../svo.py"),
    ("spacy",                 "channels/_framenet.py"),
    ("gliner2",               "ingestion extraction"),
    ("scipy",                 "topics"),
    ("igraph",                "community detection at flush"),
    ("leidenalg",             "community detection at flush"),
    ("dspy",                  "llm_assisted relations [assisted]"),
    ("rdflib",                "ontology loader [assisted]"),
]:
    check(f"import {mod}  <- {site}", lambda m=mod: importlib.import_module(m))

# ---- B. models + corpora baked in (network is cut) -----------------------
def nltk_corpus(name, probe):
    import nltk  # noqa
    probe()

check("nltk corpus wordnet  <- svo.py _phatic/_supersense",
      lambda: __import__("nltk.corpus", fromlist=["wordnet"]).wordnet.synsets("run"))
check("nltk corpus framenet_v17  <- _framenet.py _lus()",
      lambda: __import__("nltk.corpus", fromlist=["framenet"]).framenet.frames()[:1])
check("spacy en_core_web_lg  <- _framenet.py _spacy()",
      lambda: __import__("spacy").load("en_core_web_lg", disable=["ner"]))

# Model ids come from the settings/module defaults that own them — never
# hardcoded here, or this probe drifts from the app exactly like the image did.
from graphknows.settings import GraphKnowsSettings
EMBED = GraphKnowsSettings().embed_model
def embed_model():
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(EMBED, trust_remote_code=True)
check(f"embed model {EMBED}  <- settings.embed_model", embed_model)

def cross_encoder():
    from sentence_transformers import CrossEncoder
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", max_length=512)
check("cross-encoder ms-marco-MiniLM-L6-v2  <- rerank.py:31", cross_encoder)

RELEX = os.environ.get("GRAPHKNOWS_RELEX_MODEL", "knowledgator/gliner-relex-large-v1.0")
def relex_model():
    from gliner import GLiNER
    GLiNER.from_pretrained(RELEX)
check(f"relex model {RELEX}  <- gliner_model.py:15", relex_model)

VERIFIER = os.environ.get("GRAPHKNOWS_RELATION_VERIFIER_MODEL",
                          "oneryalcin/gliner2-relation-verifier")
def verifier_weights():
    from huggingface_hub import snapshot_download
    snapshot_download(VERIFIER, local_files_only=True)
check(f"relation verifier {VERIFIER}  <- settings.py:192 (default-ON)", verifier_weights)

def verifier_encoder():
    from transformers import AutoTokenizer, AutoModel
    AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    AutoModel.from_pretrained("microsoft/deberta-v3-base")
check("verifier encoder microsoft/deberta-v3-base  <- _gliner2_verifier.py:33",
      verifier_encoder)

# ---- C. cold-start latency of the first real call ------------------------
# Exercise the real entrypoints, not just the model constructors. Constructing a
# SentenceTransformer directly is not the same code path as embed(): the latter
# goes through settings, the suffix lookup and the device resolution, and it was
# embed() — not the constructor — that turned out to hit the network.
_dims = {}

def real_embed_call():
    from graphknows.storage.embedder import embed
    _dims["embed"] = embed(["Alex joined Acme as CTO in March."]).shape[1]
check("embed() end-to-end  <- the real ingest/query embedding path", real_embed_call)

def relex_loader():
    from graphknows.ingestion.extraction.entities.gliner_model import load_gliner2_model
    load_gliner2_model()
check("load_gliner2_model()  <- the default llm_free extraction path", relex_loader)

def warm_call_latency():
    """A baked model must make the SECOND call cheap and the first non-network."""
    from graphknows.storage.embedder import embed
    t0 = time.perf_counter()
    for _ in range(5):
        embed(["warm call"])
    per_call = (time.perf_counter() - t0) / 5
    _dims["warm_s"] = round(per_call, 3)
    assert per_call < 1.0, f"warm embed took {per_call:.2f}s/call — model not cached?"
check("warm embed latency < 1s/call", warm_call_latency)

print("\n--- OK ---")
for o in oks: print("  PASS", o)
print("\n--- DEFECTS ---")
for f in fails: print("  FAIL", f)
print(f"\n{len(oks)} pass, {len(fails)} fail")
if _dims:
    print(f"measured: {_dims}")
sys.exit(1 if fails else 0)
PY

echo
echo "=== probing image with NETWORK CUT (--network none) ==="
# Deliberately does NOT pass HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE. Injecting them
# here masks a real defect: the image must be self-sufficient by its OWN
# configuration. An earlier version of this script set them and therefore passed
# while `embed()` still tried to reach huggingface.co on every model load.
docker run --rm --network none \
  -e PYTHONPATH=/app \
  --entrypoint python "$IMAGE" -c "$PROBE"
rc=$?

echo
echo "=== exit $rc ($( [[ $rc -eq 0 ]] && echo GREEN || echo RED )) ==="
exit $rc
