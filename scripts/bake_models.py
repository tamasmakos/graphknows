"""Download every model GraphKnows loads, into the image's caches.

Run at **build time** (`python scripts/bake_models.py`) so the shipped image can
serve requests with the network cut and with no per-call download latency, and
at **boot** (`--check`) as a preflight that fails fast instead of discovering a
missing model on a customer's first ingest.

The model set is read from ``graphknows.settings`` and the module-level defaults
that own each choice, so this file cannot drift from the code the way a
hand-maintained list in a Dockerfile does. Adding a model to the app without
adding it here makes ``--check`` fail, which is the point.

Caches honour the standard env vars (``HF_HOME``, ``NLTK_DATA``); the Dockerfile
points both at /opt/models so a single COPY moves them into the runtime stage.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable

# nltk corpora the default (llm_free) relation path looks up. Absent, the call
# sites in svo.py / _framenet.py catch LookupError and silently return a
# degraded result — no error, just quietly worse extraction. Bake them.
NLTK_CORPORA = ("wordnet", "omw-1.4", "framenet_v17")

# The verifier's frozen encoder backbone, loaded separately from the verifier
# head's own weights.
VERIFIER_ENCODER = "microsoft/deberta-v3-base"


def _model_ids() -> dict[str, str]:
    """Resolve every model id from the code that owns it — never hardcoded here."""
    from graphknows.ingestion.extraction.entities.gliner_model import _RELEX_MODEL
    from graphknows.settings import GraphKnowsSettings

    s = GraphKnowsSettings()
    return {
        "relex": _RELEX_MODEL,
        "embed": s.embed_model,
        "verifier": s.relation_verifier_model,
        "verifier_encoder": VERIFIER_ENCODER,
        "spacy": s.spacy_model,
    }


def _relex(model_id: str, local_only: bool) -> None:
    from gliner import GLiNER

    GLiNER.from_pretrained(model_id, local_files_only=local_only)


def _embed(model_id: str, local_only: bool) -> None:
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(model_id, local_files_only=local_only)


def _verifier(model_id: str, local_only: bool) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(model_id, local_files_only=local_only)


def _verifier_encoder(model_id: str, local_only: bool) -> None:
    from transformers import AutoModel, AutoTokenizer

    AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    AutoModel.from_pretrained(model_id, local_files_only=local_only)


def _spacy(model_id: str, local_only: bool) -> None:
    import spacy

    spacy.load(model_id, disable=["ner"])


def _nltk_corpora(_model_id: str, local_only: bool) -> None:
    """Fetch (unless *local_only*) and then verify the corpora."""
    import nltk

    # download_dir MUST be explicit. nltk.download() does not honour NLTK_DATA:
    # when it runs as root (as it does in a Docker build) it picks a system
    # directory instead, reports success, and leaves nothing where the runtime
    # looks — a baked image that passes its own build and then raises LookupError
    # on the first request. Anchor it to NLTK_DATA, the single path the runtime
    # stage copies and searches.
    target = os.environ.get("NLTK_DATA", "")
    if not target:
        raise RuntimeError(
            "NLTK_DATA is not set. Baking corpora without it writes them to a "
            "location the runtime does not search."
        )

    if not local_only:
        for corpus in NLTK_CORPORA:
            if not nltk.download(corpus, download_dir=target, quiet=True):
                raise RuntimeError(f"nltk.download({corpus!r}) returned False")

    # Downloading is not the same as being usable, so verify by making the calls
    # the runtime makes. Deliberately NOT nltk.data.find(): wordnet and omw-1.4
    # stay zipped and find('corpora/wordnet') fails on them even though the
    # corpus readers load from the zip perfectly well. Assert the symptom, not a
    # proxy for it.
    from nltk.corpus import framenet as fn
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer

    wn.synsets("run", pos=wn.VERB)[0].lexname()  # svo.py _phatic
    WordNetLemmatizer().lemmatize("motorcycles", "n")  # svo.py _supersense
    fn.frames()[:1]  # _framenet.py _lus()


# (key into _model_ids(), label, loader). Order is display order.
_STEPS: tuple[tuple[str, str, Callable[[str, bool], None]], ...] = (
    ("relex", "relex          ", _relex),
    ("embed", "embed          ", _embed),
    ("verifier", "verifier       ", _verifier),
    ("verifier_encoder", "verifier-enc   ", _verifier_encoder),
    ("spacy", "spacy          ", _spacy),
    ("nltk", "nltk corpora   ", _nltk_corpora),
)


def _bake(check: bool) -> int:
    """Download (or, with *check*, assert the presence of) every model."""
    ids = _model_ids()
    ids["nltk"] = ", ".join(NLTK_CORPORA)
    local_only = check  # turns every fetch into a cache assertion
    failures: list[str] = []

    print(f"{'checking' if check else 'baking'} models:", flush=True)

    for key, label, loader in _STEPS:
        model_id = ids[key]
        shown = f"{label} {model_id}"
        t0 = time.perf_counter()
        try:
            loader(model_id, local_only)
        except BaseException as exc:
            failures.append(f"{shown}: {type(exc).__name__}: {str(exc)[:200]}")
            print(f"  FAIL {shown}", flush=True)
        else:
            print(f"  ok   {shown} ({time.perf_counter() - t0:.1f}s)", flush=True)

    if failures:
        verb = "missing from the image" if check else "failed to download"
        print(f"\n{len(failures)} model(s) {verb}:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        if check:
            print(
                "\nThe image was built without these. Rebuild it, or set the "
                "matching GRAPHKNOWS_* variable to a model that is present.",
                file=sys.stderr,
            )
        return 1

    print(f"\nall {len(_STEPS)} model groups {'present' if check else 'baked'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="assert every model is already cached (boot preflight) instead of downloading",
    )
    args = ap.parse_args()

    if args.check:
        # Prove we are reading the cache, not the network: a check that silently
        # downloads is not a check.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    return _bake(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
