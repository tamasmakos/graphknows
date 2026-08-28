"""Deterministic FrameNet semantic-frame extraction (llm_free-compatible).

For each sentence in a chunk: spaCy identifies frame-evoking target tokens
(verbs/nouns/adjectives, with a gerund->verb fallback), a lexical-unit lookup
maps each to candidate FrameNet frames, and a bge embedding of the sentence vs
each frame's definition disambiguates among them (with an accept floor + margin).
Frame elements (roles) are filled from the dependency parse: subject/object from
core FE slots, and locative/temporal/other prepositional phrases collected across
the target's clause. Fully local (spaCy + FrameNet + the configured embedder),
API-free — suitable for the llm_free profile.

Used at flush time to write a FRAME layer to LTM, and at query time to parse a
question into the same representation so retrieval can match structurally.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

import numpy as np

from graphknows.symbolic import match

log = logging.getLogger(__name__)

# Disambiguation gates (tuned on LoCoMo conv-26; see .scratch/FRAME_GRAPH_NOTES.md).
SIM_FLOOR = 0.35  # min cosine(sentence, frame definition) for a multi-candidate LU
SIM_FLOOR_SINGLE = 0.42  # stricter floor when the LU maps to exactly one frame
SIM_MARGIN = 0.04  # min top1-top2 gap when ambiguous
LIGHT_VERBS = frozenset(
    {"have", "get", "make", "take", "do", "go", "be", "let", "say", "come", "want"}
)
JUNK_FILLERS = frozenset(
    {
        "what",
        "it",
        "that",
        "this",
        "there",
        "which",
        "who",
        "thing",
        "stuff",
        "anything",
        "something",
        "one",
        "some",
        "lot",
        "kind",
    }
)
_POS = {"VERB": "v", "NOUN": "n", "ADJ": "a"}

_nlp: Any = None
_lu_index: dict[tuple[str, str], set[str]] | None = None
_onto_up: dict[str, set[str]] | None = None


def _spacy() -> Any:
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_lg", disable=["ner"])
    return _nlp


def _framenet() -> Any:
    from nltk.corpus import framenet as fn

    try:
        fn.frames()  # probe: raises LookupError if the corpus isn't downloaded
    except LookupError:
        import nltk

        nltk.download("framenet_v17", quiet=True)
    return fn


def _lus() -> dict[tuple[str, str], set[str]]:
    global _lu_index
    if _lu_index is None:
        fn = _framenet()
        idx: dict[tuple[str, str], set[str]] = defaultdict(set)
        for lu in fn.lus():
            m = re.match(r"^(.*)\.([a-z]+)$", lu.name)
            if m:
                idx[(m.group(1).lower(), m.group(2))].add(lu.frame.name)
        _lu_index = idx
    return _lu_index


def _broader_up() -> dict[str, set[str]]:
    """Frame -> strictly more general frames, persisted as SKOS ``BROADER``.

    ``Inheritance`` only. See :mod:`graphknows.symbolic.framenet.relations` for
    why ``Using`` and ``Perspective_on`` are excluded — briefly: presupposing a
    frame is not being a kind of it, and a perspective is a sibling view, not a
    parent. Including them made 683 of the hierarchy's 1464 edges (47%)
    non-subsumption, and the narrower-expansion walk handed those back as if
    they were specialisations.

    Distinct from :func:`_ontology_up`, which is a query-side EXPANSION map.
    Widening a search to ``Using``/``Perspective_on`` neighbours is legitimate;
    storing them as a taxonomy is not.
    """
    from graphknows.symbolic.framenet.relations import broader_pairs

    return broader_pairs()


def _ontology_up() -> dict[str, set[str]]:
    """Child -> parent EXPANSION map (Inheritance/Using upward; Perspective_on both).

    Deliberately wider than :func:`_broader_up`: widening a query to frames a
    target presupposes or perspectivises finds relevant chunks, and nothing here
    is stored as a claim. Do not reuse this for the persisted hierarchy.
    """
    global _onto_up
    if _onto_up is None:
        fn = _framenet()
        up: dict[str, set[str]] = defaultdict(set)
        for r in fn.frame_relations():
            a, b, t = r.get("superFrameName"), r.get("subFrameName"), r.type.name
            if not a or not b:
                continue
            if t in ("Inheritance", "Using"):
                up[b].add(a)
            elif t == "Perspective_on":
                up[a].add(b)
                up[b].add(a)
        _onto_up = up
    return _onto_up


@lru_cache(maxsize=8000)
def _frame_def(name: str) -> str:
    return re.sub(r"<[^>]+>", "", _framenet().frame(name).definition or "")[:400]


@lru_cache(maxsize=8000)
def _core_fes(name: str) -> tuple[str, ...]:
    return tuple(fe for fe, o in _framenet().frame(name).FE.items() if o.coreType == "Core")


def expand_frames(frames: set[str], hops: int = 1) -> set[str]:
    """Expand a frame set upward through the ontology by *hops* generations."""
    up = _ontology_up()
    out, frontier = set(frames), set(frames)
    for _ in range(hops):
        nxt: set[str] = set()
        for f in frontier:
            nxt |= up.get(f, set())
        out |= nxt
        frontier = nxt
    return out


def _degerund(word: str) -> list[str]:
    """Candidate verb stems for a gerund noun ('camping' -> camp/campe)."""
    if not word.endswith("ing") or len(word) < 5:
        return []
    s = word[:-3]
    cands = [s, s + "e"]
    if len(s) > 2 and s[-1] == s[-2]:
        cands.append(s[:-1])
    return cands


def _resolve(tok: Any, speaker: str, speakers: list[str]) -> str:
    low = tok.text.lower()
    if low in ("i", "me", "my", "myself", "we", "us", "our"):
        return speaker
    if low in ("you", "your") and len(speakers) == 2 and speaker:
        return next(s for s in speakers if s != speaker)
    return tok.text


def _filler(tok: Any, speaker: str, speakers: list[str]) -> str | None:
    r = _resolve(tok, speaker, speakers)
    if r.lower() in JUNK_FILLERS or len(r) < 2:
        return None
    comp = [c.text for c in tok.children if c.dep_ == "compound"]
    return " ".join([*comp, r]) if comp else r


def _clause_pps(tok: Any) -> list[tuple[str, Any]]:
    """(prep, pobj) pairs from the target's subtree + head-chain ancestors.

    Walks up to the clause verb so a gerund's locative attaches even when the PP
    hangs off the head noun/matrix verb ('camping trip in the forest', 'family
    camping at the beach').
    """
    nodes, cur = [tok], tok
    for _ in range(3):
        if cur.head is cur:
            break
        cur = cur.head
        nodes.append(cur)
        if cur.pos_ == "VERB":
            break
    pairs: list[tuple[str, Any]] = []
    for n in nodes:
        for c in n.children:
            if c.dep_ == "prep":
                for p in c.children:
                    if p.dep_ == "pobj":
                        pairs.append((c.text.lower(), p))
    return pairs


def _embed_norm(texts: list[str]) -> np.ndarray:
    from graphknows.storage.embedder import embed

    arr = np.asarray(embed(texts), dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr


def _sentence_targets(
    sent: Any, lui: dict[tuple[str, str], set[str]]
) -> list[tuple[Any, str, list[str]]]:
    """Frame-evoking ``(token, pos, sorted_candidate_frames)`` targets in a sentence."""
    targets: list[tuple[Any, str, list[str]]] = []
    for tok in sent:
        pos = _POS.get(tok.pos_)
        if not pos or (pos == "v" and tok.lemma_.lower() in LIGHT_VERBS):
            continue
        # Compound guard drops lexicalised compound HEADS ("meteor shower" ->
        # Precipitation) but must not drop gerund activities with a modifier
        # ("family camping"), which are exactly the answer-bearing instances.
        if (
            pos == "n"
            and not _degerund(tok.lemma_.lower())
            and any(c.dep_ == "compound" for c in tok.children)
        ):
            continue
        frames = lui.get((tok.lemma_.lower(), pos), set())
        if not frames and pos == "n":
            for stem in _degerund(tok.lemma_.lower()):
                frames = lui.get((stem, "v"), set())
                if frames:
                    break
        if frames:
            targets.append((tok, pos, sorted(frames)))
    return targets


def _disambiguate(targets: list[tuple[Any, str, list[str]]], stext: str) -> dict[str, float]:
    """Sentence-vs-frame-definition cosine for every candidate frame."""
    all_defs = sorted({f for _, _, fr in targets for f in fr})
    embs = _embed_norm([stext] + [_frame_def(f) for f in all_defs])
    return {f: float(embs[1 + i] @ embs[0]) for i, f in enumerate(all_defs)}


def _accept_frame(ranked: list[str], sim_of: dict[str, float]) -> tuple[str | None, float]:
    """Top frame passing the floor + margin gates, or ``(None, 0.0)``."""
    return match.accept(
        ranked, sim_of, floor=SIM_FLOOR, floor_single=SIM_FLOOR_SINGLE, margin=SIM_MARGIN
    )


def _assign_fes(
    tok: Any, cores: tuple[str, ...], speaker: str, speakers: list[str]
) -> list[tuple[str, str]]:
    """Fill frame-element roles from the dependency parse (subject/object + clause PPs)."""
    fes: list[tuple[str, str]] = []
    if tok.pos_ in ("VERB", "NOUN"):
        for c in tok.children:
            ft = _filler(c, speaker, speakers)
            if c.dep_ in ("nsubj", "nsubjpass") and cores and ft:
                fes.append((cores[0], ft))
            elif c.dep_ in ("dobj", "attr", "acomp") and len(cores) > 1 and ft:
                fes.append((cores[1], ft))
        for prep, pobj in _clause_pps(tok):
            pf = _filler(pobj, speaker, speakers)
            if pf:
                fes.append((prep, pf))
    elif tok.pos_ == "ADJ" and tok.head.pos_ in ("NOUN", "PROPN") and cores:
        ft = _filler(tok.head, speaker, speakers)
        if ft:
            fes.append((cores[0], ft))
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for r, f in fes:
        if (r, f) not in seen:
            seen.add((r, f))
            unique.append((r, f))
    return unique


def extract_frames(text: str, speaker: str, speakers: list[str]) -> list[dict[str, Any]]:
    """Extract frame instances from one utterance/blob.

    Returns dicts: ``{frame, lu, sim, speaker, fes: [(role, filler), ...]}``.
    """
    out: list[dict[str, Any]] = []
    for sent in _spacy()(text).sents:
        targets = _sentence_targets(sent, _lus())
        if not targets:
            continue
        sim_of = _disambiguate(targets, sent.text.strip())
        for tok, pos, frames in targets:
            ranked = sorted(frames, key=lambda f: -sim_of[f])
            frame, top1 = _accept_frame(ranked, sim_of)
            if frame is None:
                continue
            fes = _assign_fes(tok, _core_fes(frame), speaker, speakers)
            out.append(
                {
                    "frame": frame,
                    "lu": f"{tok.lemma_.lower()}.{pos}",
                    "sim": round(top1, 3),
                    "speaker": speaker,
                    "fes": fes,
                }
            )
    return out


def parse_query(query: str) -> dict[str, Any]:
    """Parse a question into query frames (+ontology expansion) and anchor entities."""
    nlp = _spacy()
    doc = nlp(query)
    q_frames = {i["frame"] for i in extract_frames(query, "", [])}
    expanded = expand_frames(q_frames, hops=1)
    anchors: set[str] = set()
    for t in doc:
        if t.pos_ == "PROPN":
            anchors.add(t.text.lower())
        if t.dep_ in ("nsubj", "poss", "dobj") and t.pos_ == "NOUN":
            anchors.add(t.lemma_.lower())
    return {"q_frames": q_frames, "expanded": expanded, "anchors": anchors}
