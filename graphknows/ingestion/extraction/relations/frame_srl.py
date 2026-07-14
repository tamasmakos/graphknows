"""Frame-guided semantic role labeling: FrameNet roles as a GLiNER2 schema.

The combination validated in prototyping: a chunk is embedding-matched to its
top FrameNet frames; each candidate frame's CORE frame elements (with their
FrameNet definitions) become a GLiNER2 entity schema; GLiNER2 zero-shot fills
the roles (Employee=Caroline, Position=nurse, Employer=hospital at 0.95-1.0 on
fact-bearing text). A frame instance is kept only when the best-filling candidate
fills ``MIN_CORE_ROLES`` core roles with at least one high-confidence fill — the
role-fillability gate, which targets fact-bearing content and stays quiet on
conversational filler (where frame selection is unreliable and roles don't fill).

FrameNet metadata comes from nltk (cached), so the full ontology is never
materialized wholesale — only the frames a chunk actually evokes are used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable

from graphknows.ingestion.extraction.relations.svo import _resolve  # speaker resolution

_MARKUP = re.compile(r"<[^>]+>")
# Gate: keep a frame instance only if the best candidate fills >= this many core
# roles, with >= one fill at or above the confidence floor.
MIN_CORE_ROLES = 2
ROLE_CONF_FLOOR = 0.55
# GLiNER2 role-extraction threshold (roles are precise spans; keep it moderate).
_ROLE_THRESHOLD = 0.35
# Frame elements are single entities per role for the instance's primary reading;
# GLiNER2 may return several — we keep them all but the gate counts distinct roles.
_MAX_CORE_FES = 7


@lru_cache(maxsize=4096)
def frame_ontology(name: str) -> dict[str, Any]:
    """A frame's sub-ontology from nltk FrameNet (cached).

    Returns ``{name, definition, core_fes: [(fe, def)], peripheral_fes: [(fe, def)],
    lexical_units: [(word, pos)], parents: [frame names]}``. Empty dict on lookup
    failure (unknown frame).
    """
    from graphknows.channels._framenet import _framenet, _ontology_up

    try:
        f = _framenet().frame(name)
    except Exception:  # noqa: BLE001 — unknown/removed frame
        return {}

    def _clean(text: str) -> str:
        return _MARKUP.sub("", text or "").strip()

    core, peripheral = [], []
    for fe, o in f.FE.items():
        entry = (fe, _clean(o.definition)[:200])
        (core if o.coreType == "Core" else peripheral).append(entry)
    lus = []
    for lu in f.lexUnit:
        word, _, pos = lu.rpartition(".")
        lus.append((word or lu, pos))
    return {
        "name": f.name,
        "definition": _clean(f.definition),
        "core_fes": core,
        "peripheral_fes": peripheral,
        "lexical_units": lus,
        "parents": sorted(_ontology_up().get(f.name, set())),
    }


def _role_schema(name: str) -> dict[str, dict[str, Any]]:
    """GLiNER2 entity schema from a frame's core FEs (FE definition = the hint)."""
    onto = frame_ontology(name)
    spec: dict[str, dict[str, Any]] = {}
    for fe, definition in onto.get("core_fes", [])[:_MAX_CORE_FES]:
        spec[fe] = {"description": definition or fe, "threshold": _ROLE_THRESHOLD}
    return spec


_LU_POS = {"VERB": "v", "NOUN": "n", "ADJ": "a"}


_MAIN_VERB_DEPS = {"ROOT", "conj", "xcomp", "ccomp", "advcl", "acl", "relcl"}


def lu_candidate_frames(text: str, nlp: Any, top_k: int = 4) -> list[tuple[str, float]]:
    """Candidate frames = the CLAUSE MAIN VERB's LU frames, ranked by embedding fit.

    Frame identification is hard zero-shot: definition-embedding alone picks
    plausible-but-wrong frames ('nurse' -> Law_enforcement_agency); any-content-word
    LU lookup lets a secondary trigger win ('named' beats 'adopted'). Combining the
    two precisely works best: the clause MAIN verbs (dependency root + coordinated/
    subordinate verbs) are the primary frame-evoking elements, so we take only their
    LU-triggered frames (precision), then rank those candidates by how well the frame
    DEFINITION embeds against the sentence (semantic fit) — the intersection of
    "the verb evokes it" and "it fits the sentence". Falls back to all verbs, then
    all content words, when the parse yields no main verb.
    """
    from graphknows.channels._frame_index import match_frames_text
    from graphknows.channels._framenet import _lus

    doc = nlp(text)
    lui = _lus()
    verbs = [t for t in doc if t.pos_ == "VERB" and t.dep_ in _MAIN_VERB_DEPS]
    if not verbs:
        verbs = [t for t in doc if t.pos_ == "VERB"]
    cand: set[str] = set()
    for v in verbs:
        cand |= set(lui.get((v.lemma_.lower(), "v"), ()))  # type: ignore[arg-type]
    if not cand:  # no verb triggers — fall back to noun/adj LUs
        for t in doc:
            pos = _LU_POS.get(t.pos_)
            if pos:
                cand |= set(lui.get((t.lemma_.lower(), pos), ()))  # type: ignore[arg-type]
    if not cand:
        return []
    # Rank the LU-valid candidates by embedding fit to the sentence.
    ranked = match_frames_text(text, top_k=200, min_sim=0.0)
    order = {name: sim for name, sim in ranked}
    scored = sorted(cand, key=lambda f: -order.get(f, 0.0))[:top_k]
    return [(f, order.get(f, 0.0)) for f in scored]


@dataclass
class FrameInstance:
    """One evoked frame with its GLiNER2-filled roles (fillers are entity names)."""

    frame: str
    roles: dict[str, list[str]] = field(default_factory=dict)
    match_sim: float = 0.0
    role_confidence: float = 0.0

    @property
    def filled_core(self) -> int:
        return sum(1 for v in self.roles.values() if v)


def _fill_roles(
    text: str, frame: str, gliner: Any, speaker: str, speakers: list[str]
) -> tuple[dict[str, list[str]], float]:
    """Run GLiNER2 with the frame's core-FE schema; return {role: [fillers]}, max_conf."""
    spec = _role_schema(frame)
    if not spec:
        return {}, 0.0
    schema = gliner.create_schema()
    schema.entities(spec)
    raw = gliner.extract(text, schema, _ROLE_THRESHOLD, True, True, True)
    roles: dict[str, list[str]] = {}
    max_conf = 0.0
    for role, spans in (raw.get("entities") or {}).items():
        fillers: list[str] = []
        for s in spans or []:
            surface = (s.get("text") or "").strip()
            if not surface:
                continue
            conf = float(s.get("confidence", 0.0))
            max_conf = max(max_conf, conf)
            # Resolve 1st/2nd-person fillers to the speaker (the dominant subject).
            resolved = _resolve_surface(surface, speaker, speakers)
            if resolved and resolved not in fillers:
                fillers.append(resolved)
        if fillers:
            roles[role] = fillers
    return roles, max_conf


_PRON = {"i", "me", "my", "myself", "we", "us", "our", "you", "your"}


def _resolve_surface(surface: str, speaker: str, speakers: list[str]) -> str:
    """Speaker-resolve a role filler when it's a bare 1st/2nd-person pronoun."""
    low = surface.lower().strip(" .,!?")
    if low in _PRON:
        class _T:  # minimal shim for svo._resolve (expects .text)
            text = low
        return _resolve(_T(), speaker, speakers)
    return surface


def extract_frame_instances(
    text: str,
    speaker: str,
    speakers: list[str],
    gliner: Any,
    candidate_frames: Callable[[str], list[tuple[str, float]]],
    top_k: int = 3,
) -> list[FrameInstance]:
    """Frame-guided role extraction for one chunk/sentence.

    Args:
        text: the chunk text.
        speaker, speakers: for speaker resolution of pronoun fillers.
        gliner: a loaded GLiNER2 model (create_schema/extract).
        candidate_frames: fn(text) -> [(frame_name, cosine)], the embedding matcher
            (top frames the text evokes). Trying the top-k lets the best-filling
            frame win when the single best-cosine frame is wrong.
        top_k: how many candidate frames to attempt.

    Returns the kept FrameInstances (role-fillability gate applied): the best
    candidate by filled-core-role count, kept only if it fills >= MIN_CORE_ROLES
    with a fill at/above ROLE_CONF_FLOOR. Usually 0 or 1 per short text.
    """
    cands = candidate_frames(text)[:top_k]
    best: FrameInstance | None = None
    for frame, sim in cands:
        roles, max_conf = _fill_roles(text, frame, gliner, speaker, speakers)
        inst = FrameInstance(frame=frame, roles=roles, match_sim=sim, role_confidence=max_conf)
        if best is None or inst.filled_core > best.filled_core:
            best = inst
    if best is None:
        return []
    if best.filled_core >= MIN_CORE_ROLES and best.role_confidence >= ROLE_CONF_FLOOR:
        return [best]
    return []
