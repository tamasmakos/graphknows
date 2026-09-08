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
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from graphknows.ingestion.extraction.relations._lingfeatures import (
    _resolve,  # speaker resolution
)

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
    from graphknows.symbolic.framenet.framenet import framenet, ontology_up

    fn = framenet()
    try:
        f = fn.frame(name)
    except Exception:
        return {}

    def _clean(text: str) -> str:
        return _MARKUP.sub("", text or "").strip()

    core: list[tuple[str, str]] = []
    peripheral: list[tuple[str, str]] = []
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
        "parents": sorted(ontology_up().get(f.name, set())),
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

# Light-verb constructions: the verb is a detector, not the predicate — the
# NOUN object carries the frame ('make a decision' -> Deciding, not Building).
# Mirror image of the LIGHT_VERBS drop-list in graphknows/symbolic/framenet/framenet.py
# (that one strips these verbs from query targets; here they're read as pointers).
LVC_VERBS = frozenset({"have", "take", "make", "give", "do", "get"})

# Suffixes of eventive/deverbal nouns ("decision", "destruction", "acquisition")
# vs concrete ones ("book", "letter", "medicine"). A light verb only pointer-fires
# at its dobj when the noun looks eventive -- otherwise "gave a book"/"took the
# medicine" are ordinary transitive uses and must keep their own verb frame.
_EVENTIVE_NOUN_SUFFIXES = (
    "tion",
    "sion",
    "ment",
    "ance",
    "ence",
    "ing",
    "age",
    "al",
    "ure",
)


def _looks_eventive(lemma: str) -> bool:
    """Cheap deverbal-noun heuristic: does the lemma end like a nominalisation?"""
    return lemma.lower().endswith(_EVENTIVE_NOUN_SUFFIXES)


@dataclass(frozen=True)
class FrameTrigger:
    """One frame-evoking element with its ranked candidate frames."""

    candidates: tuple[tuple[str, float], ...]
    lvc: bool = False
    # The evoking surface itself, verbatim from the chunk; "" for the fallback
    # sweep, which has no single token to point at. Unused by the local fill
    # (GLiNER2 is handed the context, not the trigger); it names the trigger for
    # a decoder that fills the roles remotely.
    trigger: str = ""
    # The sentence the trigger sits in, when known (spaCy ``token.sent.text``);
    # empty means "use the whole chunk" (verb-group/fallback triggers, or a
    # duck-typed token without ``.sent``). Scopes role-filling so a multi-sentence
    # chunk's second trigger doesn't have its roles filled from the first's clause.
    context: str = ""


def _lvc_targets(
    doc: Any, lui: dict[tuple[str, str], list[str]]
) -> tuple[list[tuple[int, set[str], Any]], set[Any], set[Any]]:
    """LVC triggers: an LVC_VERBS verb's eventive dobj NOUN child.

    Returns ``(triggers, lvc_verb_toks, lvc_noun_toks)`` — the verb/noun token
    sets let the caller exclude them from the ordinary verb group and
    nominalisation sweep below.
    """
    triggers: list[tuple[int, set[str], Any]] = []
    lvc_verb_toks: set[Any] = set()
    lvc_noun_toks: set[Any] = set()
    for v_idx, v in enumerate(doc):
        if v.pos_ != "VERB" or v.lemma_.lower() not in LVC_VERBS:
            continue
        for child in v.children:
            if child.dep_ != "dobj" or child.pos_ != "NOUN":
                continue
            if not _looks_eventive(child.lemma_):
                continue
            cand = set(lui.get((child.lemma_.lower(), "n"), ()))
            if not cand:
                continue
            triggers.append((v_idx, cand, child))
            lvc_verb_toks.add(v)
            lvc_noun_toks.add(child)
    return triggers, lvc_verb_toks, lvc_noun_toks


def _nominalisation_targets(
    doc: Any, lui: dict[tuple[str, str], list[str]], skip: set[Any]
) -> list[tuple[int, set[str], Any]]:
    """NOUN/ADJ triggers with LU frames AND argument structure (prep/poss child)."""
    triggers: list[tuple[int, set[str], Any]] = []
    for t_idx, t in enumerate(doc):
        pos = {"NOUN": "n", "ADJ": "a"}.get(t.pos_)
        if pos is None or t in skip:
            continue
        cand = set(lui.get((t.lemma_.lower(), pos), ()))
        if not cand or not any(c.dep_ in ("prep", "poss") for c in t.children):
            continue
        triggers.append((t_idx, cand, t))
    return triggers


def _verb_group(
    doc: Any, lui: dict[tuple[str, str], list[str]], lvc_verb_toks: set[Any]
) -> tuple[list[Any], set[str]]:
    """The clause MAIN verbs (or all verbs, if none match), minus LVC detectors."""
    verbs = [t for t in doc if t.pos_ == "VERB" and t.dep_ in _MAIN_VERB_DEPS]
    if not verbs:
        verbs = [t for t in doc if t.pos_ == "VERB"]
    cand: set[str] = set()
    for v in verbs:
        if v not in lvc_verb_toks:
            cand |= set(lui.get((v.lemma_.lower(), "v"), ()))
    return verbs, cand


def _fallback_sweep(doc: Any, lui: dict[tuple[str, str], list[str]]) -> set[str]:
    """The old all-content-word sweep — last resort when nothing else fired."""
    cand: set[str] = set()
    for t in doc:
        pos = _LU_POS.get(t.pos_)
        if pos:
            cand |= set(lui.get((t.lemma_.lower(), pos), ()))
    return cand


def _sent_text(tok: Any) -> str:
    """The sentence containing ``tok`` (spaCy ``.sent.text``), or "" if unknown.

    "" tells the caller to fall back to the whole chunk -- true for duck-typed
    test tokens with no ``.sent`` and for any doc with no sentence boundaries.
    """
    sent = getattr(tok, "sent", None)
    if sent is None:
        return ""
    try:
        return str(sent.text).strip()
    except Exception:
        return ""


def lu_triggers(
    text: str,
    nlp: Any,
    top_k: int = 4,
    embedding: list[float] | None = None,
    doc: Any | None = None,
) -> list[FrameTrigger]:
    """Every frame-evoking element in the text, each with its own ranked candidates.

    A chunk can evoke more than one frame: a nominalisation ("the destruction of
    the bridge") sitting next to a verb ("...happened") must not be masked by the
    verb, and a light verb ("made a decision") should point at its noun object
    instead of contributing its own irrelevant LU frames. So triggers are
    enumerated, not chosen one-at-a-time:

    1. LVC triggers — an LVC_VERBS verb's dobj NOUN child, when that child also
       *looks* eventive/deverbal (``_looks_eventive``, a nominalisation-suffix
       heuristic) — one trigger each (``lvc=True``); only then is the verb
       excluded from the verb group below. A concrete dobj ("gave a book",
       "took the medicine") is an ordinary transitive use, not a light-verb
       construction, so the verb keeps its own frame.
    2. Nominalisation triggers — a remaining NOUN or ADJ with LU frames AND
       argument structure (a ``prep``/``poss`` child). Argument structure is
       required so this doesn't turn every noun/adjective in a chunk into an
       extra GLiNER2 pass.
    3. The verb group — the clause MAIN verbs (dependency root + coordinated/
       subordinate verbs), falling back to all verbs when none match, minus any
       verb consumed as an LVC detector — union of their LU frames as ONE trigger
       (unchanged arity from the old verb-only path).
    4. Last-resort fallback — the old all-content-word sweep, run only when 1-3
       together found nothing (not merely when the verb set was empty, or it
       would re-import the junk frames the LVC rule just removed).

    Frame identification is hard zero-shot: definition-embedding alone picks
    plausible-but-wrong frames ('nurse' -> Law_enforcement_agency); any-content-word
    LU lookup lets a secondary trigger win ('named' beats 'adopted'). Combining the
    two precisely works best: take only LU-triggered frames per trigger (precision),
    then rank by how well the frame DEFINITION embeds against the sentence (semantic
    fit) — computed ONCE and shared across every trigger (no extra embedding call
    per trigger).

    ``embedding``/``doc`` are reused exactly as before: pass the chunk's precomputed
    vector and shared parse so this never re-embeds or re-parses what a caller
    already computed.
    """
    from graphknows.symbolic.framenet.framenet import lus
    from graphknows.symbolic.framenet.index import match_frames, match_frames_text

    if doc is None:
        doc = nlp(text)
    lui = lus()

    lvc_triggers, lvc_verb_toks, lvc_noun_toks = _lvc_targets(doc, lui)  # type: ignore[arg-type]
    nominal_triggers = _nominalisation_targets(doc, lui, lvc_noun_toks)  # type: ignore[arg-type]
    # (doc index, is_lvc, candidate frames, trigger token) — sorted below.
    noun_raw: list[tuple[int, bool, set[str], Any]] = [
        (idx, True, cand, tok) for idx, cand, tok in lvc_triggers
    ] + [(idx, False, cand, tok) for idx, cand, tok in nominal_triggers]
    noun_raw.sort(key=lambda item: item[0])
    # ponytail: cap noun/LVC triggers per chunk at 4 to bound GLiNER2 calls on a
    # chunk with many nominalisations; raise it if an audit shows real
    # nominalisations being cut.
    noun_raw = noun_raw[:4]

    verbs, verb_cand = _verb_group(doc, lui, lvc_verb_toks)  # type: ignore[arg-type]

    # Nothing found yet — last-resort sweep (not run merely when verbs are absent,
    # or it would re-import the junk frames the LVC rule just removed).
    fallback_cand = _fallback_sweep(doc, lui) if not noun_raw and not verb_cand else set()  # type: ignore[arg-type]

    if not noun_raw and not verb_cand and not fallback_cand:
        return []

    # Rank every candidate frame by embedding fit, once, reusing the precomputed
    # chunk embedding when available (no re-embed), else embedding the text once.
    ranked = (
        match_frames(embedding, top_k=200, min_sim=0.0)
        if embedding is not None
        else match_frames_text(text, top_k=200, min_sim=0.0)
    )
    order = {name: sim for name, sim in ranked}

    def _rank(cand: set[str]) -> tuple[tuple[str, float], ...]:
        scored = sorted(cand, key=lambda f: -order.get(f, 0.0))[:top_k]
        return tuple((f, order.get(f, 0.0)) for f in scored)

    triggers: list[FrameTrigger] = []
    for _, is_lvc, cand, tok in noun_raw:
        ranked_cand = _rank(cand)
        if ranked_cand:
            triggers.append(
                FrameTrigger(
                    candidates=ranked_cand,
                    lvc=is_lvc,
                    trigger=str(getattr(tok, "text", "")),
                    context=_sent_text(tok),
                )
            )
    if verb_cand:
        ranked_cand = _rank(verb_cand)
        if ranked_cand:
            triggers.append(
                FrameTrigger(
                    candidates=ranked_cand,
                    lvc=False,
                    trigger=str(getattr(verbs[0], "text", "")),
                    context=_sent_text(verbs[0]),
                )
            )
    if not triggers and fallback_cand:
        ranked_cand = _rank(fallback_cand)
        if ranked_cand:
            triggers.append(FrameTrigger(candidates=ranked_cand, lvc=False))
    return triggers


@dataclass
class FrameInstance:
    """One evoked frame with its GLiNER2-filled roles (fillers are entity names)."""

    frame: str
    roles: dict[str, list[str]] = field(default_factory=dict)
    match_sim: float = 0.0
    role_confidence: float = 0.0
    # Where this instance's trigger sits in the chunk (SRL path only, else -1).
    # Carried so nesting — which is computed over trigger offsets — survives the
    # fillability gate dropping some instances.
    trigger_offset: int = -1
    # Trigger was a light-verb construction's noun object (e.g. "make a decision"
    # -> Deciding), not the frame's own lexical unit.
    lvc: bool = False

    @property
    def filled_core(self) -> int:
        """Number of core roles that have at least one filler."""
        return sum(1 for v in self.roles.values() if v)


def _fill_roles(
    text: str, frame: str, gliner: Any, speaker: str, speakers: list[str]
) -> tuple[dict[str, list[str]], float]:
    """Fill the frame's core FEs by typing spans with the FE names as labels.

    This is span typing, not relation extraction: the frame's core frame-element
    names ARE the label set, so a span labelled "Employee" is that role's filler.
    Runs on the shared extraction model (``extractor.gliner``) to avoid loading a
    second ~0.5B model per worker — which is why this must speak that model's API
    and not GLiNER2's ``create_schema``/``extract``.
    """
    spec = _role_schema(frame)
    if not spec:
        return {}, 0.0
    # With return_relations=False the model returns ONLY the entities list, not
    # an (entities, relations) tuple — unpacking blind raises ValueError.
    out = gliner.inference(
        texts=[text],
        labels=list(spec.keys()),
        relations=[],
        threshold=_ROLE_THRESHOLD,
        return_relations=False,
        flat_ner=False,
    )
    entities = out[0] if isinstance(out, tuple) else out
    spans = entities[0] if entities else []
    roles: dict[str, list[str]] = {}
    max_conf = 0.0
    for s in spans or []:
        role = str(s.get("label") or "").strip()
        surface = str(s.get("text") or "").strip()
        if not role or not surface:
            continue
        conf = float(s.get("score", 0.0) or 0.0)
        max_conf = max(max_conf, conf)
        # Resolve 1st/2nd-person fillers to the speaker (the dominant subject).
        resolved = _resolve_surface(surface, speaker, speakers)
        if resolved and resolved not in roles.setdefault(role, []):
            roles[role].append(resolved)
    return {r: f for r, f in roles.items() if f}, max_conf


_PRON = {"i", "me", "my", "myself", "we", "us", "our", "you", "your"}


def _resolve_surface(surface: str, speaker: str, speakers: list[str]) -> str:
    """Speaker-resolve a role filler when it's a bare 1st/2nd-person pronoun."""
    low = surface.lower().strip(" .,!?")
    if low in _PRON:

        class _T:  # minimal shim for _lingfeatures._resolve (expects .text)
            text = low

        return _resolve(_T(), speaker, speakers)
    return surface


def _passes_gate(inst: FrameInstance) -> bool:
    """The role-fillability gate: enough filled core roles, one confident fill.

    The one thing standing between this layer and a graph full of ``Trendiness``
    off "that's cool" — frame selection is unreliable on conversational filler
    and roles do not fill there. Applies per instance whichever identifier ran.
    """
    return inst.filled_core >= MIN_CORE_ROLES and inst.role_confidence >= ROLE_CONF_FLOOR


def extract_frame_instances_srl(
    text: str,
    speaker: str,
    speakers: list[str],
    gliner: Any,
    hits: list[Any],
) -> list[FrameInstance]:
    """One instance per parsed trigger (see :mod:`graphknows.symbolic.framenet.srl`).

    The difference from :func:`extract_frame_instances` is arity, not method:
    there the whole chunk gets one frame guess and therefore at most one kept
    instance; here every trigger the parser found is filled on its own
    trigger-marked line, so a chunk stating five facts can contribute five.
    Role filling, speaker resolution, and the fillability gate are shared.

    ``hits`` are :class:`~graphknows.symbolic.framenet.srl.FrameHit` (duck-typed to keep
    the optional extra out of this module's imports).
    """
    from graphknows.symbolic.framenet.srl import marked_line

    out: list[FrameInstance] = []
    for hit in hits:
        # The marked line, not the whole chunk: it scopes the roles to this
        # trigger's turn (keeping its speaker) and disambiguates two instances
        # of one frame on one line.
        context = marked_line(text, hit.offset, hit.trigger)
        roles, conf = _fill_roles(context, hit.frame, gliner, speaker, speakers)
        inst = FrameInstance(
            frame=hit.frame,
            roles=roles,
            match_sim=1.0,
            role_confidence=conf,
            trigger_offset=int(getattr(hit, "offset", -1)),
        )
        if _passes_gate(inst):
            out.append(inst)
    return out


def extract_frame_instances(
    text: str,
    speaker: str,
    speakers: list[str],
    gliner: Any,
    triggers: Callable[[str], list[FrameTrigger]],
    top_k: int = 3,
) -> list[FrameInstance]:
    """Frame-guided role extraction for one chunk/sentence.

    Args:
        text: the chunk text.
        speaker: the current utterance's speaker, for pronoun resolution.
        speakers: all speakers in the session, for pronoun resolution.
        gliner: the shared extraction model (``inference``); see _fill_roles.
        triggers: fn(text) -> [FrameTrigger, ...] (see :func:`lu_triggers`) — every
            frame-evoking element in the text, each with its own ranked candidates.
        top_k: how many of a trigger's candidate frames to attempt.

    A chunk can now contribute one instance per trigger, not exactly one: a
    nominalisation next to a verb, or a light verb's noun object, each gets its
    own independent shot. Per trigger, the existing best-of-top_k logic picks
    the candidate with the most filled core roles, then the role-fillability
    gate (>= MIN_CORE_ROLES filled, one fill >= ROLE_CONF_FLOOR) decides whether
    it survives. Usually 0 or 1 instance per trigger, 0+ per chunk.

    Role filling runs over ``trig.context`` (the trigger's own sentence) when
    known, not the whole chunk -- otherwise a multi-sentence chunk lets one
    trigger's roles fill from another clause entirely, and the arity change
    that makes each trigger its own instance turns that from a capped single
    error into one per trigger. Falls back to the whole chunk when the trigger
    carries no sentence (a verb group's representative token or a duck-typed
    token without ``.sent``).
    """
    out: list[FrameInstance] = []
    for trig in triggers(text):
        context = trig.context or text
        best: FrameInstance | None = None
        for frame, sim in trig.candidates[:top_k]:
            # A frame with fewer than MIN_CORE_ROLES core FEs can never clear the
            # gate below — skip the GLiNER2 call rather than pay for a fill that
            # is mathematically incapable of passing.
            if len(_role_schema(frame)) < MIN_CORE_ROLES:
                continue
            roles, max_conf = _fill_roles(context, frame, gliner, speaker, speakers)
            inst = FrameInstance(
                frame=frame, roles=roles, match_sim=sim, role_confidence=max_conf, lvc=trig.lvc
            )
            if best is None or inst.filled_core > best.filled_core:
                best = inst
        if best is not None and _passes_gate(best):
            out.append(best)
    return out
