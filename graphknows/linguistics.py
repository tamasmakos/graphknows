"""Polarity, modality and epistemic status from a dependency parse.

A frame instance records *that* something was said. Without this module it does
not record whether it was denied, hypothesised, hedged, or attributed to someone
else — so "she said she might move" is stored as "she moved", and a question
about what actually happened gets a confident wrong answer.

Everything here is a rule over a spaCy ``Doc`` that has already been parsed
upstream. **No model is loaded and no text is re-parsed**: the caller passes the
``Doc`` it already has, which is what keeps this off the per-chunk cost budget.

The labels are spaCy's English set (ClearNLP/OntoNotes: ``nsubjpass``, ``dobj``,
``attr``, ``acomp``, ``agent``), *not* Universal Dependencies. A UD-trained
pipeline would need ``nsubj:pass``/``obj`` instead.

Every result carries ``rules`` — which rules fired. These values look
authoritative once they are graph properties, and a heuristic over an imperfect
parse should be auditable rather than anonymous. The defaults are also the
*status quo* (asserted, positive, active), so a missed cue degrades to today's
behaviour instead of to a new wrong claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Modal auxiliaries -> the flavour of modality they carry. "would"/"will" are
# intention rather than possibility: "I would move to Berlin" is a stated plan,
# not a hedge, and conflating them loses a distinction a memory system needs.
_MODALS: dict[str, str] = {
    "can": "ability",
    "could": "possibility",
    "may": "possibility",
    "might": "possibility",
    "must": "obligation",
    "should": "obligation",
    "ought": "obligation",
    "shall": "obligation",
    "will": "intention",
    "would": "intention",
}

# Saying-verbs put their complement in someone else's mouth: the content becomes
# a report, and the matrix subject is who it is attributed to.
_COMMUNICATION: frozenset[str] = frozenset(
    {
        "say",
        "tell",
        "mention",
        "claim",
        "report",
        "explain",
        "add",
        "note",
        "admit",
        "argue",
        "insist",
        "reply",
        "answer",
        "announce",
        "write",
        "suggest",
        "complain",
        "promise",
        "warn",
    }
)

# Cognition verbs are NOT reporting: "I think she moved" is the speaker hedging
# their own claim, not attributing it elsewhere. Treated as a hedge.
_COGNITION: frozenset[str] = frozenset(
    {"think", "believe", "feel", "guess", "suppose", "reckon", "assume", "suspect"}
)

# Desire/intention verbs: the complement is wanted, not asserted to have happened.
_DESIRE: frozenset[str] = frozenset({"want", "hope", "wish", "plan", "intend", "aim", "consider"})

_HEDGES: frozenset[str] = frozenset(
    {
        "maybe",
        "perhaps",
        "probably",
        "possibly",
        "apparently",
        "seemingly",
        "allegedly",
        "arguably",
        "presumably",
        "supposedly",
        "likely",
        "unlikely",
        "seem",
        "appear",
    }
)

# Negation that arrives as a determiner or adverb rather than a `neg` arc.
_NEGATORS: frozenset[str] = frozenset(
    {"no", "never", "none", "nothing", "nobody", "neither", "nor"}
)

_CONDITIONAL_MARKS: frozenset[str] = frozenset({"if", "unless", "whether"})


@dataclass(frozen=True, slots=True)
class Modality:
    """How a frame instance is asserted. Every default is the status quo."""

    polarity: str = "pos"  # pos | neg
    modality: str = "none"  # none | ability | permission | obligation | possibility | intention
    epistemic: str = "asserted"  # asserted | conditional | counterfactual | reported | desired
    hedged: bool = False
    voice: str = "active"  # active | passive
    tense: str = ""  # past | pres | ""
    attributed_to: str = ""
    rules: tuple[str, ...] = field(default_factory=tuple)

    def as_props(self) -> dict[str, Any]:
        """The graph-property form written onto FRAME_INSTANCE and derived REL."""
        return {
            "polarity": self.polarity,
            "modality": self.modality,
            "epistemic": self.epistemic,
            "hedged": self.hedged,
            "voice": self.voice,
            "tense": self.tense,
            "attributed_to": self.attributed_to,
            "modality_rules": ",".join(self.rules),
        }


def _token_at(doc: Any, offset: int) -> Any:
    """The token covering character *offset*, or None."""
    if offset < 0:
        return None
    for tok in doc:
        if tok.idx <= offset < tok.idx + len(tok.text):
            return tok
    return None


def _clause_head(tok: Any) -> Any:
    """Climb to the verb heading this token's own clause."""
    cur = tok
    seen = {tok.i}
    while cur.head is not cur and cur.pos_ not in ("VERB", "AUX"):
        if cur.head.i in seen:
            break
        cur = cur.head
        seen.add(cur.i)
    return cur


def _ancestors(tok: Any) -> list[Any]:
    """The token's head chain, outermost last. Guarded against parse cycles."""
    out: list[Any] = []
    cur, seen = tok, {tok.i}
    while cur.head is not cur and cur.head.i not in seen:
        cur = cur.head
        seen.add(cur.i)
        out.append(cur)
    return out


def _keys(tok: Any) -> set[str]:
    """Lemma plus surface forms, for matching against the closed verb classes.

    ``en_core_web_lg`` lemmatizes "hopes" to "hop", so a lemma-only lookup misses
    the desire verb outright and "Melanie hopes to move" is stored as an
    assertion that she moved. Checking the surface form and a naive -s strip too
    costs nothing and covers the whole family of mis-lemmatisations rather than
    that one word.
    """
    surface = tok.text.lower()
    keys = {tok.lemma_.lower(), surface}
    if surface.endswith("s") and len(surface) > 3:
        keys.add(surface[:-1])
    return keys


def _polarity(kids: list[Any], rules: list[str]) -> str:
    if any(c.dep_ == "neg" for c in kids):
        rules.append("neg-dep")
        return "neg"
    if any(c.lemma_.lower() in _NEGATORS for c in kids):
        rules.append("neg-lexical")
        return "neg"
    return "pos"


def _modality(kids: list[Any], rules: list[str]) -> str:
    for c in kids:
        if c.dep_ in ("aux", "auxpass"):
            flavour = _MODALS.get(c.lemma_.lower())
            if flavour:
                rules.append(f"modal:{c.lemma_.lower()}")
                return flavour
    return "none"


def _is_conditional(verb: Any, kids: list[Any]) -> bool:
    if any(c.dep_ == "mark" and c.lemma_.lower() in _CONDITIONAL_MARKS for c in kids):
        return True
    return any(
        a.dep_ == "advcl"
        and any(m.dep_ == "mark" and m.lemma_.lower() in _CONDITIONAL_MARKS for m in a.children)
        for a in [verb, *_ancestors(verb)]
    )


def _governing_status(verb: Any, rules: list[str]) -> tuple[str, str]:
    """``(epistemic, attributed_to)`` from whatever governs this clause."""
    for anc in _ancestors(verb):
        if anc.pos_ not in ("VERB", "AUX"):
            continue
        keys = _keys(anc)
        lemma = anc.lemma_.lower()
        if _COMMUNICATION & keys:
            subj = next((c for c in anc.children if c.dep_ in ("nsubj", "nsubjpass")), None)
            rules.append(f"reported:{lemma}")
            return "reported", (subj.text if subj is not None else "")
        if _DESIRE & keys:
            rules.append(f"desired:{lemma}")
            return "desired", ""
        if _COGNITION & keys:
            rules.append(f"cognition:{lemma}")
            return "asserted", ""
    return "asserted", ""


def analyse(doc: Any, trigger_offset: int) -> Modality:
    """Classify how the frame triggered at *trigger_offset* is asserted.

    A negative offset means the caller has no trigger — the non-SRL frame path
    guesses one frame per chunk with no span, so there is nothing to anchor a
    clause-scoped rule to. Returns the defaults with ``rules=('no-trigger',)``
    rather than analysing an arbitrary token: an absent modality is better than
    a wrong one.
    """
    trigger = _token_at(doc, trigger_offset)
    if trigger is None:
        return Modality(rules=("no-trigger",))

    verb = _clause_head(trigger)
    kids = list(verb.children)
    rules: list[str] = []

    polarity = _polarity(kids, rules)
    modality = _modality(kids, rules)

    voice = "active"
    if any(c.dep_ in ("nsubjpass", "auxpass") for c in kids):
        voice = "passive"
        rules.append("passive")

    tense = ""
    morph_tense = verb.morph.get("Tense")
    if morph_tense:
        tense = str(morph_tense[0]).lower()[:4]
        rules.append(f"tense:{tense}")

    epistemic, attributed_to = "asserted", ""
    if _is_conditional(verb, kids):
        epistemic = "conditional"
        rules.append("conditional")
        # "would have moved" — a conditional whose modal is past-shifted is
        # counterfactual: it asserts the thing did NOT happen.
        if modality == "intention" and any(c.lemma_.lower() == "have" for c in kids):
            epistemic = "counterfactual"
            rules.append("counterfactual")
    else:
        epistemic, attributed_to = _governing_status(verb, rules)

    hedged = any(_HEDGES & _keys(t) for t in trigger.sent) or any(
        r.startswith("cognition:") for r in rules
    )
    if hedged:
        rules.append("hedged")

    return Modality(
        polarity=polarity,
        modality=modality,
        epistemic=epistemic,
        hedged=hedged,
        voice=voice,
        tense=tense,
        attributed_to=attributed_to,
        rules=tuple(rules),
    )
