"""Vocabulary constants for the SpaCy intentionality-relation extractor."""

from __future__ import annotations

_ASSERTIVE = "ASSERTS"
_EPISTEMIC = "BELIEVES"
_CONATIVE = "INTENDS"
_AFFECTIVE = "FEELS"
_PERCEPTUAL = "PERCEIVES"
_DOUBTING = "DOUBTS"
_ILLOCUTIONARY_DEC = "DECLARES"
_ILLOCUTIONARY_PRO = "PROMISES"
_ILLOCUTIONARY_COM = "COMMANDS"
_ILLOCUTIONARY_EXP = "EXPRESSES"  # Searle expressives: thank, apologize, congratulate
_REFERENTIAL = "REFERS_TO"
_DENOTATIONAL = "DENOTES"
_EXPRESSIVE = "EXPRESSES"
_PRESUPPOSITIONAL = "PRESUPPOSES"
_DEFINITIONAL = "DEFINED_AS"
_DISTINCTION = "DISTINGUISHES"
_EXEMPLIFICATION = "EXEMPLIFIES"
_ENTAILMENT = "ENTAILS"
_SUPPORTIVE = "SUPPORTS"
_EXPLANATORY = "EXPLAINS"
_CONTRADICTION = "CONTRADICTS"
_GROUNDING = "GROUNDS"
_CRITICAL = "CRITIQUES"
_DEFENSIVE = "DEFENDS"

COGNITIVE_VERB_MAP: dict[str, str] = {
    # ── Searle: ASSERTIVES — speaker commits to truth of p ────────────────────
    # Verba dicendi: the complete English semantic class of saying/reporting verbs
    "say": _ASSERTIVE,
    "tell": _ASSERTIVE,
    "state": _ASSERTIVE,
    "claim": _ASSERTIVE,
    "assert": _ASSERTIVE,
    "report": _ASSERTIVE,
    "announce": _ASSERTIVE,
    "mention": _ASSERTIVE,
    "note": _ASSERTIVE,
    "remark": _ASSERTIVE,
    "describe": _ASSERTIVE,
    "specify": _ASSERTIVE,
    "narrate": _ASSERTIVE,
    "allege": _ASSERTIVE,
    "submit": _ASSERTIVE,
    "affirm": _ASSERTIVE,
    "aver": _ASSERTIVE,
    "attest": _ASSERTIVE,
    "inform": _ASSERTIVE,
    "reply": _ASSERTIVE,
    "answer": _ASSERTIVE,
    "reveal": _ASSERTIVE,
    "disclose": _ASSERTIVE,
    "divulge": _ASSERTIVE,
    "publish": _ASSERTIVE,
    # Epistemic commitment expressed as public speech act
    "acknowledge": _ASSERTIVE,
    "admit": _ASSERTIVE,
    "concede": _ASSERTIVE,
    "grant": _ASSERTIVE,
    "confess": _ASSERTIVE,
    # Argumentative / evidential assertives
    "argue": _ASSERTIVE,
    "maintain": _ASSERTIVE,
    "contend": _ASSERTIVE,
    "insist": _ASSERTIVE,
    "hold": _ASSERTIVE,
    "posit": _ASSERTIVE,
    "hypothesize": _ASSERTIVE,
    "theorize": _ASSERTIVE,
    "conjecture": _ASSERTIVE,
    "propose": _ASSERTIVE,
    "suggest": _ASSERTIVE,
    "show": _ASSERTIVE,
    "demonstrate": _ASSERTIVE,
    "prove": _ASSERTIVE,
    "establish": _ASSERTIVE,
    "conclude": _ASSERTIVE,
    "find": _ASSERTIVE,
    "assume": _ASSERTIVE,
    "presume": _ASSERTIVE,
    "suppose": _ASSERTIVE,
    "predict": _ASSERTIVE,
    "forecast": _ASSERTIVE,
    "anticipate": _ASSERTIVE,
    # ── Brentano: DOXASTIC — propositional attitude of belief / knowledge ─────
    "believe": _EPISTEMIC,
    "know": _EPISTEMIC,
    "think": _EPISTEMIC,
    "recognize": _EPISTEMIC,
    "understand": _EPISTEMIC,
    "suspect": _EPISTEMIC,
    # ── Brentano: CONATIVE — propositional attitude of desire / intention ─────
    "intend": _CONATIVE,
    "plan": _CONATIVE,
    "aim": _CONATIVE,
    "seek": _CONATIVE,
    "want": _CONATIVE,
    "try": _CONATIVE,
    "attempt": _CONATIVE,
    "mean": _CONATIVE,
    # ── Brentano: AFFECTIVE — propositional attitude of emotion ──────────────
    "fear": _AFFECTIVE,
    "resent": _AFFECTIVE,
    "admire": _AFFECTIVE,
    "love": _AFFECTIVE,
    "hate": _AFFECTIVE,
    "dread": _AFFECTIVE,
    "hope": _AFFECTIVE,
    "worry": _AFFECTIVE,
    "regret": _AFFECTIVE,
    # ── Brentano: PERCEPTUAL ─────────────────────────────────────────────────
    "perceive": _PERCEPTUAL,
    "observe": _PERCEPTUAL,
    "see": _PERCEPTUAL,
    "notice": _PERCEPTUAL,
    "detect": _PERCEPTUAL,
    # ── Epistemic DOUBT (negated assertive commitment) ───────────────────────
    "doubt": _DOUBTING,
    "question": _DOUBTING,
    "deny": _DOUBTING,
    # ── Searle: DIRECTIVES — speaker directs hearer to bring about p ─────────
    "order": _ILLOCUTIONARY_COM,
    "command": _ILLOCUTIONARY_COM,
    "demand": _ILLOCUTIONARY_COM,
    "require": _ILLOCUTIONARY_COM,
    "request": _ILLOCUTIONARY_COM,
    "beg": _ILLOCUTIONARY_COM,
    "urge": _ILLOCUTIONARY_COM,
    "instruct": _ILLOCUTIONARY_COM,
    "direct": _ILLOCUTIONARY_COM,
    "invite": _ILLOCUTIONARY_COM,
    "implore": _ILLOCUTIONARY_COM,
    "entreat": _ILLOCUTIONARY_COM,
    "petition": _ILLOCUTIONARY_COM,
    "advise": _ILLOCUTIONARY_COM,
    "recommend": _ILLOCUTIONARY_COM,
    "warn": _ILLOCUTIONARY_COM,
    "encourage": _ILLOCUTIONARY_COM,
    "discourage": _ILLOCUTIONARY_COM,
    "forbid": _ILLOCUTIONARY_COM,
    "prohibit": _ILLOCUTIONARY_COM,
    "permit": _ILLOCUTIONARY_COM,
    "allow": _ILLOCUTIONARY_COM,
    # ── Searle: COMMISSIVES — speaker commits to future action ───────────────
    "promise": _ILLOCUTIONARY_PRO,
    "pledge": _ILLOCUTIONARY_PRO,
    "vow": _ILLOCUTIONARY_PRO,
    "swear": _ILLOCUTIONARY_PRO,
    "guarantee": _ILLOCUTIONARY_PRO,
    "warrant": _ILLOCUTIONARY_PRO,
    "undertake": _ILLOCUTIONARY_PRO,
    "commit": _ILLOCUTIONARY_PRO,
    "offer": _ILLOCUTIONARY_PRO,
    "volunteer": _ILLOCUTIONARY_PRO,
    "threaten": _ILLOCUTIONARY_PRO,
    # ── Searle: EXPRESSIVES — speaker expresses psychological state about p ──
    "thank": _ILLOCUTIONARY_EXP,
    "congratulate": _ILLOCUTIONARY_EXP,
    "praise": _ILLOCUTIONARY_EXP,
    "compliment": _ILLOCUTIONARY_EXP,
    "welcome": _ILLOCUTIONARY_EXP,
    "condole": _ILLOCUTIONARY_EXP,
    "sympathize": _ILLOCUTIONARY_EXP,
    "lament": _ILLOCUTIONARY_EXP,
    "deplore": _ILLOCUTIONARY_EXP,
    "apologize": _ILLOCUTIONARY_EXP,
    # ── Searle: DECLARATIVES — speaker changes institutional reality ──────────
    "declare": _ILLOCUTIONARY_DEC,
    "proclaim": _ILLOCUTIONARY_DEC,
    "rule": _ILLOCUTIONARY_DEC,
    "pronounce": _ILLOCUTIONARY_DEC,
    "adjourn": _ILLOCUTIONARY_DEC,
    "appoint": _ILLOCUTIONARY_DEC,
    "nominate": _ILLOCUTIONARY_DEC,
    "designate": _ILLOCUTIONARY_DEC,
    "elect": _ILLOCUTIONARY_DEC,
    "hire": _ILLOCUTIONARY_DEC,
    "dismiss": _ILLOCUTIONARY_DEC,
    "fire": _ILLOCUTIONARY_DEC,
    "sentence": _ILLOCUTIONARY_DEC,
    "acquit": _ILLOCUTIONARY_DEC,
    "convict": _ILLOCUTIONARY_DEC,
    "enact": _ILLOCUTIONARY_DEC,
    "ratify": _ILLOCUTIONARY_DEC,
    "veto": _ILLOCUTIONARY_DEC,
    "resign": _ILLOCUTIONARY_DEC,
    "inaugurate": _ILLOCUTIONARY_DEC,
    # ── Semiotic: how signs relate to their objects (Peirce semiosis) ─────────
    "refer": _REFERENTIAL,
    "point": _REFERENTIAL,
    "denote": _DENOTATIONAL,
    "name": _DENOTATIONAL,
    "signify": _DENOTATIONAL,
    "represent": _EXPRESSIVE,
    "convey": _EXPRESSIVE,
    "encode": _EXPRESSIVE,
    "express": _EXPRESSIVE,
    "presuppose": _PRESUPPOSITIONAL,
    # ── Definitional / taxonomic ──────────────────────────────────────────────
    "define": _DEFINITIONAL,
    "characterize": _DEFINITIONAL,
    "constitute": _DEFINITIONAL,
    # ── Distinction ──────────────────────────────────────────────────────────
    "distinguish": _DISTINCTION,
    "differentiate": _DISTINCTION,
    "contrast": _DISTINCTION,
    "separate": _DISTINCTION,
    "compare": _DISTINCTION,
    # ── Exemplification ──────────────────────────────────────────────────────
    "exhibit": _EXEMPLIFICATION,
    "exemplify": _EXEMPLIFICATION,
    "illustrate": _EXEMPLIFICATION,
    "instantiate": _EXEMPLIFICATION,
    "display": _EXEMPLIFICATION,
    # ── Peirce: DEDUCTION — what necessarily follows ──────────────────────────
    "entail": _ENTAILMENT,
    "imply": _ENTAILMENT,
    "follow": _ENTAILMENT,
    # ── Peirce: INDUCTION — what the evidence supports ───────────────────────
    "support": _SUPPORTIVE,
    "confirm": _SUPPORTIVE,
    "corroborate": _SUPPORTIVE,
    "indicate": _SUPPORTIVE,
    # ── Peirce: ABDUCTION — what best explains ───────────────────────────────
    "explain": _EXPLANATORY,
    "account": _EXPLANATORY,
    "cause": _EXPLANATORY,
    # ── Meta-reasoning ───────────────────────────────────────────────────────
    "contradict": _CONTRADICTION,
    "conflict": _CONTRADICTION,
    "ground": _GROUNDING,
    # ── Critical / defensive ─────────────────────────────────────────────────
    "critique": _CRITICAL,
    "criticize": _CRITICAL,
    "reject": _CRITICAL,
    "refute": _CRITICAL,
    "challenge": _CRITICAL,
    "oppose": _CRITICAL,
    "defend": _DEFENSIVE,
    "advocate": _DEFENSIVE,
}

_COPULA_LEMMAS: frozenset[str] = frozenset({"be", "become", "remain"})
_DEFINITIONAL_LEMMAS: frozenset[str] = frozenset(
    {"define", "characterize", "constitute", "mean", "represent"}
)

# Stratum IV — directive illocution: deontic modals
_DEONTIC_MODALS: frozenset[str] = frozenset({"must", "shall", "should", "ought"})

# Stratum V — Peirce abductive connectives (causal "mark" dep → EXPLAINS)
_CAUSAL_MARKS: frozenset[str] = frozenset({"because", "since"})

# Stratum V — Peirce deductive connectives (discourse marker → ENTAILS)
# Inductive ("for example", "such as") is covered verbally via COGNITIVE_VERB_MAP
# (support/confirm/corroborate/indicate) but lacks a discourse-marker pattern.
_ENTAILMENT_CONNECTIVES: frozenset[str] = frozenset(
    {"therefore", "thus", "hence", "consequently", "accordingly", "ergo"}
)

# SpaCy NER type → intentionality label fallback
_SPACY_NER_TO_LABEL: dict[str, str] = {
    "PERSON": "AGENT",
    "ORG": "AGENT",
    "NORP": "AGENT",
    "FAC": "LOCATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "PRODUCT": "ARTIFACT",
    "WORK_OF_ART": "ARTIFACT",
    "LAW": "ARTIFACT",
    "EVENT": "EVENT",
    "DATE": "TIME",
    "TIME": "TIME",
    "MONEY": "CONCEPT",
    "PERCENT": "CONCEPT",
    "QUANTITY": "CONCEPT",
}
