"""The junk gate every ENTITY-minting path shares.

``is_graph_entity_name`` is the single predicate deciding whether a span belongs
in the ENTITY graph; ``graph_entities`` applies it to an extraction's entity
list. This is the extraction layer's gate, and the only place a name is judged:
every write path (entity/MENTIONS, relation endpoints, frame role fillers) runs
a name through it before an ENTITY node exists at all.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# spaCy NER labels that are numeric/temporal noise as graph entities. Dates and
# times are captured separately as canonical TEMPORAL nodes; numbers/percents/
# money saturate the entity graph and pollute community/topic membership without
# ever serving as useful retrieval discriminators.
_NON_GRAPH_ENTITY_TYPES = frozenset(
    {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}
)

# Titles of creative works are conventionally verb-headed ("Saving Private
# Ryan", "The Empire Strikes Back", "Live and Let Die") -- that is what makes
# them titles, not predications. The verb-head gate exists to catch
# predications with no nominal referent (#160); a typed title already has a
# referent, so it is exempt regardless of its syntactic head.
_VERB_HEAD_EXEMPT_TYPES = frozenset({"WORK_OF_ART"})

# GLiNER often leaves date/time/number spans UNTYPED (type=None), so the type
# allowlist above misses them and they leak in as ENTITY nodes (e.g. "23 July
# 2023", "10:31 am", "2023") — corpus-polluting noise that also corrupts
# community detection. This pattern catches them by SHAPE. It requires a digit
# alongside any month word, so real person/place names ("May", "June") survive.
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_TIME = r"\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?"
_TEMPORAL_NAME_RE = re.compile(
    r"^("
    r"[\W\d]+"  # bare numbers / punctuation / "10:31" / "100" / "4"
    rf"|{_TIME}(?:\s+on\s+.{{0,40}})?"  # "10:31 am" / "1:56 pm on 8 May, 2023"
    rf"|\d{{1,4}}\s+{_MONTH}(?:,?\s+\d{{1,4}})?"  # "23 July 2023" / "8 May, 2023"
    # month-first, with optional day, an optional "- <month> <day>" range end,
    # and an optional trailing year: "July 2023" / "December 15, 2023" /
    # "November 1 - November 15, 2023" / "January 16 - February 15, 2024".
    # RUF001 suppressed: the en/em dashes are the literal characters real date
    # ranges are written with — replacing them with a hyphen breaks the match.
    rf"|{_MONTH}\s+\d{{1,4}}(?:\s*[-–—]\s*(?:{_MONTH}\s+)?\d{{1,4}})?(?:,?\s+\d{{2,4}})?"  # noqa: RUF001
    r"|\d{4}"  # bare year
    r")$",
    re.IGNORECASE,
)

# Relative-date shapes carry no digit, so they need their own branch rather than
# a loosened digit rule (which would kill the person names "May"/"June").
# Weekday names — unlike month names — are not person names, so bare weekdays
# are safe to drop.
_WEEKDAY = r"(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|satur|sun)(?:day)?s?"
_RELATIVE_DATE_RE = re.compile(
    rf"^(?:(?:last|next|this|every)\s+)?"
    rf"(?:{_WEEKDAY}|week(?:end)?|month|year|morning|afternoon|evening|night|"
    rf"yesterday|today|tomorrow)$",
    re.IGNORECASE,
)


def _is_temporal_or_numeric_name(name: str) -> bool:
    stripped = (name or "").strip()
    return bool(_TEMPORAL_NAME_RE.match(stripped)) or bool(_RELATIVE_DATE_RE.match(stripped))


# Closed-class function words (articles, prepositions, conjunctions, pronouns,
# auxiliaries, determiners, wh-words). GLiNER over-generates these on any text —
# a census of two live graphs found "the"/"and"/"of"/"be"/"these" minted as
# ENTITY nodes. They are never entities by grammar. Content words that merely
# happen to be generic ("project", "database") are deliberately NOT listed:
# retrieval already de-weights high-frequency entities (_discriminative_entities),
# so this set is only the closed class, where a drop can never lose a real entity.
_STOPWORD_NAMES = frozenset(
    """
    a an the this that these those such
    and or but nor so yet
    of to in on at by for from with without into onto over under about
    is are was were be been being am
    have has had do does did done
    would shall should can could might must ought
    it its he she him her they them we us you i me my your his their our
    as if then than there here not no yes
    what which who whom whose when where why how
    all any some each every both either neither few many much more most other
    """.split()
)

# Code, markup, URLs: GLiNER on technical text mints spans like
# "http://localhost:5000" or "Flask application: ```python from flask import".
_CODE_MARKUP_RE = re.compile(r"```|https?://|www\.|://|[{}<>=|]|::|\(\)|\[\]")
# A sentence ending INSIDE a span: three or more letters, a period, then a new
# capitalised word. Abbreviations ("Dr. Nagy", "U.S. Army", "J. R. Ewing") have
# one or two letters before the period and are deliberately not matched.
_SENTENCE_BREAK_RE = re.compile(r"[A-Za-z]{3,}\.\s+[A-Z]")
# A proper entity is a noun phrase, not a sentence or a heading — cap the words.
#
# DECIDED (#160): the cap STAYS at 8. Lowering it to 3 was already measured to
# delete legitimate 4-word titles ("Charlotte's Web: A Review"), and length was
# only ever standing in for the clause case — a predication minted as an ENTITY
# because the string rules cannot see its syntactic head. That case is now
# decided by the span's head POS instead (``_is_verb_headed``), not by length.
# Re-measure this decision against the per-ingest
# ``abstentions["untyped_entities"]/["entities"]`` share and the flush's
# ``entity_coverage["isolated"]`` count landed under this same issue.
_MAX_ENTITY_WORDS = 8


# Generic non-referential nouns that object extraction surfaces as junk hub
# entities ("others", "many", "kind"): a node standing for "thing" answers no
# question about anything. Dropped at the mint point, which is the only place a
# name is judged now that the flush-time resolver is gone.
_JUNK_ENTITIES = frozenset(
    """
    others many kind thing things everyone someone anyone everything nothing
    lot lots one ones some all none way ways bit part
    """.split()
)

# Pronouns and auxiliaries that can only ever OPEN a clause, never head a noun
# phrase. A multi-word name starting with one of these is a sentence fragment
# ("I can teach", "it looks awesome", "since I was a kid"), not an entity.
#
# Deliberately excludes articles and prepositions: "the White House" and "of
# Mice and Men" are real names that open with closed-class words, so gating on
# those would delete legitimate entities. Pronoun/auxiliary/subordinator-initial
# is the subset where a drop cannot lose a real name.
_CLAUSE_OPENERS = frozenset(
    """
    i me my mine myself we us our ours ourselves
    you your yours yourself yourselves
    he him his she her hers it its they them their theirs em
    am is are was were be been being
    do does did done have has had
    can could will would shall should may might must
    lets let
    since because although though unless whenever wherever whereas
    """.split()
)


def _is_junk_hub(name: str) -> bool:
    """A generic non-referential noun — a node standing for nothing in particular."""
    return name.strip().casefold() in _JUNK_ENTITIES


#: Coordinators that join two things. A name built around one names a
#: CONJUNCTION of things, and a node standing for two things answers questions
#: about neither.
_COORDINATORS = frozenset({"and", "or"})


def _is_coordinated_phrase(name: str) -> bool:
    """An all-lowercase phrase joining two things with "and"/"or" — not one thing.

    Two of the three spans #160 recorded as admitted defects are this shape
    ("one-on-one mentoring and training", "new offers and promotions"), and the
    LLM decoder mints more of them: measured on conv-26/conv-30, "equality and
    inclusivity", "love and diversity", "inclusivity and support", "counseling
    or working in mental health". A head-gate cannot reach any of them — the
    ROOT parses as a NOUN — so the coordination itself is the signal.

    Gated on the name carrying NO capital letter, which is what keeps real names
    off it: "Barnes and Noble", "Crosby, Stills and Nash", "Beauty and the
    Beast" all capitalise. The cost is an all-lowercase genre like "rock and
    roll", which this rejects; against the junk it removes that trade is worth
    making, and a capitalised writing of the same name survives.
    """
    tokens = name.split()
    if len(tokens) < 3 or any(char.isupper() for char in name):
        return False
    return any(token.casefold() in _COORDINATORS for token in tokens[1:-1])


def _is_clause_fragment(name: str) -> bool:
    """A multi-word span that opens like a clause, or ends like a sentence.

    Two independent signals, both grammatical rather than heuristic:

    * a name containing ``!`` or ``?`` is an utterance, never a noun phrase.
      Terminally it ends one ("Hey Gina!"); INTERNALLY it spans a sentence
      boundary, so the span is two fragments and not a phrase at all. Measured
      on LoCoMo conv-30: "Gina! I", from "Thanks, Gina! I won't quit." spoken
      by Jon, was minted PERSON and carried 11 of the graph's 43 relation edges
      — a node that is neither speaker absorbing Gina's first-person clauses;
    * a multi-word name whose first token is a pronoun, auxiliary or
      subordinator is a clause fragment — the extractor caught a predicate and
      called it a thing.
    """
    stripped = name.strip()
    if "!" in stripped or "?" in stripped:
        return True
    tokens = stripped.split()
    if not tokens:
        return False
    first = tokens[0].casefold().strip("\"'")
    # Contractions carry the opener in the stem: "I'm" -> i, "we've" -> we,
    # "let's" -> let, "'em" -> (leading quote stripped above) em.
    stem = first.split("'")[0] if "'" in first else first
    if len(tokens) >= 2:
        return stem in _CLAUSE_OPENERS
    # A lone contraction is a pronoun+auxiliary, never a name: "I'm", "we've".
    return "'" in tokens[0] and (stem in _CLAUSE_OPENERS or stem == "")


def _spans_a_sentence_boundary(name: str) -> bool:
    """A span running across a sentence break — two fragments, not a name.

    The sibling of ``_is_clause_fragment``'s ``!``/``?`` rule, for the boundary
    that rule cannot see: a model given several consecutive segments can open a
    span in one and close it in the next.

    A bare ``.`` is NOT enough, because real names carry them ("Dr. Nagy",
    "U.S. Army", "J. R. Ewing"). The signal is a sentence-ending period — a word
    of three or more letters, then ``.``, then whitespace and a capital — which
    no abbreviation produces.
    """
    stripped = name.strip()
    return "\n" in stripped or bool(_SENTENCE_BREAK_RE.search(stripped))


def _is_stopword_name(name: str) -> bool:
    stripped = name.strip()
    # Acronym-shaped tokens (all-caps, 2-3 chars) collide with function words
    # once casefolded — "US"/"IT"/"UK"/"OR" vs "us"/"it"/... — so keep them.
    if stripped.isupper() and 2 <= len(stripped) <= 3:
        return False
    return stripped.casefold() in _STOPWORD_NAMES


def _is_code_or_clause(name: str) -> bool:
    stripped = name.strip()
    return bool(_CODE_MARKUP_RE.search(stripped)) or len(stripped.split()) > _MAX_ENTITY_WORDS


_NLP_UNAVAILABLE = False


def _parser() -> Any:
    """The shared spaCy pipeline, memoized by :func:`graphknows.nlp.load_spacy_model`.

    Soft-fails: a missing model logs once at WARNING and returns ``None`` rather
    than raising, so this gate degrades to "admit" instead of emptying the graph.
    """
    global _NLP_UNAVAILABLE
    if _NLP_UNAVAILABLE:
        return None
    try:
        from graphknows.nlp import load_spacy_model
        from graphknows.settings import get_settings

        return load_spacy_model(get_settings().spacy_model)
    except Exception as exc:  # a missing model must not take ingest down
        _NLP_UNAVAILABLE = True
        log.warning("Verb-head entity gate skipped: spaCy unavailable (%s)", exc)
        return None


@functools.lru_cache(maxsize=4096)
def _is_verb_headed(name: str) -> bool:
    """A span of 3+ tokens whose syntactic ROOT is a verb is a predication, not a name.

    Only applied to names of three or more whitespace tokens: one- and two-token
    names ("dance studio", "New York") are the overwhelming majority of real
    entities and are POS-ambiguous in isolation, and none of the measured
    propositions from #160 is shorter than four tokens. Returns False (admit)
    when the parser is unavailable or no ROOT token is found — today's
    behaviour, unchanged.
    """
    tokens = name.split()
    if len(tokens) < 3:
        return False
    nlp = _parser()
    if nlp is None:
        return False
    doc = nlp(" ".join(tokens))
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return False
    return root.pos_ in {"VERB", "AUX"}


def _is_degenerate_name(name: str) -> bool:
    """Single characters, punctuation-only, and 1-2 char lowercase fragments.

    "db"/"id"/"io"/"to" are noise; an uppercase short form ("AI", "US") is a
    real acronym and kept, so the length-2 drop requires the token be lowercase.
    """
    stripped = name.strip()
    if len(stripped) <= 1:
        return True
    if not any(c.isalpha() for c in stripped):  # pure punctuation / digits / symbols
        return True
    return len(stripped) <= 2 and stripped == stripped.lower()


#: The shape rejections, in the order they were added. A tuple rather than a
#: chain of ``if``s so that adding one is a line here instead of a branch in
#: :func:`is_graph_entity_name` — which the complexity gate caps, and which is
#: the reason the list reads as a list of reasons.
_REJECTS = (
    _is_stopword_name,
    _is_junk_hub,
    _is_clause_fragment,
    _spans_a_sentence_boundary,
    _is_coordinated_phrase,
    _is_degenerate_name,
    _is_code_or_clause,
    _is_temporal_or_numeric_name,
)


def is_graph_entity_name(name: str, ent_type: str = "") -> bool:
    """Whether ``name`` belongs in the ENTITY graph — the ONE junk predicate.

    False for empty, stopword, degenerate (single/short/punctuation), code/markup,
    over-long clause, and temporal/numeric names (by type or by shape). Exported
    so every minting path applies the same rule: ``graph_entities`` for the
    entity/MENTIONS path, and the relation gate in ``_write_extracted`` for
    relation endpoints, which previously bypassed all filtering and minted date
    ENTITY nodes (which then received a community_id and fragmented the topics).
    """
    stripped = (name or "").strip()
    if not stripped:
        return False
    if (ent_type or "").upper() in _NON_GRAPH_ENTITY_TYPES:
        return False
    if any(reject(stripped) for reject in _REJECTS):
        return False
    if (ent_type or "").upper() in _VERB_HEAD_EXEMPT_TYPES:
        return True
    return not _is_verb_headed(" ".join(stripped.split()))


def graph_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop numeric/temporal spans that should not become ENTITY nodes.

    Filters by spaCy/GLiNER type AND by name shape — the latter catches the
    common case where the extractor leaves a date/time/number untyped.
    """
    return [e for e in entities if is_graph_entity_name(e.get("name", ""), e.get("type") or "")]
