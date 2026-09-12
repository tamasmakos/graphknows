"""Small spaCy/WordNet helpers shared by the extraction passes.

This file is what is left of ``relations/svo.py`` after the SVO relation miner
was deleted. That miner produced the relation label space relex extracted
against, and measured on a live conv-30 graph it produced 28 of the graph's 43
relation edges — 25 of which were ``<speaker> HAS <common noun>`` (``Jon HAS
back``, ``Jon HAS corner``, ``Jon HAS side``) over a predicate vocabulary of
``{HAS: 25, BE_TO: 1, IS_A: 1, LOSE_AT: 1}``. The label space now comes from the
ontology properties nearest the chunk embedding; see
``entities/extractor.py::_relex``.

None of the three helpers below is relation mining, and each still has a caller:

* :func:`_resolve`        — :func:`resolve_deixis` below speaker-resolves surfaces
* :func:`noun_supersense` — ``entities/extractor.py::_arg_type`` types arguments
* :func:`parse_speaker`   — ``entities/extractor.py::_line_speakers`` reads turns

:func:`resolve_deixis` is new and builds on ``_resolve``: the relation pass reads
a surface in which first-person pronouns have become the speaker's name, because
a pronoun is not an entity span and so the subject of every self-stated fact was
missing from the graph.
"""

from __future__ import annotations

import bisect
import logging
import re
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

_FIRST_PERSON = {"i", "me", "my", "myself", "we", "us", "our"}
_SECOND_PERSON = {"you", "your", "yourself"}

# Genitive pronouns lose their marker when swapped for a name: "my collection"
# -> "Gina collection". The apostrophe-s is what keeps the phrase a phrase.
_POSSESSIVE = {"my", "mine", "our", "ours", "your", "yours"}

# A clitic is its own token, so replacing the pronoun in front of it leaves
# "Gina'm" / "Gina've" — not English, and not something an extractor can parse.
# Expanded to the 3rd-person singular form, which is what a proper name takes.
_CLITICS = {"'m": "is", "'ve": "has", "'re": "is", "'ll": "will", "'d": "would"}

_SPEAKER_LINE = re.compile(r"^(?:.*?\|\s*)?([A-Za-z][\w' ]*?):\s*(.+)$")


def _resolve(tok: Any, speaker: str, speakers: list[str]) -> str:
    """Resolve a 1st/2nd-person pronoun to the speaker (or the other of two)."""
    low = tok.text.lower()
    if low in _FIRST_PERSON:
        return speaker
    if low in _SECOND_PERSON and len(speakers) == 2 and speaker:
        return next((s for s in speakers if s != speaker), tok.text)
    return tok.text


def resolve_deixis(doc: Any, speaker: str = "") -> str:
    """``doc``'s text with 1st/2nd-person pronouns rewritten to the speaker's name.

    The relation extractor needs two entity SPANS to emit a relation, and
    ``I`` / ``my`` / ``I've`` are not spans — so in dialogue, where speakers state
    facts about themselves in the first person ("I opened an online clothing
    store", "My collection has twenty pieces"), the SUBJECT of nearly every
    stated fact is invisible and the relation has only one end. Measured on 40
    real conv-30 turns, same model, same ontology labels, same threshold: 3
    relations from the raw text, 19 from this surface.

    Rebuilt TOKEN-WISE and joined on each token's own trailing whitespace so the
    result stays well-formed English. A regex over the raw text mangles
    contractions — it turned "I'm over the moon" into "gina is over the moon be".

    Speaker per LINE (``"ts | Speaker: text"``), falling back to ``speaker`` for
    a line with no turn prefix, so a multi-turn chunk resolves each turn to its
    own speaker. Second person only resolves when the chunk has exactly two
    speakers (see :func:`_resolve`); a single-turn chunk therefore leaves "you"
    alone, and the caller's pronoun gate drops any relation anchored on it.

    Returns "" when no speaker is known anywhere — the caller's signal to fall
    back to the raw text.

    ponytail: main-verb agreement is left crude ("Gina work at the store"); only
    the clitic, which is not a word at all after the swap, is fixed. Inflecting
    the verb needs a morphologiser and the 6.3x was measured without one.
    """
    lines = doc.text.split("\n")
    starts: list[int] = []
    line_speakers: list[str] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        line_speakers.append(parse_speaker(line) or speaker)
        pos += len(line) + 1  # + the '\n' consumed by split
    known = list(dict.fromkeys(s for s in line_speakers if s))
    if not known:
        return ""

    out: list[str] = []
    swapped = False
    for tok in doc:
        low = tok.text.lower()
        if swapped and low in _CLITICS:
            out.append(" " + _CLITICS[low] + tok.whitespace_)
            swapped = False
            continue
        current = line_speakers[bisect.bisect_right(starts, tok.idx) - 1]
        name = _resolve(tok, current, known) if current else tok.text
        swapped = name != tok.text
        if swapped and low in _POSSESSIVE:
            name += "'s"
        out.append(name + tok.whitespace_)
    return "".join(out)


@lru_cache(maxsize=8192)
def noun_supersense(phrase: str) -> str:
    """WordNet supersense of a noun phrase's head ("noun.artifact"), or "" if OOV.

    The lexicographer file of the head noun's first sense. spaCy NER types only
    NAMED entities, but dialogue relation arguments are overwhelmingly common
    nouns ("a nurse", "cat", "job"), so without this they reach the graph as
    untyped ENTITY vertices. Head = rightmost token.
    """
    head = phrase.strip().split()[-1].lower() if phrase.strip() else ""
    if not head:
        return ""
    try:
        from nltk.corpus import wordnet as wn
        from nltk.stem import WordNetLemmatizer

        ss = wn.synsets(WordNetLemmatizer().lemmatize(head, "n"), pos=wn.NOUN)
    except LookupError:
        log.warning("WordNet corpus unavailable; typing %r as untyped", phrase)
        return ""
    return ss[0].lexname() if ss else ""


def parse_speaker(line: str) -> str:
    """Speaker of a LoCoMo 'ts | Speaker: text' line, or '' when there is none."""
    m = _SPEAKER_LINE.match(line.strip())
    return m.group(1).strip() if m else ""
