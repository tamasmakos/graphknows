"""Resolve first- and second-person reference PER TURN in a window.

THE BUG THIS FIXES
------------------
A 5-turn window renders as::

    Gina: Hey Jon! Good to see you.
    Jon: Lost my job as a banker yesterday.
    Gina: Sorry about your job Jon, but I also lost my job at Door Dash.

and the extractor returns ``Jon -[works for]-> Door Dash``. It was GINA who lost
the Door Dash job. The extractor has no way to know that: "I" is unresolved, and
"Jon" is the nearest person name, so the relation lands on the wrong human. Every
extra turn in the window makes this more likely, which means the window that
buys cross-turn relations also buys cross-turn ERRORS.

The repo already has the single-speaker version of this idea — the extractor
builds a ``deixis_surface`` rewriting first/second person to the speaker's name
for the relation pass. It cannot help here because a window has many speakers and
one surface. So the rewrite has to happen PER LINE, before the lines are joined.

WHAT IT REWRITES
----------------
* first person ("I", "my", "me", "we", "our") -> the line's own speaker;
* second person ("you", "your") -> the ADDRESSEE, taken as the other participant
  in the window. Conversations here are two-party, so the addressee is
  unambiguous; with three or more speakers the rewrite is skipped rather than
  guessed, because a wrong name is worse than a pronoun.

Possessives become "Gina's", plain pronouns become "Gina". Case is preserved at
sentence start. Nothing else is touched: the text the extractor sees stays the
speaker's own words apart from the pronouns, and CHUNK.text is unaffected because
this surface is built for extraction only.
"""

from __future__ import annotations

import re

_FIRST_POSSESSIVE = re.compile(r"\b(my|our)\b", re.IGNORECASE)
_FIRST_PLAIN = re.compile(r"\b(i|me|we|us)\b", re.IGNORECASE)
_SECOND_POSSESSIVE = re.compile(r"\b(your)\b", re.IGNORECASE)
_SECOND_PLAIN = re.compile(r"\b(you)\b", re.IGNORECASE)

# Contractions must be expanded BEFORE the pronoun rules run. An apostrophe is a
# word boundary, so ``\bI\b`` matches the "I" inside "I'm" and rewrites it to
# "Jon'm" — which is not a typo but a second PERSON node, splitting the speaker
# in two and taking half their relations with it. Measured on a smoke ingest:
# every one of Jon's seven relations hung off "Jon'm" instead of "Jon".
_CONTRACTIONS = [
    (re.compile(r"\bI'm\b", re.IGNORECASE), "{s} is"),
    (re.compile(r"\bI've\b", re.IGNORECASE), "{s} has"),
    (re.compile(r"\bI'll\b", re.IGNORECASE), "{s} will"),
    (re.compile(r"\bI'd\b", re.IGNORECASE), "{s} would"),
    (re.compile(r"\bwe're\b", re.IGNORECASE), "{s} is"),
    (re.compile(r"\bwe've\b", re.IGNORECASE), "{s} has"),
    (re.compile(r"\bwe'll\b", re.IGNORECASE), "{s} will"),
    (re.compile(r"\bwe'd\b", re.IGNORECASE), "{s} would"),
]
_SECOND_CONTRACTIONS = [
    (re.compile(r"\byou're\b", re.IGNORECASE), "{s} is"),
    (re.compile(r"\byou've\b", re.IGNORECASE), "{s} has"),
    (re.compile(r"\byou'll\b", re.IGNORECASE), "{s} will"),
    (re.compile(r"\byou'd\b", re.IGNORECASE), "{s} would"),
]


def resolve_line(text: str, speaker: str, addressee: str) -> str:
    """One turn with its pronouns replaced by names."""
    out = text
    if speaker:
        for pattern, template in _CONTRACTIONS:
            out = pattern.sub(template.format(s=speaker), out)
        out = _FIRST_POSSESSIVE.sub(f"{speaker}'s", out)
        out = _FIRST_PLAIN.sub(speaker, out)
    if addressee:
        for pattern, template in _SECOND_CONTRACTIONS:
            out = pattern.sub(template.format(s=addressee), out)
        out = _SECOND_POSSESSIVE.sub(f"{addressee}'s", out)
        out = _SECOND_PLAIN.sub(addressee, out)
    return out


def resolve_window(window: list[dict]) -> str:
    """The window as extraction-ready text, every turn's deixis resolved.

    Speakers are still prefixed, because attribution of the UTTERANCE is
    separate from resolution of its pronouns and the prefix is what lets a
    reader (and the entity-name gate) tell the two apart.
    """
    speakers = list(dict.fromkeys(t.get("speaker", "") for t in window if t.get("speaker")))
    lines = []
    for turn in window:
        speaker = str(turn.get("speaker") or "")
        # Two-party is resolvable; anything else is a guess, so leave it alone.
        others = [s for s in speakers if s != speaker]
        addressee = others[0] if len(others) == 1 else ""
        lines.append(f"{speaker}: {resolve_line(str(turn.get('text') or ''), speaker, addressee)}")
    return "\n".join(lines)
