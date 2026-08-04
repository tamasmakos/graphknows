"""Keep a relation's subject in the turn that actually asserted it.

THE RESIDUAL BUG
----------------
Per-turn deixis resolution (``deixis.py``) fixes pronouns, so "I also lost my job
at Door Dash" in Gina's turn becomes "Gina also lost Gina's job at Door Dash" and
the correct edge appears. But it does not stop the extractor CROSSING SUBJECTS
when two people assert parallel facts in one window::

    Jon:  Lost Jon's job as a banker yesterday.
    Gina: Gina also lost Gina's job at Door Dash.

yields BOTH ``Gina -[formerly worked for]-> Door Dash`` (right) and
``Jon -[formerly worked for]-> Door Dash`` (wrong). Both people lost a job, both
names are in the window, and relex has no notion of who spoke which line.

THE GUARD
---------
The window is built from turns whose boundaries we know, so a relation can be
located: find the turn containing the TAIL, and ask which people are named in it.
If that turn names people at all, only those people may head a relation into it.
A tail whose turn names nobody is left alone — that is the cross-turn case the
window exists to capture, and this guard must not eat it.

This is deliberately narrow. It does not try to decide who is right when several
people appear in one turn; it only refuses subjects the asserting turn never
mentions.
"""

from __future__ import annotations


def turn_spans(window: list[dict], rendered: str) -> list[tuple[int, int, str]]:
    """``(start, end, speaker)`` character ranges of each turn in the rendered text."""
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for turn in window:
        speaker = str(turn.get("speaker") or "")
        line_len = len(speaker) + 2 + len(str(turn.get("resolved") or turn.get("text") or ""))
        spans.append((cursor, cursor + line_len, speaker))
        cursor += line_len + 1  # + the newline the join adds
    return spans


def owning_turn(text: str, mention: str, spans: list[tuple[int, int, str]]) -> int:
    """Index of the turn containing ``mention``, or -1 when it cannot be located."""
    at = text.find(mention)
    if at < 0:
        return -1
    for i, (start, end, _) in enumerate(spans):
        if start <= at < end:
            return i
    return -1


def admits(
    head: str,
    tail: str,
    text: str,
    spans: list[tuple[int, int, str]],
    speakers: set[str],
) -> bool:
    """May ``head`` assert a relation into ``tail``?

    Refused only when the tail's own turn names people and ``head`` is not among
    them. Everything else — an unlocatable span, a turn naming nobody, a
    non-person head — is admitted, because this guard exists to remove a specific
    confusion and not to prune the graph.
    """
    if head not in speakers:  # only person-vs-person confusion is in scope
        return True
    turn = owning_turn(text, tail, spans)
    if turn < 0:
        return True
    start, end, _speaker = spans[turn]
    line = text[start:end]
    named = {s for s in speakers if s in line}
    if len(named) < 2:
        # Nobody, or only one candidate — nothing to disambiguate, and the
        # cross-turn case the window exists for must pass through untouched.
        return not named or head in named

    # Several people are named in the asserting turn, which is exactly the
    # confusion this exists for — and it is COMMON once deixis resolution has
    # rewritten the addressee's pronoun into the line ("Sorry about Jon's job
    # Jon, but Gina also lost Gina's job at Door Dash" names both). Presence is
    # therefore not evidence; PROXIMITY is. The subject of a claim sits nearer
    # its object than an unrelated person mentioned earlier in the same breath.
    at = line.find(tail)
    if at < 0:
        return True
    nearest = min(named, key=lambda person: _distance(line, person, at))
    return head == nearest


def _distance(line: str, person: str, target: int) -> int:
    """Characters between the closest mention of ``person`` and ``target``."""
    best = len(line)
    start = 0
    while (at := line.find(person, start)) >= 0:
        best = min(best, abs(at - target))
        start = at + 1
    return best
