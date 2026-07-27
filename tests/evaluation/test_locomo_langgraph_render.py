"""The LoCoMo agent's memory renderer — both arms depend on it.

The clean arm dates a memory from ``hit["ts"]``; the ``--prefix`` arm has no
``ts`` and carries its date inline in the text. If the clean arm ever silently
took the fallback branch it would render undated against a prompt that promises
dates, and the arm would score badly for a reason nothing in the output names.
"""

from __future__ import annotations

from evaluation.scripts.locomo_langgraph import parse_turn, render, render_hit


def test_clean_hit_renders_a_month_day_year_prefix_and_the_speaker() -> None:
    """ANSWER_PROMPT Stage 5 keys its date arithmetic on a "(Month DD, YYYY)"
    prefix and forbids carrying a clock time into an answer, so the raw
    "4:04 pm on 20 January, 2023" must reduce to the day."""
    block, when = render_hit(
        {"text": "Mochi's vet is Dr. Nagy.", "speaker": "Gina", "ts": "4:04 pm on 20 January, 2023"}
    )
    assert block == "(January 20, 2023) Gina: Mochi's vet is Dr. Nagy."
    assert when is not None and (when.year, when.month, when.day) == (2023, 1, 20)


def test_prefixed_hit_without_ts_is_passed_through_untouched() -> None:
    """The --prefix arm's text already carries its own date and speaker; adding
    a second prefix would double the date, and dropping it would leave the arm
    undated. Its date is still recovered, so it sorts chronologically."""
    text = "4:04 pm on 20 January, 2023 | Gina: Mochi's vet is Dr. Nagy."
    block, when = render_hit({"text": text, "speaker": "", "ts": ""})
    assert block == text
    assert when is not None and (when.year, when.month, when.day) == (2023, 1, 20)


def test_render_sorts_chronologically_not_by_rank_and_counts_dated() -> None:
    """Memories arrive in relevance order and must read oldest-first: Stage 5
    walks the timeline. The counts are what make a silent fallback visible."""
    body, dated, undated = render(
        [
            {"text": "later", "speaker": "Jon", "ts": "1 March, 2023"},
            {"text": "earlier", "speaker": "Gina", "ts": "1 January, 2023"},
        ]
    )
    assert body.index("earlier") < body.index("later")
    assert (dated, undated) == (2, 0)


def test_render_drops_exact_duplicates() -> None:
    hit = {"text": "same", "speaker": "Gina", "ts": "1 January, 2023"}
    body, dated, _ = render([hit, dict(hit)])
    assert body.count("same") == 1
    assert dated == 1


def test_undated_memory_is_counted_and_kept_after_the_dated_ones() -> None:
    body, dated, undated = render(
        [
            {"text": "no date here at all", "speaker": "", "ts": ""},
            {"text": "dated one", "speaker": "Gina", "ts": "1 January, 2023"},
        ]
    )
    assert (dated, undated) == (1, 1)
    assert body.index("dated one") < body.index("no date here at all")


def test_prefix_flag_reproduces_the_baseline_exactly() -> None:
    """Full replication: the text is flattened AND the metadata fields are left
    empty, so CHUNK.speaker falls back to the role and CHUNK.ts stays "". A
    prefix that also set the fields would be a third design, testing nothing."""
    msg = {"role": "Gina", "timestamp": "4:04 pm on 20 January, 2023", "content": "Hey Jon!"}

    clean = parse_turn(msg)
    assert clean == {
        "speaker": "Gina",
        "timestamp": "4:04 pm on 20 January, 2023",
        "text": "Hey Jon!",
    }

    prefixed = parse_turn(msg, prefix=True)
    assert prefixed == {
        "speaker": "",
        "timestamp": "",
        "text": "4:04 pm on 20 January, 2023 | Gina: Hey Jon!",
    }
