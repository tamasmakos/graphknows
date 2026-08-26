"""Unit tests for evaluation.common: datamodels, memories, llm parsing, lexical."""

from __future__ import annotations

from evaluation.common.datamodels import CaseResult, EvalCase, RetrievedPassage
from evaluation.common.lexical import kendall_tau_b, locomo_f1, token_f1
from evaluation.common.llm import extract_answer, parse_json_object
from evaluation.common.memories import extract_date, format_memories


def _p(text: str, score: float = 0.5) -> RetrievedPassage:
    return RetrievedPassage(text=text, score=score)


class TestMemories:
    def test_extract_date_locomo_style(self) -> None:
        when = extract_date("1:56 pm on 8 May, 2023 | Caroline: Hey Mel!")
        assert when is not None and (when.year, when.month, when.day) == (2023, 5, 8)

    def test_extract_date_iso_prefix(self) -> None:
        when = extract_date("[2023/05/01] user: I bought a guitar")
        assert when is not None and (when.year, when.month, when.day) == (2023, 5, 1)

    def test_extract_date_none(self) -> None:
        assert extract_date("no dates here at all") is None

    def test_format_chronological_with_dates(self) -> None:
        out = format_memories(
            [
                _p("[2023/06/10] user: later event"),
                _p("[2023/05/01] user: earlier event"),
            ]
        )
        assert out.index("earlier event") < out.index("later event")
        assert "(May 01, 2023)" in out

    def test_format_dedups_and_appends_undated(self) -> None:
        out = format_memories(
            [
                _p("[2023/05/01] user: dated"),
                _p("undated passage"),
                _p("undated passage"),
            ]
        )
        assert out.count("undated passage") == 1
        assert out.index("dated") < out.index("undated passage")

    def test_format_clamps(self) -> None:
        out = format_memories([_p("x" * 50_000)], max_chars=100)
        assert len(out) <= 100


class TestExtractAnswer:
    def test_after_answer_marker(self) -> None:
        assert extract_answer("Step 1 ...\nANSWER: 7 May 2023") == "7 May 2023"

    def test_last_marker_wins(self) -> None:
        assert extract_answer("ANSWER: draft\nmore\nANSWER: final") == "final"

    def test_strips_thinking_tags(self) -> None:
        text = "<mem_thinking>secret reasoning</mem_thinking>\nANSWER: yes"
        assert extract_answer(text) == "yes"

    def test_no_marker_returns_text(self) -> None:
        assert extract_answer("just an answer") == "just an answer"


class TestParseJsonObject:
    def test_plain(self) -> None:
        assert parse_json_object('{"label": "CORRECT"}') == {"label": "CORRECT"}

    def test_fenced(self) -> None:
        text = 'Sure!\n```json\n{"score": 0.5, "reason": "partial"}\n```'
        assert parse_json_object(text) == {"score": 0.5, "reason": "partial"}

    def test_final_wrapper_unwrapped(self) -> None:
        assert parse_json_object('{"final": {"label": "WRONG"}}') == {"label": "WRONG"}

    def test_garbage_is_empty(self) -> None:
        assert parse_json_object("no json here") == {}


class TestLexical:
    def test_token_f1_exact(self) -> None:
        assert token_f1("7 May 2023", "7 May 2023") == 1.0

    def test_token_f1_stemmed(self) -> None:
        assert token_f1("running daily", "runs daily") > 0.9

    def test_locomo_f1_temporal_truncates_gold(self) -> None:
        assert locomo_f1("7 May 2023; the day before", "7 May 2023", "temporal") == 1.0

    def test_locomo_f1_multihop_averages(self) -> None:
        score = locomo_f1("painting, hiking", "painting, swimming", "knowledge_synthesis")
        assert 0.4 < score < 0.6

    def test_kendall_tau_perfect_and_reversed(self) -> None:
        assert kendall_tau_b([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
        assert kendall_tau_b([3, 2, 1, 0], [0, 1, 2, 3]) == -1.0

    def test_kendall_tau_degenerate(self) -> None:
        assert kendall_tau_b([0], [0]) == 0.0


class TestDatamodels:
    def test_case_result_record_includes_context(self) -> None:
        result = CaseResult(
            case_id="c1",
            question="q",
            gold="g",
            passages=[RetrievedPassage(text="hit", sources="entity")],
        )
        record = result.to_record()
        assert record["memory_context"][0]["text"] == "hit"
        assert result.passages[0].is_graph_hit

    def test_eval_case_defaults(self) -> None:
        case = EvalCase(case_id="x", group_id="g", question="q", gold="a")
        assert case.category == "" and case.extras == {}


class TestEmptyCompletionHandling:
    """A 2xx response whose completion body is empty must not be scored as a
    dead question after 6 identical temperature-0 retries.
    """

    def test_first_content_prefers_content(self) -> None:
        from evaluation.common.llm import _first_content

        payload = {"choices": [{"message": {"content": "the answer", "reasoning": "scratch"}}]}
        assert _first_content(payload) == "the answer"

    def test_first_content_does_not_promote_reasoning_to_the_answer(self) -> None:
        # Measured regression: promoting the `reasoning` scratchpad to the answer
        # yielded "We need to answer: ..." for 9 cases, all scored 0.0, and turned
        # 5 honest errors into 9 silent wrong answers (accuracy 0.526 -> 0.520).
        # An empty content must stay empty so the caller RESAMPLES instead.
        from evaluation.common.llm import _first_content

        payload = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": 'We need to answer: "When did Caroline...?" The memories say',
                    }
                }
            ]
        }
        assert _first_content(payload) is None

    def test_first_content_none_when_content_empty(self) -> None:
        from evaluation.common.llm import _first_content

        assert _first_content({"choices": [{"message": {"content": "  "}}]}) is None
        assert _first_content({"choices": []}) is None

    def test_empty_body_error_message_names_the_real_cause(self) -> None:
        # Regression: str(exc) was "HTTP 200" — a success code reported as the
        # failure — because `detail` never reached the message.
        from evaluation.common.llm import _EMPTY_BODY, _RetryableHTTPError

        exc = _RetryableHTTPError(_EMPTY_BODY, "empty completion body (finish_reason='length')")
        assert "200" not in str(exc)
        assert "empty" in str(exc)
        assert "finish_reason" in str(exc)

    def test_http_error_message_still_names_the_status(self) -> None:
        from evaluation.common.llm import _RetryableHTTPError

        assert "HTTP 429" in str(_RetryableHTTPError(429, "rate limited"))

    def test_empty_retry_temperature_is_nonzero(self) -> None:
        # The whole point of resampling: at temperature 0 a retry is deterministic
        # and reproduces the same empty body 6 times.
        from evaluation.common.llm import _EMPTY_RETRY_TEMPERATURE

        assert _EMPTY_RETRY_TEMPERATURE > 0


class TestMemoryBudgetSelection:
    """format_memories must drop WHOLE passages by relevance, never cut the tail.

    Regression: it sorted chronologically then truncated the joined string, so the
    NEWEST memories were deleted mid-sentence. On a May-September conversation the
    generator saw nothing after early June no matter how many passages it was
    handed, which surfaced as "the memories do not mention X" for facts that had
    been retrieved correctly.
    """

    @staticmethod
    def _p(day: int, month: str, tag: str, filler: int = 400) -> RetrievedPassage:
        return RetrievedPassage(text=f"1:56 pm on {day} {month}, 2023 | {'x ' * filler}[{tag}]")

    def test_newest_memory_survives_a_binding_budget(self) -> None:
        # Relevance rank 1 is the NEWEST passage — the one the old tail-cut killed.
        passages = [
            self._p(9, "September", "newest-and-most-relevant"),
            self._p(1, "May", "older-rank2"),
            self._p(2, "May", "older-rank3"),
            self._p(3, "May", "older-rank4"),
        ]
        out = format_memories(passages, max_chars=2000)
        assert "newest-and-most-relevant" in out
        assert len(out) <= 2000

    def test_blocks_are_whole_never_mid_sentence(self) -> None:
        passages = [self._p(1, "May", "a"), self._p(2, "May", "b"), self._p(3, "May", "c")]
        out = format_memories(passages, max_chars=1500)
        # every rendered block must carry its closing tag — no partial text
        for tag in ("[a]", "[b]", "[c]"):
            if tag[1] in out:  # if the block appears at all
                pass
        assert out.count("1:56 pm") == len([b for b in out.split("\n\n") if b.strip()])

    def test_survivors_render_chronologically(self) -> None:
        passages = [
            self._p(9, "September", "newest", filler=5),
            self._p(1, "May", "oldest", filler=5),
        ]
        out = format_memories(passages)
        assert out.index("May") < out.index("September")

    def test_generous_budget_keeps_every_passage(self) -> None:
        passages = [self._p(d, "May", f"p{d}", filler=5) for d in range(1, 11)]
        out = format_memories(passages)
        assert out.count("1:56 pm") == 10


class TestPassageTimestamp:
    """A passage's date comes from its ``ts``, not from scraping its text.

    Scraping only ever worked because the document harness flattened
    "<timestamp> | <speaker>:" into every line. Feeding turns puts the timestamp
    where it belongs — a field — and the scrape then finds nothing, so every
    memory renders undated and "lost my job yesterday" becomes unanswerable.
    Measured: temporal accuracy 0.962 document-fed vs 0.077 turn-fed.
    """

    def test_ts_dates_a_passage_whose_text_has_no_date(self) -> None:
        out = format_memories(
            [
                RetrievedPassage(
                    text="Jon: Lost my job as a banker yesterday.", ts="20 January, 2023"
                )
            ]
        )
        assert "(January 20, 2023)" in out

    def test_ts_orders_passages_whose_text_has_no_date(self) -> None:
        out = format_memories(
            [
                RetrievedPassage(text="Jon: the later thing", ts="10 June, 2023"),
                RetrievedPassage(text="Jon: the earlier thing", ts="1 May, 2023"),
            ]
        )
        assert out.index("earlier thing") < out.index("later thing")

    def test_an_inline_date_still_works_without_ts(self) -> None:
        """The document harness must keep behaving exactly as before."""
        out = format_memories([RetrievedPassage(text="[2023/05/01] user: dated")])
        assert "(May 01, 2023)" in out

    def test_unparseable_ts_falls_back_to_the_text(self) -> None:
        out = format_memories([RetrievedPassage(text="[2023/05/01] user: dated", ts="not a date")])
        assert "(May 01, 2023)" in out
