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
            passages=[RetrievedPassage(text="hit", sources="ltm_graph")],
        )
        record = result.to_record()
        assert record["memory_context"][0]["text"] == "hit"
        assert result.passages[0].is_ltm_graph

    def test_eval_case_defaults(self) -> None:
        case = EvalCase(case_id="x", group_id="g", question="q", gold="a")
        assert case.category == "" and case.extras == {}
