"""Operational metrics: latency / cost / behaviour, as a README would quote them."""

from __future__ import annotations

from evaluation.common.datamodels import CaseResult
from evaluation.common.reporting import summarize


def _case(
    unit: str, *, retrieve: float, generate: float, cost: float, abstained: bool
) -> CaseResult:
    return CaseResult(
        case_id="c",
        question="q",
        gold="g",
        category="temporal",
        group_id=unit,
        passages=[],
        generated_answer="a",
        score=1.0,
        metrics={
            "retrieve_ms": retrieve,
            "gen_cost_usd": cost,
            "gen_prompt_tokens": 19000,
            "gen_completion_tokens": 300,
            "abstained": abstained,
            "fact_count": 12,
            "context_passage_count": 50,
            # ingest_* is copied onto EVERY case of a unit by build_case_result
            "ingest_cost_usd": 0.0,
            "ingest_elapsed_s": 750.0,
            "ingested_chunks": 63,
            "flush_nodes": 998,
        },
        metadata={},
    )


def _latency(cases: list[CaseResult]) -> dict:
    return summarize(cases)["operational"]["latency_ms"]


class TestPercentile:
    def test_nearest_rank(self) -> None:
        cases = [
            _case("u", retrieve=float(i), generate=1.0, cost=0.0, abstained=False)
            for i in range(1, 101)
        ]
        lat = _latency(cases)["retrieve_ms"]
        assert lat["p50"] == 50.0
        assert lat["p95"] == 95.0
        assert lat["max"] == 100.0

    def test_single_sample(self) -> None:
        cases = [_case("u", retrieve=7.0, generate=1.0, cost=0.0, abstained=False)]
        lat = _latency(cases)["retrieve_ms"]
        assert lat["p50"] == lat["p95"] == lat["max"] == 7.0

    def test_no_samples(self) -> None:
        """No cases means no timings — report nothing rather than a made-up 0.0."""
        assert _latency([]) == {}


class TestOperational:
    def test_only_retrieval_is_timed(self) -> None:
        """Retrieval is the only stage this system owns. Generation/judge time is
        the configured provider's — reporting it under our name would change with
        every model swap and say nothing about the memory layer."""
        cases = [_case("u", retrieve=10.0, generate=2000.0, cost=0.004, abstained=False)]
        lat = summarize(cases)["operational"]["latency_ms"]
        assert set(lat) == {"retrieve_ms"}, f"model-dependent timing leaked in: {set(lat)}"
        assert lat["retrieve_ms"]["p50"] == 10.0

    def test_model_cost_is_still_reported(self) -> None:
        """Cost stays: it is what a user actually spends, model or not."""
        cases = [_case("u", retrieve=10.0, generate=2000.0, cost=0.004, abstained=False)]
        assert summarize(cases)["operational"]["cost_usd"]["query_total"] == 0.004

    def test_latency_reports_percentiles_not_just_mean(self) -> None:
        cases = [
            _case("u", retrieve=float(r), generate=100.0, cost=0.0, abstained=False)
            for r in (10, 10, 10, 10, 900)
        ]
        lat = summarize(cases)["operational"]["latency_ms"]["retrieve_ms"]
        assert lat["p50"] == 10.0, "median must not be dragged by the tail"
        assert lat["p95"] == 900.0, "p95 must expose the tail a mean would hide"
        assert lat["mean"] > lat["p50"]

    def test_ingest_counted_per_unit_not_per_question(self) -> None:
        """ingest_* is repeated onto every case; summing it would multiply by the
        question count and report a wildly inflated cost/duration."""
        cases = [
            _case("conv-26", retrieve=1.0, generate=1.0, cost=0.0, abstained=False)
            for _ in range(20)
        ]
        cases += [
            _case("conv-30", retrieve=1.0, generate=1.0, cost=0.0, abstained=False)
            for _ in range(20)
        ]
        ing = summarize(cases)["operational"]["ingest"]
        assert ing["units"] == 2
        assert ing["elapsed_s_total"] == 1500.0, "2 x 750s, not 40 x 750s"
        assert ing["chunks_total"] == 126, "2 x 63, not 40 x 63"

    def test_llm_free_ingest_costs_nothing(self) -> None:
        """The load-bearing claim: the graph is built with no LLM call."""
        cases = [_case("u", retrieve=1.0, generate=1.0, cost=0.004, abstained=False)]
        cost = summarize(cases)["operational"]["cost_usd"]
        assert cost["ingest_total"] == 0.0
        assert cost["query_total"] == 0.004
        assert cost["total"] == 0.004

    def test_query_cost_is_summed_and_averaged(self) -> None:
        cases = [
            _case("u", retrieve=1.0, generate=1.0, cost=0.01, abstained=False) for _ in range(10)
        ]
        cost = summarize(cases)["operational"]["cost_usd"]
        assert abs(cost["query_total"] - 0.1) < 1e-9
        assert abs(cost["query_per_question"] - 0.01) < 1e-9

    def test_abstention_is_its_own_rate(self) -> None:
        """Refusing to answer is a different defect from answering wrongly, and
        accuracy alone cannot tell them apart."""
        cases = [
            _case("u", retrieve=1.0, generate=1.0, cost=0.0, abstained=i < 3) for i in range(10)
        ]
        assert summarize(cases)["operational"]["behaviour"]["abstention_rate"] == 0.3
