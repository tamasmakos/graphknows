"""Tests for local JSON/JSONL memory benchmark adapters."""

from __future__ import annotations

import json
import asyncio

from evaluation.core.file_backed import FileBackedBenchmarkOptions, run_file_backed_benchmark


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        if name == "memory_ingest":
            return {"chunks": 1, "entities": 2}
        if name == "memory_query":
            return {"hits": [{"text": "Alex visited Prague after Berlin."}]}
        return {"errors": []}


def test_file_backed_runner_ingests_sessions_and_scores_answer(tmp_path) -> None:
    data_root = tmp_path / "locomo"
    data_root.mkdir()
    row = {
        "id": "case-1",
        "category": "temporal memory recall",
        "question": "Which city did Alex visit after Berlin?",
        "answer": "Prague",
        "sessions": [
            {
                "session_id": "s1",
                "messages": [
                    {"role": "user", "content": "Alex went to Berlin."},
                    {"role": "assistant", "content": "Then Alex visited Prague."},
                ],
            }
        ],
    }
    (data_root / "questions.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    client = FakeClient()

    cases = asyncio.run(
        run_file_backed_benchmark(
            client,  # type: ignore[arg-type]
            FileBackedBenchmarkOptions(
                benchmark="locomo",
                data_root=data_root,
                questions_file="questions.jsonl",
                run_id="run",
            ),
        )
    )

    assert [name for name, _args in client.calls] == [
        "memory_ingest",
        "memory_query",
        "memory_purge",
    ]
    assert cases[0].case_id == "case-1"
    assert cases[0].metrics["answer_support_hit"] is True
    assert cases[0].score == 1.0
