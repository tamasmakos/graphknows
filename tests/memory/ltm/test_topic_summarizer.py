"""Unit tests for TopicSummarizer.summarize()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from graphknows.memory.ltm.topic_summarizer import TopicSummarizer


@pytest.mark.asyncio
async def test_no_api_key_returns_fallback_title():
    summarizer = TopicSummarizer(api_key="", model="meta-llama/llama-3.1-8b-instruct")
    title, summary = await summarizer.summarize(
        members=["EU", "France"],
        passages="EU and France cooperate.",
        fallback_title="EU, France",
    )
    assert title == "EU, France"
    assert summary == ""


@pytest.mark.asyncio
async def test_none_api_key_returns_fallback_title():
    summarizer = TopicSummarizer(api_key=None, model="meta-llama/llama-3.1-8b-instruct")
    title, summary = await summarizer.summarize(
        members=["Germany"],
        passages="German economy.",
        fallback_title="Germany",
    )
    assert title == "Germany"
    assert summary == ""


@pytest.mark.asyncio
async def test_llm_path_returns_prediction(monkeypatch):
    prediction = MagicMock()
    prediction.title = "European Policy"
    prediction.summary = "EU and France coordinate on policy matters."

    fake_lm = MagicMock()
    fake_module = MagicMock()
    fake_module.return_value = prediction

    with (
        patch("dspy.LM", return_value=fake_lm),
        patch(
            "graphknows.memory.ltm.topic_summarizer.TopicSummaryModule", return_value=fake_module
        ),
    ):
        summarizer = TopicSummarizer(api_key="sk-test", model="meta-llama/llama-3.1-8b-instruct")
        summarizer._lm = fake_lm

        async def _fake_to_thread(fn, *a, **kw):
            return fn()

        monkeypatch.setattr("asyncio.to_thread", _fake_to_thread)
        title, summary = await summarizer.summarize(
            members=["EU", "France"],
            passages="EU and France cooperate.",
            fallback_title="EU, France",
        )

    assert title == "European Policy"
    assert summary == "EU and France coordinate on policy matters."


@pytest.mark.asyncio
async def test_llm_failure_returns_fallback(monkeypatch):
    fake_lm = MagicMock()

    with (
        patch("dspy.LM", return_value=fake_lm),
        patch("graphknows.memory.ltm.topic_summarizer.TopicSummaryModule"),
    ):
        summarizer = TopicSummarizer(api_key="sk-test", model="meta-llama/llama-3.1-8b-instruct")
        summarizer._lm = fake_lm

        async def _raise(*a, **kw):
            raise RuntimeError("API error")

        monkeypatch.setattr("asyncio.to_thread", _raise)
        title, summary = await summarizer.summarize(
            members=["EU"],
            passages="EU policy.",
            fallback_title="EU",
        )

    assert title == "EU"
    assert summary == ""
