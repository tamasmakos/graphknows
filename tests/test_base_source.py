"""Tests for BaseSource + ParsedDocument ingestion contract."""
from __future__ import annotations

import asyncio
import hashlib
import pytest

from services.search.parsed_document import ParsedDocument, markdown_to_chunks
from services.search.base_source import BaseSource


# ------------------------------------------------------------------ #
# ParsedDocument tests
# ------------------------------------------------------------------ #

def make_doc(**kwargs) -> ParsedDocument:
    defaults = dict(
        source_id="test_source",
        source_type="TEST",
        url="https://example.com/doc/1",
        title="Test Document",
        date="2024-01-15",
        institution="Test Institution",
        language="en",
        raw_text="Some raw text.",
        markdown="# Test Document\n\n## Section One\n\nContent here.\n\n## Section Two\n\nMore content.",
    )
    defaults.update(kwargs)
    return ParsedDocument(**defaults)


def test_parsed_document_doc_id_auto_generated():
    doc = make_doc()
    expected = hashlib.sha1(b"test_source:https://example.com/doc/1").hexdigest()[:16]
    assert doc.doc_id == expected


def test_parsed_document_explicit_doc_id():
    doc = make_doc(doc_id="custom_id_123")
    assert doc.doc_id == "custom_id_123"


def test_parsed_document_doc_id_is_stable():
    doc1 = make_doc()
    doc2 = make_doc()
    assert doc1.doc_id == doc2.doc_id


def test_parsed_document_different_urls_different_ids():
    doc1 = make_doc(url="https://example.com/doc/1")
    doc2 = make_doc(url="https://example.com/doc/2")
    assert doc1.doc_id != doc2.doc_id


# ------------------------------------------------------------------ #
# markdown_to_chunks tests
# ------------------------------------------------------------------ #

def test_markdown_to_chunks_splits_on_h2():
    doc = make_doc(
        markdown="# Title\n\n## Alpha\n\nAlpha content.\n\n## Beta\n\nBeta content."
    )
    chunks = markdown_to_chunks(doc)
    assert len(chunks) == 3  # intro + Alpha + Beta
    headings = [c["heading"] for c in chunks]
    assert "Alpha" in headings
    assert "Beta" in headings


def test_markdown_to_chunks_fallback_when_no_h2():
    doc = make_doc(markdown="# Just a title\n\nNo sections here.")
    chunks = markdown_to_chunks(doc)
    assert len(chunks) == 1
    assert chunks[0]["text"]


def test_markdown_to_chunks_fields_present():
    doc = make_doc()
    chunks = markdown_to_chunks(doc)
    required = {"chunk_id", "doc_id", "source_id", "source_type", "url", "date",
                 "institution", "language", "heading", "text"}
    for chunk in chunks:
        assert required.issubset(chunk.keys()), f"Missing keys in chunk: {chunk.keys()}"


def test_markdown_to_chunks_respects_max_tokens():
    long_text = "word " * 10000
    doc = make_doc(markdown=f"# T\n\n## S\n\n{long_text}")
    chunks = markdown_to_chunks(doc, max_tokens=512)
    # text should be truncated to ~512*4 chars
    for c in chunks:
        assert len(c["text"]) <= 512 * 4 + 100  # some slack for heading


def test_markdown_to_chunks_chunk_id_deterministic():
    doc = make_doc()
    c1 = markdown_to_chunks(doc)
    c2 = markdown_to_chunks(doc)
    assert [x["chunk_id"] for x in c1] == [x["chunk_id"] for x in c2]


# ------------------------------------------------------------------ #
# BaseSource helper tests (via concrete subclass)
# ------------------------------------------------------------------ #

class _ConcreteSource(BaseSource):
    source_id = "test"
    source_type = "TEST"
    institution = "Test Inst"

    async def scrape(self, limit=10):
        return []

    async def fetch_one(self, url):
        return make_doc(url=url)


def test_make_doc_id_deterministic():
    src = _ConcreteSource()
    id1 = src._make_doc_id("ep_resolutions", "https://example.com/1")
    id2 = src._make_doc_id("ep_resolutions", "https://example.com/1")
    assert id1 == id2
    assert len(id1) == 16


def test_make_doc_id_different_parts_differ():
    src = _ConcreteSource()
    id1 = src._make_doc_id("ep_resolutions", "https://example.com/1")
    id2 = src._make_doc_id("ep_resolutions", "https://example.com/2")
    assert id1 != id2


def test_html_to_markdown_strips_tags():
    src = _ConcreteSource()
    html = "<h2>Hello</h2><p>World</p>"
    md = src._html_to_markdown(html)
    assert "Hello" in md
    assert "World" in md
    assert "<h2>" not in md
    assert "<p>" not in md


def test_html_to_markdown_no_excessive_newlines():
    src = _ConcreteSource()
    html = "<p>A</p>\n\n\n\n\n<p>B</p>"
    md = src._html_to_markdown(html)
    assert "\n\n\n" not in md


def test_detect_language_fallback():
    src = _ConcreteSource()
    # Should never raise, even with garbage
    lang = src._detect_language("xzq " * 5)
    assert isinstance(lang, str)
    assert len(lang) >= 2


def test_detect_language_english():
    src = _ConcreteSource()
    lang = src._detect_language("The European Parliament adopted a resolution on climate change policy.")
    assert lang == "en"


@pytest.mark.asyncio
async def test_base_source_context_manager():
    src = _ConcreteSource()
    async with src as s:
        assert s._client is not None
    assert src._client is None
