"""Unit tests for EU source adapters (no real network calls)."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.search.adapters.council_conclusions import CouncilConclusionsSource
from services.search.adapters.ec_press import ECPressSource
from services.search.adapters.ecb_publications import ECBPublicationsSource
from services.search.adapters.eca_reports import ECAReportsSource
from services.search.adapters.eprs_briefings import EPRSBriefingsSource
from services.search.parsed_document import ParsedDocument, markdown_to_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


SAMPLE_HTML = """
<html><head><title>Test Document 2024-01-15</title></head>
<body>
<time datetime="2024-01-15">15 January 2024</time>
<h2>Introduction</h2><p>This is the introduction text.</p>
<h2>Key Findings</h2><p>These are the key findings of the report.</p>
<h2>Conclusions</h2><p>Final conclusions here.</p>
</body></html>
"""

SAMPLE_PDF_MARKDOWN = "## Page 1\n\nEuropean Council Conclusions\n\nThe Council met on 15 December 2023.\n\n## Page 2\n\nFurther conclusions text."


# ---------------------------------------------------------------------------
# 1. CouncilConclusionsSource
# ---------------------------------------------------------------------------

class TestCouncilConclusionsSource:
    def test_instantiation(self):
        src = CouncilConclusionsSource()
        assert src.source_id == "council_conclusions"
        assert src.source_type == "COUNCIL_CONCLUSION"
        assert src.institution == "European Council"

    def test_scrape_finds_pdf_links(self):
        listing_html = '''
        <html><body>
        <a href="/media/abcdef/conclusions-2024-01-15.pdf">Conclusions Jan 2024</a>
        <a href="/media/xyz/conclusions-2023-12-14.pdf">Conclusions Dec 2023</a>
        <a href="/other/page.html">Not a PDF</a>
        </body></html>
        '''
        src = CouncilConclusionsSource()
        src._get = AsyncMock(return_value=listing_html)

        async def fake_fetch(url):
            return ParsedDocument(
                source_id=src.source_id,
                source_type=src.source_type,
                url=url,
                title="Test Conclusions",
                date="2024-01-15",
                institution=src.institution,
                language="en",
                raw_text="sample text",
                markdown=f"# Test Conclusions\n\n## Page 1\n\nSample content",
            )
        src.fetch_one = fake_fetch

        docs = run(src.scrape(limit=5))
        assert len(docs) == 2
        assert all(isinstance(d, ParsedDocument) for d in docs)

    def test_fetch_one_returns_parsed_document(self):
        src = CouncilConclusionsSource()
        pdf_bytes = b"%PDF-1.4 fake pdf content"

        mock_resp = MagicMock()
        mock_resp.content = pdf_bytes
        mock_resp.raise_for_status = MagicMock()

        src._client = MagicMock()
        src._client.get = AsyncMock(return_value=mock_resp)
        src._pdf_to_markdown = MagicMock(return_value=SAMPLE_PDF_MARKDOWN)

        doc = run(src.fetch_one("https://www.consilium.europa.eu/media/abc/conclusions-2023-12-14.pdf"))
        assert isinstance(doc, ParsedDocument)
        assert doc.source_id == "council_conclusions"
        assert doc.date == "2023-12-14"
        assert "# " in doc.markdown

    def test_chunks_from_council_doc(self):
        doc = ParsedDocument(
            source_id="council_conclusions",
            source_type="COUNCIL_CONCLUSION",
            url="https://example.com/conclusions.pdf",
            title="European Council Conclusions December 2023",
            date="2023-12-14",
            institution="European Council",
            language="en",
            raw_text="Council met. Decisions taken.",
            markdown="# European Council Conclusions\n\n## Page 1\n\nThe Council met.\n\n## Page 2\n\nFurther decisions.",
        )
        chunks = markdown_to_chunks(doc)
        assert len(chunks) >= 1
        assert chunks[0]["source_id"] == "council_conclusions"
        assert "heading" in chunks[0]
        assert "text" in chunks[0]


# ---------------------------------------------------------------------------
# 2. ECPressSource
# ---------------------------------------------------------------------------

class TestECPressSource:
    def test_instantiation(self):
        src = ECPressSource()
        assert src.source_id == "ec_press"
        assert src.source_type == "EC_PRESS_RELEASE"
        assert src.institution == "European Commission"

    def test_scrape_parses_json(self):
        api_response = json.dumps({
            "items": [
                {"title": "Press Release 1", "url": "https://ec.europa.eu/commission/presscorner/detail/en/IP_24_001", "date": "2024-01-10"},
                {"title": "Press Release 2", "url": "https://ec.europa.eu/commission/presscorner/detail/en/IP_24_002", "date": "2024-01-11"},
            ]
        })
        src = ECPressSource()

        call_count = 0
        async def fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "api" in url or "limit" in url:
                return api_response
            return SAMPLE_HTML

        src._get = fake_get
        docs = run(src.scrape(limit=2))
        assert len(docs) == 2
        assert docs[0].source_id == "ec_press"
        assert docs[0].title  # non-empty title

    def test_fetch_one_returns_parsed_document(self):
        src = ECPressSource()
        src._get = AsyncMock(return_value=SAMPLE_HTML)
        doc = run(src.fetch_one("https://ec.europa.eu/commission/presscorner/detail/en/IP_24_001"))
        assert isinstance(doc, ParsedDocument)
        assert doc.source_id == "ec_press"
        assert doc.date == "2024-01-15"
        assert doc.institution == "European Commission"

    def test_chunks_from_ec_press_doc(self):
        doc = ParsedDocument(
            source_id="ec_press",
            source_type="EC_PRESS_RELEASE",
            url="https://ec.europa.eu/commission/presscorner/detail/en/IP_24_001",
            title="Commission launches new initiative",
            date="2024-01-15",
            institution="European Commission",
            language="en",
            raw_text="Commission text",
            markdown="# Commission launches new initiative\n\n## Background\n\nThe Commission today...\n\n## Details\n\nFurther details.",
        )
        chunks = markdown_to_chunks(doc)
        assert len(chunks) >= 1
        assert all(c["source_type"] == "EC_PRESS_RELEASE" for c in chunks)


# ---------------------------------------------------------------------------
# 3. ECBPublicationsSource
# ---------------------------------------------------------------------------

class TestECBPublicationsSource:
    def test_instantiation(self):
        src = ECBPublicationsSource()
        assert src.source_id == "ecb_publications"
        assert src.source_type == "ECB_ECONOMIC_BULLETIN"
        assert src.institution == "European Central Bank"

    def test_scrape_finds_bulletin_links(self):
        listing_html = '''
        <html><body>
        <a href="/pub/economic-bulletin/html/eb202301~abcdef.en.html">EB 2023/1</a>
        <a href="/pub/economic-bulletin/html/eb202302~ghijkl.en.html">EB 2023/2</a>
        <a href="/other/page.html">Not a bulletin</a>
        </body></html>
        '''
        src = ECBPublicationsSource()
        call_count = 0
        async def fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "index" in url:
                return listing_html
            return SAMPLE_HTML
        src._get = fake_get
        docs = run(src.scrape(limit=2))
        assert len(docs) == 2
        assert all(d.source_id == "ecb_publications" for d in docs)

    def test_fetch_one_returns_parsed_document(self):
        src = ECBPublicationsSource()
        src._get = AsyncMock(return_value=SAMPLE_HTML)
        doc = run(src.fetch_one("https://www.ecb.europa.eu/pub/economic-bulletin/html/eb202301~abc.en.html"))
        assert isinstance(doc, ParsedDocument)
        assert doc.source_id == "ecb_publications"
        assert doc.institution == "European Central Bank"

    def test_date_extracted_from_url(self):
        src = ECBPublicationsSource()
        src._get = AsyncMock(return_value="<html><body>no date here</body></html>")
        doc = run(src.fetch_one("https://www.ecb.europa.eu/pub/economic-bulletin/html/eb202306~abc.en.html"))
        assert doc.date == "2023-06-01"

    def test_chunks_from_ecb_doc(self):
        doc = ParsedDocument(
            source_id="ecb_publications",
            source_type="ECB_ECONOMIC_BULLETIN",
            url="https://www.ecb.europa.eu/pub/economic-bulletin/html/eb202301.en.html",
            title="ECB Economic Bulletin 2023/1",
            date="2023-01-01",
            institution="European Central Bank",
            language="en",
            raw_text="Economic bulletin text",
            markdown="# ECB Economic Bulletin\n\n## Overview\n\nEconomic overview.\n\n## Inflation\n\nInflation data.",
        )
        chunks = markdown_to_chunks(doc)
        assert len(chunks) >= 1
        assert chunks[0]["institution"] == "European Central Bank"


# ---------------------------------------------------------------------------
# 4. ECAReportsSource
# ---------------------------------------------------------------------------

class TestECAReportsSource:
    def test_instantiation(self):
        src = ECAReportsSource()
        assert src.source_id == "eca_reports"
        assert src.source_type == "ECA_SPECIAL_REPORT"
        assert src.institution == "European Court of Auditors"

    def test_scrape_finds_report_links(self):
        listing_html = '''
        <html><body>
        <a href="/en/publications/SR-2023-15">Special Report 15/2023</a>
        <a href="/en/publications/SR-2023-14">Special Report 14/2023</a>
        <a href="/en/publications?category=special_report">All reports</a>
        </body></html>
        '''
        src = ECAReportsSource()
        call_count = 0
        async def fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "?category" in url:
                return listing_html
            return SAMPLE_HTML
        src._get = fake_get
        docs = run(src.scrape(limit=2))
        assert len(docs) >= 1

    def test_fetch_one_returns_parsed_document(self):
        src = ECAReportsSource()
        rich_html = """
        <html><head><title>Special Report 15/2023 2023-09-12</title></head>
        <body>
        <time datetime="2023-09-12">12 September 2023</time>
        <section class="executive-summary">
        <h2>Executive summary</h2><p>The ECA examined...</p>
        </section>
        </body></html>
        """
        src._get = AsyncMock(return_value=rich_html)
        doc = run(src.fetch_one("https://www.eca.europa.eu/en/publications/SR-2023-15"))
        assert isinstance(doc, ParsedDocument)
        assert doc.source_id == "eca_reports"
        assert doc.date == "2023-09-12"

    def test_chunks_from_eca_doc(self):
        doc = ParsedDocument(
            source_id="eca_reports",
            source_type="ECA_SPECIAL_REPORT",
            url="https://www.eca.europa.eu/en/publications/SR-2023-15",
            title="ECA Special Report 15/2023",
            date="2023-09-12",
            institution="European Court of Auditors",
            language="en",
            raw_text="Audit findings text",
            markdown="# ECA Special Report 15/2023\n\n## Executive summary\n\nThe ECA audited...\n\n## Key findings\n\nFindings here.",
        )
        chunks = markdown_to_chunks(doc)
        assert len(chunks) >= 1
        assert chunks[0]["source_type"] == "ECA_SPECIAL_REPORT"


# ---------------------------------------------------------------------------
# 5. EPRSBriefingsSource
# ---------------------------------------------------------------------------

class TestEPRSBriefingsSource:
    def test_instantiation(self):
        src = EPRSBriefingsSource()
        assert src.source_id == "eprs_briefings"
        assert src.source_type == "EPRS_BRIEFING"
        assert src.institution == "EP Research Service"

    def test_scrape_finds_briefing_links(self):
        listing_html = '''
        <html><body>
        <a href="/thinktank/en/document/EPRS_BRI(2024)760373">Briefing 1</a>
        <a href="/thinktank/en/document/EPRS_BRI(2024)760374">Briefing 2</a>
        <a href="/thinktank/en/search.html?category=briefing">Search page</a>
        </body></html>
        '''
        src = EPRSBriefingsSource()
        call_count = 0
        async def fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "search.html" in url:
                return listing_html
            return SAMPLE_HTML
        src._get = fake_get
        docs = run(src.scrape(limit=2))
        assert len(docs) == 2
        assert all(d.source_id == "eprs_briefings" for d in docs)

    def test_fetch_one_returns_parsed_document(self):
        src = EPRSBriefingsSource()
        html_with_author = SAMPLE_HTML.replace(
            '<body>',
            '<body><meta name="author" content="Jane Doe"><meta name="keywords" content="EU, policy">'
        )
        src._get = AsyncMock(return_value=html_with_author)
        doc = run(src.fetch_one("https://www.europarl.europa.eu/thinktank/en/document/EPRS_BRI(2024)760373"))
        assert isinstance(doc, ParsedDocument)
        assert doc.source_id == "eprs_briefings"
        assert doc.metadata.get("author") == "Jane Doe"

    def test_chunks_from_eprs_doc(self):
        doc = ParsedDocument(
            source_id="eprs_briefings",
            source_type="EPRS_BRIEFING",
            url="https://www.europarl.europa.eu/thinktank/en/document/EPRS_BRI(2024)760373",
            title="EU Digital Policy Briefing",
            date="2024-01-15",
            institution="EP Research Service",
            language="en",
            raw_text="Briefing text content",
            markdown="# EU Digital Policy Briefing\n\n## Background\n\nPolicy context.\n\n## Analysis\n\nDetailed analysis.",
        )
        chunks = markdown_to_chunks(doc)
        assert len(chunks) >= 1
        assert chunks[0]["source_id"] == "eprs_briefings"
        assert "heading" in chunks[0]
        assert "text" in chunks[0]

    def test_date_fallback_to_today(self):
        src = EPRSBriefingsSource()
        src._get = AsyncMock(return_value="<html><body>no date info</body></html>")
        doc = run(src.fetch_one("https://www.europarl.europa.eu/thinktank/en/document/EPRS_BRI(2024)760373"))
        assert doc is not None
        # Date should be a valid YYYY-MM-DD string (today's date)
        from datetime import datetime
        try:
            datetime.strptime(doc.date, "%Y-%m-%d")
        except ValueError:
            pytest.fail(f"Date '{doc.date}' is not in YYYY-MM-DD format")
