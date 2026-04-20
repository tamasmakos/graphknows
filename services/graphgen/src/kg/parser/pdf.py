"""PDF parser using pymupdf4llm (Markdown extraction from PDFs)."""

from __future__ import annotations

from pathlib import Path

from kg.parser import BaseParser, ParsedDocument


class PDFParser(BaseParser):
    extensions = ("pdf",)
    mime_types = ("application/pdf",)

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import pymupdf4llm  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pymupdf4llm is required for PDF parsing: pip install pymupdf4llm"
            ) from exc

        doc = ParsedDocument.from_path(path, "application/pdf")
        md_text: str = pymupdf4llm.to_markdown(str(path))

        # Delegate to MarkdownParser for heading-aware chunking
        from kg.parser.markdown import MarkdownParser

        md_parser = MarkdownParser()
        doc.chunks = list(md_parser._chunk(doc.doc_id, md_text))
        return doc
