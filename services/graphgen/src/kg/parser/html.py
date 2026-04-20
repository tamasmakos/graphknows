"""HTML parser using trafilatura + markdownify."""

from __future__ import annotations

from pathlib import Path

from kg.parser import BaseParser, ParsedDocument


class HTMLParser(BaseParser):
    extensions = ("html", "htm")
    mime_types = ("text/html",)

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import trafilatura  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("trafilatura is required: pip install trafilatura") from exc

        doc = ParsedDocument.from_path(path, "text/html")
        raw_html = path.read_text(errors="replace")

        # Extract main body text via trafilatura
        extracted = (
            trafilatura.extract(
                raw_html,
                include_comments=False,
                include_tables=True,
                output_format="markdown",
            )
            or ""
        )

        from kg.parser.markdown import MarkdownParser

        md_parser = MarkdownParser()
        doc.chunks = list(md_parser._chunk(doc.doc_id, extracted))
        return doc
