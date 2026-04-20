"""Plain text parser (txt, csv, log, etc.)."""

from __future__ import annotations

from pathlib import Path

from kg.parser import BaseParser, ParsedChunk, ParsedDocument


class TextParser(BaseParser):
    extensions = ("txt", "csv", "log", "tsv")
    mime_types = ("text/plain", "text/csv")

    def parse(self, path: Path, chunk_size: int = 1200, chunk_overlap: int = 100) -> ParsedDocument:
        doc = ParsedDocument.from_path(path, "text/plain")
        text = path.read_text(errors="replace")
        doc.chunks = _split_text(text, doc.doc_id, chunk_size, chunk_overlap)
        return doc


def _split_text(
    text: str,
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ParsedChunk]:
    words = text.split()
    chunks: list[ParsedChunk] = []
    step = max(1, chunk_size - chunk_overlap)
    for i, start in enumerate(range(0, len(words), step)):
        segment = " ".join(words[start : start + chunk_size])
        if segment.strip():
            chunks.append(ParsedChunk.make(doc_id, i, segment))
    return chunks
