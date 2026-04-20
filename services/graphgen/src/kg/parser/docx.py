"""Word document parser (.docx) using python-docx."""

from __future__ import annotations

from pathlib import Path

from kg.parser import BaseParser, ParsedChunk, ParsedDocument


class DocxParser(BaseParser):
    extensions = ("docx",)
    mime_types = ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",)

    def parse(self, path: Path, chunk_size: int = 1200, chunk_overlap: int = 100) -> ParsedDocument:
        try:
            import docx  # python-docx
        except ImportError as exc:
            raise ImportError("python-docx is required: pip install python-docx") from exc

        doc = ParsedDocument.from_path(path, self.mime_types[0])
        document = docx.Document(str(path))

        heading_path: list[str] = []
        text_buffer: list[str] = []

        chunks: list[ParsedChunk] = []
        position = 0

        def _flush(heading: list[str]) -> None:
            nonlocal position
            combined = " ".join(text_buffer).strip()
            if not combined:
                return
            words = combined.split()
            step = max(1, chunk_size - chunk_overlap)
            for start in range(0, max(1, len(words)), step):
                segment = " ".join(words[start : start + chunk_size])
                if segment.strip():
                    chunks.append(ParsedChunk.make(doc.doc_id, position, segment, list(heading)))
                    position += 1
            text_buffer.clear()

        for para in document.paragraphs:
            style = para.style.name if para.style else ""
            if style.startswith("Heading"):
                _flush(heading_path)
                level = int(style.replace("Heading ", "").strip() or "1")
                while len(heading_path) >= level:
                    heading_path.pop()
                heading_path.append(para.text.strip())
            else:
                if para.text.strip():
                    text_buffer.append(para.text.strip())

        _flush(heading_path)
        doc.chunks = chunks
        return doc
