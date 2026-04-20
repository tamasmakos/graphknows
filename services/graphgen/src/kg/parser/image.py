"""Image parser using pytesseract OCR (png, jpg, tiff, bmp)."""

from __future__ import annotations

from pathlib import Path

from kg.parser import BaseParser, ParsedChunk, ParsedDocument


class ImageParser(BaseParser):
    extensions = ("png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp")
    mime_types = ("image/png", "image/jpeg", "image/tiff", "image/bmp")

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import pytesseract  # type: ignore[import]
            from PIL import Image  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pytesseract and Pillow are required for OCR: pip install pytesseract Pillow"
            ) from exc

        doc = ParsedDocument.from_path(path, "image/png")
        image = Image.open(str(path))
        text: str = pytesseract.image_to_string(image)

        from kg.parser.text import _split_text

        doc.chunks = _split_text(text, doc.doc_id, 1200, 100)
        return doc
