"""
Parser registry: imports all concrete parsers so they self-register.
Import this module once at startup to activate all parsers.
"""

# ruff: noqa: F401
from kg.parser.text import TextParser
from kg.parser.markdown import MarkdownParser
from kg.parser.pdf import PDFParser
from kg.parser.docx import DocxParser
from kg.parser.pptx import PptxParser
from kg.parser.excel import ExcelParser
from kg.parser.html import HTMLParser
from kg.parser.image import ImageParser

__all__ = [
    "TextParser",
    "MarkdownParser",
    "PDFParser",
    "DocxParser",
    "PptxParser",
    "ExcelParser",
    "HTMLParser",
    "ImageParser",
]
