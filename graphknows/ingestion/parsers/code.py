"""Code parser: Python AST-aware, fallback heuristic for other languages."""

from __future__ import annotations

import ast
import os

from .base import BaseParser, ParsedFile, TextSegment


class CodeParser(BaseParser):
    """Parse source code files into Markdown-formatted segments."""

    def parse(self, text: str, source_hint: str = "", title: str = "") -> ParsedFile:
        """Parse source code into Markdown-formatted segments.

        Args:
            text: Raw source code.
            source_hint: Original path or URL for metadata and hashing.
            title: Optional title override.

        Returns:
            ParsedFile with ordered TextSegment list.
        """
        text = text.replace("\r\n", "\n")
        source_id = self._make_source_id(source_hint, text)
        ext = os.path.splitext(source_hint)[-1].lower() if source_hint else ".txt"
        lang = self._ext_to_lang(ext)

        if not title:
            title = os.path.basename(source_hint) if source_hint else "Untitled"

        if ext == ".py":
            segments = self._parse_python(text, title, lang)
        else:
            segments = self._parse_generic(text, title, lang)

        return ParsedFile(
            source_id=source_id,
            title=title,
            parser_name="CodeParser",
            segments=segments,
        )

    def _parse_python(self, text: str, title: str, lang: str) -> list[TextSegment]:
        segments: list[TextSegment] = []
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # Fallback to generic parsing if AST fails
            return self._parse_generic(text, title, lang)

        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            segments.append(
                TextSegment(
                    text=module_docstring,
                    heading_path="## Module",
                    char_offset=0,
                    metadata={
                        "language": lang,
                        "chunk_type": "module",
                        "name": "module",
                        "lineno": 1,
                    },
                )
            )

        lines = text.split("\n")
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", len(lines))
                if end is None:
                    end = len(lines)
                source_segment = "\n".join(lines[start:end])

                node_docstring = ast.get_docstring(node)
                prose = f"""## {node.name}\n\n"""
                if node_docstring:
                    prose += f"{node_docstring}\n\n"
                prose += f"```{lang}\n{source_segment}\n```"

                chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
                segments.append(
                    TextSegment(
                        text=prose,
                        heading_path=f"## {node.name}",
                        char_offset=node.lineno - 1,
                        metadata={
                            "language": lang,
                            "chunk_type": chunk_type,
                            "name": node.name,
                            "lineno": node.lineno,
                        },
                    )
                )

        if not segments:
            return self._parse_generic(text, title, lang)
        return segments

    def _parse_generic(self, text: str, title: str, lang: str) -> list[TextSegment]:
        heading = f"## {title}"
        body = f"```{lang}\n{text}\n```"
        return [
            TextSegment(
                text=f"{heading}\n\n{body}",
                heading_path=heading,
                char_offset=0,
                metadata={"language": lang, "chunk_type": "file", "name": title},
            ),
        ]

    def _ext_to_lang(self, ext: str) -> str:
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".sql": "sql",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
        }
        return mapping.get(ext, "text")
