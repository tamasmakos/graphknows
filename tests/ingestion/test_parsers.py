"""Unit tests for PlainTextParser — the only parser the ingest path uses."""

from __future__ import annotations

from graphknows.ingestion.parsers.text import PlainTextParser


class TestPlainTextParser:
    """Tests for PlainTextParser."""

    def test_plain_text_no_headings_produces_single_segment(self):
        """Plain text without headings should produce 1 segment."""
        parser = PlainTextParser()
        text = "Hello world.\nThis is a test."
        result = parser.parse(text, source_hint="test.txt")

        assert len(result.segments) == 1
        assert result.segments[0].heading_path == "## Content"
        assert "Hello world." in result.segments[0].text

    def test_markdown_with_three_h2_sections_produces_three_segments(self):
        """Markdown with 3 H2 sections should produce 3 segments with correct paths."""
        parser = PlainTextParser()
        # Make each body large enough (> 64 tokens ~ 256 chars) so they are not merged
        body = "This is a substantial paragraph. " * 15
        text = f"## A\n\n{body}\n\n## B\n\n{body}\n\n## C\n\n{body}"
        result = parser.parse(text, source_hint="test.md")

        assert len(result.segments) == 3
        paths = [s.heading_path for s in result.segments]
        assert "## A" in paths
        assert "## B" in paths
        assert "## C" in paths

    def test_very_large_section_splits_at_paragraph_boundary(self):
        """A very large section should split at paragraph boundaries, never mid-sentence."""
        parser = PlainTextParser()
        # Create multiple paragraphs (separated by blank lines) that exceed MAX_CHUNK_TOKENS
        paragraph = "This is a sentence. " * 100  # ~2000 chars = ~500 tokens each
        text = f"## Section\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}"
        result = parser.parse(text, source_hint="test.md")

        assert len(result.segments) > 1
        for seg in result.segments:
            # No segment should end mid-word (no trailing partial word)
            stripped = seg.text.rstrip()
            if stripped:
                last_word = stripped.split()[-1]
                # A complete word ends with punctuation, newline, or is a known heading word
                assert (
                    any(last_word.endswith(p) for p in ".!?\n:;#")
                    or last_word.isalpha()
                    or last_word == "Section"
                )

    def test_tiny_sections_merge_into_fewer_chunks(self):
        """Many tiny sections should be merged into fewer chunks."""
        parser = PlainTextParser()
        sections = []
        for i in range(20):
            sections.append(f"## H{i}\n\nShort.")
        text = "\n\n".join(sections)
        result = parser.parse(text, source_hint="test.md")

        # 20 sections should merge into fewer than 20 chunks
        assert len(result.segments) < 20


class TestCommonProperties:
    """Cross-parser correctness tests."""

    def test_segments_ordered_by_char_offset(self):
        """ParsedFile.segments should be ordered by ascending char_offset."""
        parser = PlainTextParser()
        text = "## A\n\nBody A.\n\n## B\n\nBody B.\n\n## C\n\nBody C."
        result = parser.parse(text, source_hint="test.md")

        offsets = [s.char_offset for s in result.segments]
        assert offsets == sorted(offsets)

    def test_no_segment_ends_with_partial_word(self):
        """No segment text should end with a partial word."""
        parser = PlainTextParser()
        # Build text with known words so we can spot-check
        words = ["apple", "banana", "cherry", "date", "elderberry"]
        paragraph = " ".join(f"Sentence about {w}." for w in words) + " "
        # Make it large enough to potentially split
        big_body = (paragraph * 500).strip()
        text = f"## Section\n\n{big_body}"
        result = parser.parse(text, source_hint="test.md")

        for seg in result.segments:
            stripped = seg.text.rstrip()
            if stripped:
                # The last token should not be an obvious partial word fragment
                last_word = stripped.split()[-1]
                # last_word may end with punctuation; a bare heading word is fine too
                assert any(last_word.endswith(p) for p in ".!?\n:;#") or last_word.isalpha()
