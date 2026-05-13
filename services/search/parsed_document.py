from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import re


@dataclass
class ParsedDocument:
    source_id: str       # e.g. 'ep_resolutions'
    source_type: str     # e.g. 'EP_RESOLUTION'
    url: str
    title: str
    date: str            # ISO 8601 YYYY-MM-DD
    institution: str     # e.g. 'European Parliament'
    language: str        # ISO 639-1
    raw_text: str
    markdown: str        # H1=title, H2=section, H3=subsection
    metadata: dict = field(default_factory=dict)
    doc_id: str = ''     # set post-init if empty

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.sha1(
                f"{self.source_id}:{self.url}".encode()
            ).hexdigest()[:16]


def markdown_to_chunks(
    doc: ParsedDocument,
    max_tokens: int = 512,
    overlap_sentences: int = 2,
) -> list[dict]:
    """Split markdown on H2 boundaries. Each chunk: dict with chunk_id, doc_id,
    source_id, source_type, url, date, institution, heading, text."""
    # Split on H2 headings
    sections = re.split(r'(?m)^## ', doc.markdown)
    chunks = []
    for i, section in enumerate(sections):
        if not section.strip():
            continue
        lines = section.strip().splitlines()
        heading = lines[0].strip() if lines else f'Section {i}'
        text = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''
        if not text:
            text = heading
        chunk_id = hashlib.sha1(f"{doc.doc_id}:{i}".encode()).hexdigest()[:16]
        chunks.append({
            'chunk_id': chunk_id,
            'doc_id': doc.doc_id,
            'source_id': doc.source_id,
            'source_type': doc.source_type,
            'url': doc.url,
            'date': doc.date,
            'institution': doc.institution,
            'language': doc.language,
            'heading': heading,
            'text': text[:max_tokens * 4],  # approx token budget
        })
    return chunks or [{
        'chunk_id': doc.doc_id,
        'doc_id': doc.doc_id,
        'source_id': doc.source_id,
        'source_type': doc.source_type,
        'url': doc.url,
        'date': doc.date,
        'institution': doc.institution,
        'language': doc.language,
        'heading': doc.title,
        'text': doc.markdown[:max_tokens * 4],
    }]
