"""RetrievalContext — structured XML output format for HybridRAG results.

The XML schema is the single wire format between the retriever and any consumer
(MCP tool, REST API, or agent LLM). No plain-dict alternative is exposed.

Schema example::

    <retrieval_context session_id="s1" query="AI surveillance" total_chunks="8">
      <graph_matches count="3">
        <entity name="EUROPOL" label="ORG" similarity="0.95">
          <chunk id="c1" doc_id="d1" mention_weight="0.88">
            <text>Europol reported ...</text>
            <metadata title="EP Debate" url="https://..." source="eurlex" date="2024-03-01"/>
          </chunk>
        </entity>
      </graph_matches>
      <semantic_matches count="5">
        <chunk id="c2" doc_id="d2" cosine_score="0.85">
          <text>AI Act provisions ...</text>
          <metadata title="AI Act" url="https://..." source="eurlex" date="2024-02-01"/>
        </chunk>
      </semantic_matches>
      <fused_results count="8">
        <chunk id="c1" doc_id="d1" rrf_score="0.033" sources="graph,semantic" rank="1">
          <text>Europol reported ...</text>
          <metadata title="EP Debate" url="https://..." source="eurlex" date="2024-03-01"/>
          <entities>
            <entity name="EUROPOL" label="ORG"/>
          </entities>
        </chunk>
      </fused_results>
    </retrieval_context>
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field


@dataclass
class ChunkMatch:
    """A single text chunk returned by retrieval.

    Attributes:
        chunk_id: Unique chunk identifier.
        doc_id: Source document identifier.
        text: Full chunk text.
        metadata: Dict with optional keys: title, url, source, date.
        entities: List of entity dicts linked to this chunk (name, label).
    """

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    entities: list[dict] = field(default_factory=list)


@dataclass
class EntityMatch:
    """An entity returned by cosine search with its associated chunks.

    Attributes:
        name: Entity name (normalised, uppercase).
        label: Entity type label.
        similarity: Cosine similarity score (0–1).
        chunks: Chunks linked to this entity via MENTIONS edges.
    """

    name: str
    label: str
    similarity: float
    chunks: list[ChunkMatch] = field(default_factory=list)


@dataclass
class RetrievalContext:
    """Full retrieval result: entity-first graph matches + semantic matches + RRF fusion.

    Attributes:
        query: Original query string.
        session_id: Session namespace used for pgvector search.
        graph_matches: Entities found via cosine search, each with linked chunks.
        semantic_matches: Chunks found via direct cosine search.
        fused_results: RRF-fused deduped chunks, sorted by rrf_score descending.
    """

    query: str
    session_id: str
    graph_matches: list[EntityMatch] = field(default_factory=list)
    semantic_matches: list[ChunkMatch] = field(default_factory=list)
    fused_results: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # XML serialisation (single wire format)
    # ------------------------------------------------------------------

    def to_xml(self) -> str:
        """Serialise this context to the canonical XML string.

        All user-controlled strings are HTML-escaped to prevent injection.
        """
        total = len(self.fused_results)
        lines: list[str] = [
            f'<retrieval_context session_id="{_esc(self.session_id)}"'
            f' query="{_esc(self.query)}" total_chunks="{total}">',
        ]

        # ── graph_matches ──────────────────────────────────────────────
        lines.append(f'  <graph_matches count="{len(self.graph_matches)}">')
        for em in self.graph_matches:
            lines.append(
                f'    <entity name="{_esc(em.name)}" label="{_esc(em.label)}"'
                f' similarity="{em.similarity:.4f}">'
            )
            for cm in em.chunks:
                lines.append(
                    f'      <chunk id="{_esc(cm.chunk_id)}" doc_id="{_esc(cm.doc_id)}"'
                    f' mention_weight="{_mention_weight(em.similarity, cm):.4f}">'
                )
                lines.append(f"        <text>{_esc(cm.text)}</text>")
                lines.append(_metadata_tag(cm.metadata, indent=8))
                lines.append("      </chunk>")
            lines.append("    </entity>")
        lines.append("  </graph_matches>")

        # ── semantic_matches ───────────────────────────────────────────
        lines.append(f'  <semantic_matches count="{len(self.semantic_matches)}">')
        for cm in self.semantic_matches:
            score = cm.metadata.pop("cosine_score", 0.0) if isinstance(cm.metadata, dict) else 0.0
            lines.append(
                f'    <chunk id="{_esc(cm.chunk_id)}" doc_id="{_esc(cm.doc_id)}"'
                f' cosine_score="{score:.4f}">'
            )
            lines.append(f"      <text>{_esc(cm.text)}</text>")
            lines.append(_metadata_tag(cm.metadata, indent=6))
            lines.append("    </chunk>")
        lines.append("  </semantic_matches>")

        # ── fused_results ──────────────────────────────────────────────
        lines.append(f'  <fused_results count="{len(self.fused_results)}">')
        for fc in self.fused_results:
            lines.append(
                f'    <chunk id="{_esc(fc.get("chunk_id",""))}"'
                f' doc_id="{_esc(fc.get("doc_id",""))}"'
                f' rrf_score="{fc.get("rrf_score", 0.0):.6f}"'
                f' sources="{_esc(fc.get("sources",""))}"'
                f' rank="{fc.get("rank", 0)}">'
            )
            lines.append(f'      <text>{_esc(fc.get("text",""))}</text>')
            lines.append(_metadata_tag(fc.get("metadata", {}), indent=6))
            ents = fc.get("entities", [])
            if ents:
                lines.append("      <entities>")
                for e in ents:
                    lines.append(
                        f'        <entity name="{_esc(e.get("name",""))}"'
                        f' label="{_esc(e.get("label",""))}"/>'
                    )
                lines.append("      </entities>")
            lines.append("    </chunk>")
        lines.append("  </fused_results>")

        lines.append("</retrieval_context>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _esc(value: object) -> str:
    """HTML-escape a value for safe XML attribute/text embedding."""
    return html.escape(str(value) if value is not None else "", quote=True)


def _metadata_tag(meta: dict, indent: int = 0) -> str:
    """Render a <metadata> tag from a dict, escaping all values."""
    pad = " " * indent
    title = _esc(meta.get("title", ""))
    url = _esc(meta.get("url", ""))
    source = _esc(meta.get("source", ""))
    date = _esc(meta.get("date", ""))
    return f'{pad}<metadata title="{title}" url="{url}" source="{source}" date="{date}"/>'


def _mention_weight(entity_similarity: float, chunk: ChunkMatch) -> float:
    """Approximate mention weight — entity similarity scaled by chunk position."""
    # Placeholder: the retriever sets this on the chunk's metadata
    return chunk.metadata.get("mention_weight", entity_similarity)
