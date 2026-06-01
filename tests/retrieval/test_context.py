"""Tests for RetrievalContext XML/dict serialisation and helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from graphknows.memory.query.context import (
    ChunkMatch,
    EntityMatch,
    RetrievalContext,
    _esc,
    _mention_weight,
    _metadata_tag,
)

# ---------------------------------------------------------------------------
# _esc helper
# ---------------------------------------------------------------------------


def test_esc_escapes_less_than():
    assert "&lt;" in _esc("<")


def test_esc_escapes_greater_than():
    assert "&gt;" in _esc(">")


def test_esc_escapes_ampersand():
    assert "&amp;" in _esc("&")


def test_esc_escapes_double_quote():
    assert "&quot;" in _esc('"')


def test_esc_none_returns_string():
    result = _esc(None)
    assert isinstance(result, str)


def test_esc_plain_text_unchanged():
    assert _esc("hello world") == "hello world"


def test_esc_xss_payload_fully_escaped():
    result = _esc("<script>alert(1)</script>")
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


# ---------------------------------------------------------------------------
# _metadata_tag helper
# ---------------------------------------------------------------------------


def test_metadata_tag_all_fields_present():
    tag = _metadata_tag(
        {"title": "My Title", "url": "http://x.com", "source": "src", "date": "2024-01-01"}
    )
    assert 'title="My Title"' in tag
    assert 'url="http://x.com"' in tag
    assert 'source="src"' in tag
    assert 'date="2024-01-01"' in tag


def test_metadata_tag_missing_keys_produce_empty_attributes():
    tag = _metadata_tag({})
    assert 'title=""' in tag
    assert 'url=""' in tag
    assert 'source=""' in tag
    assert 'date=""' in tag


def test_metadata_tag_with_indent_adds_leading_spaces():
    tag = _metadata_tag({}, indent=4)
    assert tag.startswith("    ")


def test_metadata_tag_zero_indent_no_leading_space():
    tag = _metadata_tag({}, indent=0)
    assert tag.startswith("<metadata")


def test_metadata_tag_escapes_special_chars_in_values():
    tag = _metadata_tag({"title": "<script>alert(1)</script>"})
    assert "<script>" not in tag
    assert "&lt;script&gt;" in tag


# ---------------------------------------------------------------------------
# _mention_weight helper
# ---------------------------------------------------------------------------


def test_mention_weight_returns_metadata_value_when_present():
    chunk = ChunkMatch(
        chunk_id="c1",
        doc_id="d1",
        text="text",
        metadata={"mention_weight": 0.75},
    )
    result = _mention_weight(0.5, chunk)
    assert result == 0.75


def test_mention_weight_falls_back_to_entity_similarity_when_missing():
    chunk = ChunkMatch(chunk_id="c1", doc_id="d1", text="text", metadata={})
    result = _mention_weight(0.88, chunk)
    assert result == 0.88


def test_mention_weight_metadata_value_overrides_entity_similarity():
    chunk = ChunkMatch(
        chunk_id="c1",
        doc_id="d1",
        text="text",
        metadata={"mention_weight": 0.3},
    )
    result = _mention_weight(0.99, chunk)
    assert result == 0.3
    assert result != 0.99


# ---------------------------------------------------------------------------
# RetrievalContext.to_dict()
# ---------------------------------------------------------------------------


def _make_entity_chunk(chunk_id="c1", doc_id="d1", text="chunk text"):
    return ChunkMatch(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        metadata={"title": "Doc"},
        entities=[{"name": "EU", "label": "ORG"}],
    )


def _make_entity_match(name="EUROPOL", label="ORG", similarity=0.95):
    return EntityMatch(name=name, label=label, similarity=similarity, chunks=[_make_entity_chunk()])


def _make_full_context():
    sem_chunk = ChunkMatch(
        chunk_id="c2",
        doc_id="d2",
        text="AI provision",
        metadata={"cosine_score": 0.85},
    )
    fused = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text": "EU text",
            "rrf_score": 0.033,
            "sources": "graph",
            "rank": 1,
        }
    ]
    return RetrievalContext(
        query="EU AI policy",
        session_id="sess-1",
        graph_matches=[_make_entity_match()],
        semantic_matches=[sem_chunk],
        fused_results=fused,
    )


def test_to_dict_has_required_top_level_keys():
    ctx = _make_full_context()
    d = ctx.to_dict()
    for key in (
        "query",
        "session_id",
        "search_type",
        "graph_matches",
        "semantic_matches",
        "fused_results",
    ):
        assert key in d


def test_to_dict_graph_matches_structure():
    ctx = _make_full_context()
    d = ctx.to_dict()
    gm = d["graph_matches"]
    assert len(gm) == 1
    entry = gm[0]
    assert entry["name"] == "EUROPOL"
    assert entry["label"] == "ORG"
    assert entry["similarity"] == round(0.95, 4)
    assert len(entry["chunks"]) == 1
    assert entry["chunks"][0]["text"] == "chunk text"


def test_to_dict_empty_context_has_empty_lists_not_none():
    ctx = RetrievalContext(query="q", session_id="s")
    d = ctx.to_dict()
    assert d["graph_matches"] == []
    assert d["semantic_matches"] == []
    assert d["fused_results"] == []


def test_to_dict_search_type_preserved():
    ctx = RetrievalContext(query="q", session_id="s", search_type="SEMANTIC")
    assert ctx.to_dict()["search_type"] == "SEMANTIC"


def test_to_dict_similarity_is_rounded_to_4dp():
    em = EntityMatch(name="X", label="ORG", similarity=0.123456789)
    ctx = RetrievalContext(query="q", session_id="s", graph_matches=[em])
    d = ctx.to_dict()
    assert d["graph_matches"][0]["similarity"] == round(0.123456789, 4)


# ---------------------------------------------------------------------------
# RetrievalContext.to_xml()
# ---------------------------------------------------------------------------


def _simple_ctx(**kwargs):
    return RetrievalContext(query="test query", session_id="s1", **kwargs)


def test_to_xml_produces_valid_xml_for_full_context():
    ctx = _make_full_context()
    xml_str = ctx.to_xml()
    ET.fromstring(xml_str)  # raises if invalid


def test_to_xml_root_element_has_correct_attributes():
    ctx = _simple_ctx()
    root = ET.fromstring(ctx.to_xml())
    assert root.tag == "retrieval_context"
    assert root.attrib["session_id"] == "s1"
    assert root.attrib["query"] == "test query"
    assert root.attrib["total_chunks"] == "0"


def test_to_xml_total_chunks_equals_fused_results_count():
    fused = [
        {
            "chunk_id": f"c{i}",
            "doc_id": "d",
            "text": f"t{i}",
            "rrf_score": 0.1,
            "sources": "graph",
            "rank": i,
        }
        for i in range(3)
    ]
    ctx = _simple_ctx(fused_results=fused)
    root = ET.fromstring(ctx.to_xml())
    assert root.attrib["total_chunks"] == "3"


def test_to_xml_graph_matches_count_attr():
    em1 = EntityMatch(name="EU", label="ORG", similarity=0.9)
    em2 = EntityMatch(name="NATO", label="ORG", similarity=0.8)
    ctx = _simple_ctx(graph_matches=[em1, em2])
    root = ET.fromstring(ctx.to_xml())
    gm = root.find("graph_matches")
    assert gm is not None
    assert gm.attrib["count"] == "2"


def test_to_xml_semantic_matches_count_attr():
    chunks = [ChunkMatch(chunk_id=f"c{i}", doc_id="d1", text=f"text{i}") for i in range(3)]
    ctx = _simple_ctx(semantic_matches=chunks)
    root = ET.fromstring(ctx.to_xml())
    sm = root.find("semantic_matches")
    assert sm is not None
    assert sm.attrib["count"] == "3"


def test_to_xml_fused_results_count_attr():
    fused = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text": "t",
            "rrf_score": 0.1,
            "sources": "graph",
            "rank": 1,
        }
    ]
    ctx = _simple_ctx(fused_results=fused)
    root = ET.fromstring(ctx.to_xml())
    fr = root.find("fused_results")
    assert fr is not None
    assert fr.attrib["count"] == "1"


def test_to_xml_escapes_xss_in_query_attr():
    ctx = RetrievalContext(query="<script>alert(1)</script>", session_id="s")
    xml_str = ctx.to_xml()
    assert "<script>" not in xml_str
    root = ET.fromstring(xml_str)
    assert root.attrib["query"] == "<script>alert(1)</script>"


def test_to_xml_escapes_xss_in_chunk_text():
    chunk = ChunkMatch(
        chunk_id="c1",
        doc_id="d1",
        text='<b onclick="evil()">click</b>',
    )
    em = EntityMatch(name="EU", label="ORG", similarity=0.9, chunks=[chunk])
    ctx = _simple_ctx(graph_matches=[em])
    xml_str = ctx.to_xml()
    assert '<b onclick="evil()">' not in xml_str
    root = ET.fromstring(xml_str)
    text_el = root.find(".//text")
    assert text_el is not None
    # XML parser restores the escaped value
    assert text_el.text == '<b onclick="evil()">click</b>'


def test_to_xml_empty_context_is_valid_xml():
    ctx = _simple_ctx()
    ET.fromstring(ctx.to_xml())


def test_to_xml_entity_name_and_label_in_graph_matches():
    chunk = _make_entity_chunk()
    em = EntityMatch(name="EUROPOL", label="LAW_ENFORCEMENT", similarity=0.95, chunks=[chunk])
    ctx = _simple_ctx(graph_matches=[em])
    root = ET.fromstring(ctx.to_xml())
    entity_el = root.find(".//entity")
    assert entity_el is not None
    assert entity_el.attrib["name"] == "EUROPOL"
    assert entity_el.attrib["label"] == "LAW_ENFORCEMENT"
