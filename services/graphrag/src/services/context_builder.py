"""
Context builder utilities for formatting graph data into LLM-consumable context.

Extracted from retrieval.py to provide a clean, focused module for context generation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def clean_entity_name(name: str) -> str:
    """
    Clean entity names by removing numeric prefixes that might have been
    introduced during extraction (e.g., '12 fideszkdnp' -> 'fideszkdnp').
    """
    if not name or not isinstance(name, str):
        return name
    return re.sub(r'^\d+\s+', '', name)


def filter_node_properties(props: dict, labels: List[str]) -> dict:
    """
    Filter node properties to only include distilled, human-readable information.
    Removes raw pipeline outputs, embeddings, internal metadata, and intermediate calculations.
    """
    if not props:
        return {}

    labels_lower = {label.lower() for label in labels} if labels else set()

    is_day = "day" in labels_lower
    is_segment = "segment" in labels_lower
    is_chunk = "chunk" in labels_lower
    is_entity = "entity" in labels_lower or "entity_concept" in labels_lower
    is_topic = "topic" in labels_lower or (
        props.get("title")
        and props.get("summary")
        and props.get("community_id")
        and not props.get("name")
    )
    is_subtopic = "subtopic" in labels_lower or (
        props.get("title")
        and props.get("summary")
        and (props.get("subtopic_local_id") or props.get("parent_topic"))
        and not props.get("name")
    )
    is_community = "community" in labels_lower and not is_topic and not is_subtopic
    is_subcommunity = "subcommunity" in labels_lower
    is_conversation = "conversation" in labels_lower

    filtered: Dict[str, Any] = {}

    always_exclude = {
        "embedding",
        "text_embedding",
        "kge_embedding",
        "graph_type",
        "element_id",
        "id",
        "node_type",
        "raw_extraction",
        "extraction_successful",
        "has_summary",
        "updated_at",
        "community",
        "chunk_ids",
        "entity_ids",
        "rdf_type",
        "foaf_name",
        "foaf_firstName",
        "foaf_lastName",
        "foaf_gender",
        "bio_birth",
        "Text_ID",
        "Title",
        "Date",
        "Body",
        "Term",
        "Party_status",
        "degree_centrality_distance_from_mean",
        "degree_centrality_deviation_from_mean",
        "degree_centrality_z_score",
        "betweenness_centrality_distance_from_mean",
        "betweenness_centrality_deviation_from_mean",
        "betweenness_centrality_z_score",
        "closeness_centrality_distance_from_mean",
        "closeness_centrality_deviation_from_mean",
        "closeness_centrality_z_score",
        "eigenvector_centrality_distance_from_mean",
        "eigenvector_centrality_deviation_from_mean",
        "eigenvector_centrality_z_score",
        "pagerank_centrality_distance_from_mean",
        "pagerank_centrality_deviation_from_mean",
        "pagerank_centrality_z_score",
        "harmonic_centrality_distance_from_mean",
        "harmonic_centrality_deviation_from_mean",
        "harmonic_centrality_z_score",
        "load_centrality_distance_from_mean",
        "load_centrality_deviation_from_mean",
        "load_centrality_z_score",
    }

    if is_day:
        keep_keys = {"date", "name", "episode_count", "segment_count", "document_date"}
    elif is_segment:
        keep_keys = {"content", "date", "document_date", "sentiment", "name", "line_number"}
    elif is_chunk:
        keep_keys = {"text", "llama_metadata", "knowledge_triplets"}
    elif is_entity:
        keep_keys = {"name", "entity_type", "centrality_summary", "pagerank_centrality"}
    elif is_topic or is_community:
        keep_keys = {"title", "summary", "community_id"}
    elif is_subtopic or is_subcommunity:
        keep_keys = {"title", "summary", "community_id"}
    elif is_conversation:
        keep_keys = {"time", "name", "location", "date"}
    else:
        keep_keys = {"name", "title", "summary"}

    for key, value in props.items():
        if key in always_exclude:
            continue
        if key in keep_keys:
            filtered[key] = value
        elif key not in always_exclude and not any(
            ex in key.lower()
            for ex in ["_id", "_idx", "_embedding", "_z_score", "_distance", "_deviation"]
        ):
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered[key] = value

    return filtered


def format_graph_context(nodes: dict, edges: list) -> str:
    """
    Format graph data using XML structure for better model adherence.
    Prioritizes high-signal summaries and limits verbose text.
    """
    if not nodes:
        return "No relevant information found in the knowledge base."

    grouped = {
        "documents": [],
        "entities": [],
        "timeline": [],
        "topics": []
    }

    def get_name(n):
        p = n.get("properties", {})
        return clean_entity_name(n.get("display_id") or n.get("id") or p.get("name") or p.get("title") or "Unknown")

    for n in nodes.values():
        props = n.get("properties", {})
        labels = [l.lower() for l in n.get("labels", [])]
        
        if "chunk" in labels or props.get("text"):
            grouped["documents"].append(n)
        elif "segment" in labels or "day" in labels or "conversation" in labels or props.get("date") or props.get("time"):
            grouped["timeline"].append(n)
        elif "topic" in labels or "subtopic" in labels:
            grouped["topics"].append(n)
        else:
            grouped["entities"].append(n)

    parts = []

    # 1. TOPICS (High-level context)
    if grouped["topics"]:
        parts.append("<topics>")
        for n in grouped["topics"]:
            p = n.get("properties", {})
            title = p.get("title", "Topic")
            summary = p.get("summary", "")
            if summary:
                parts.append(f'<topic name="{title}">{summary}</topic>')
        parts.append("</topics>\n")

    # 2. TIMELINE (Chronological context)
    if grouped["timeline"]:
        segment_triplets = {}
        for edge in edges:
            if edge.get("type") == "HAS_CHUNK":
                s_id, c_id = edge.get("start"), edge.get("end")
                if s_id in nodes and c_id in nodes:
                    chunk = nodes[c_id]
                    triplets = chunk.get("properties", {}).get("knowledge_triplets", [])
                    if triplets and isinstance(triplets, list):
                        segment_triplets.setdefault(s_id, []).extend(triplets)

        parts.append("<timeline>")
        sorted_timeline = sorted(
            grouped["timeline"], 
            key=lambda x: (x.get("properties") or {}).get("date") or (x.get("properties") or {}).get("time") or (x.get("properties") or {}).get("document_date") or ""
        )
        for n in sorted_timeline:
            p = n.get("properties", {})
            date = p.get("date") or p.get("time") or p.get("document_date")
            
            seg_id = n.get("element_id") or n.get("id")
            triplets = segment_triplets.get(seg_id, [])
            
            associated_info = []
            related_chunks = []
            
            # Add image_description if present directly on the node (new simplified schema)
            img_desc = p.get("image_description")
            if img_desc:
                associated_info.append(f"Visual Context: {img_desc}")
            
            for edge in edges:
                if edge.get("type") in ["HAS_CHUNK", "HAPPENED_AT", "HAS_CONTEXT"]:
                    source_id = edge.get("start")
                    target_id = edge.get("end")
                    
                    if str(source_id) == str(seg_id) and target_id in nodes:
                        target_node = nodes[target_id]
                        t_labels = target_node.get("labels", [])
                        
                        if "PLACE" in t_labels:
                            p_name = target_node.get("properties", {}).get("name", "Unknown Place")
                            associated_info.append(f"Location: {p_name}")
                        elif "CONTEXT" in t_labels:
                            desc = target_node.get("properties", {}).get("description", "")
                            if desc:
                                associated_info.append(f"Context: {desc}")
                        elif "CHUNK" in t_labels:
                            related_chunks.append(target_node)

            body_parts = []
            if associated_info:
                body_parts.extend(associated_info)

            if triplets:
                formatted_triplets = []
                for t in triplets:
                    if isinstance(t, list) and len(t) >= 3:
                        formatted_triplets.append(f"{t[0]} --[{t[1]}]--> {t[2]}")
                    elif isinstance(t, str):
                        formatted_triplets.append(t)
                
                unique_triplets = sorted(list(set(formatted_triplets)))
                if unique_triplets:
                    body_parts.append("Key Events/Relations:")
                    body_parts.extend(unique_triplets)
            
            if related_chunks:
                chunk_texts = []
                for c in related_chunks:
                    c_text = c.get("properties", {}).get("text", "")
                    if c_text:
                        chunk_texts.append(c_text)
                
                if chunk_texts:
                    combined_text = "\n".join(chunk_texts)
                    body_parts.append(f"Description: {combined_text}")

            if body_parts:
                body = "\n".join(body_parts)
                parts.append(f'<event date="{date}">{body}</event>')
        parts.append("</timeline>\n")

    # 3. ENTITIES (Knowledge Graph nodes)
    if grouped["entities"]:
        parts.append("<entities>")
        sorted_entities = sorted(
            grouped["entities"],
            key=lambda x: (x.get("properties") or {}).get("pagerank_centrality", 0),
            reverse=True
        )
        
        for n in sorted_entities[:10]:
            p = n.get("properties", {})
            name = get_name(n)
            etype = p.get("entity_type", "Entity")
            summary = p.get("centrality_summary") or ""
            
            boilerplate_markers = [
                "highly important in the network",
                "showing strong centrality",
                "centrality across multiple measures"
            ]
            if any(marker in summary for marker in boilerplate_markers):
                summary = ""
            
            if summary:
                parts.append(f'<entity name="{name}" type="{etype}">{summary}</entity>')
            else:
                parts.append(f'<entity name="{name}" type="{etype}"/>')
        parts.append("</entities>\n")

    # 4. RELATIONSHIPS (Entity-Entity connection)
    relevant_rels = []
    node_names = {nid: get_name(n) for nid, n in nodes.items()}
    seen_rels = set()
    
    for edge in edges:
        rtype = edge.get("type") or edge.get("relation")
        if not rtype: 
            continue
            
        if rtype in ["HAS_ENTITY", "HAS_CHUNK", "HAS_SEGMENT", "IN_TOPIC", "PARENT_TOPIC"]:
            continue
            
        s_id = edge.get("start") or edge.get("src_node")
        e_id = edge.get("end") or edge.get("dest_node")
        
        if s_id in nodes and e_id in nodes:
            s_name = node_names.get(s_id, "Unknown")
            e_name = node_names.get(e_id, "Unknown")
            
            rel_key = f"{s_name}-{rtype}-{e_name}"
            if rel_key not in seen_rels:
                relevant_rels.append(f'{s_name} --[{rtype}]--> {e_name}')
                seen_rels.add(rel_key)

    if relevant_rels:
        parts.append("<relationships>")
        for rel_str in relevant_rels[:50]:
            parts.append(rel_str)
        parts.append("</relationships>\n")

    return "\n".join(parts)

