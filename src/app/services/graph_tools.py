"""
LlamaIndex agent tools for knowledge graph exploration.

These tools wrap the existing retrieval logic as LlamaIndex-compatible FunctionTools,
enabling the agent to proactively explore the graph and build context before answering.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llama_index.core.tools import FunctionTool

from src.app.llama.graph_store import get_graph_store, GraphStore
from src.app.llama.embeddings import embed_query
from src.app.services.graph_retriever import (
    get_seed_entities,
    expand_subgraph,
    filter_subgraph_by_centrality,
    enrich_with_triplets,
)
from src.app.services.context_builder import format_graph_context, format_compact_context
from src.app.services.graph_context import (
    capture_nodes,
    capture_relationships,
    capture_graph_data,
    capture_text_context
)

logger = logging.getLogger(__name__)


# ============================================================================
# Core Tool Functions
# ============================================================================

async def search_entities_by_keywords(
    keywords: str,
    node_types: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    Search for entities in the knowledge graph by keywords.
    
    Use this to find entities (people, places, concepts) that match specific terms.
    This is the most direct way to locate relevant information.
    
    Args:
        keywords: Comma-separated keywords to search for (e.g., "mom, birthday, kitchen")
        node_types: Optional comma-separated node types to filter (e.g., "ENTITY_CONCEPT,PLACE")
        limit: Maximum number of results to return
        
    Returns:
        Found entities with their types and basic information
    """
    try:
        store = get_graph_store()
        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
        type_list = [t.strip() for t in node_types.split(",")] if node_types else None
        
        nodes = store.search_by_keywords(keyword_list, type_list, limit)
        
        if not nodes:
            return f"No entities found matching keywords: {keywords}"
        
        # Capture for visualization
        capture_nodes(nodes)
        
        result = format_compact_context(nodes, [])
        capture_text_context(result)
        return result
    except Exception as e:
        logger.error("search_entities_by_keywords failed: %s", e)
        return f"Search failed: {str(e)}"


async def get_entity_connections(
    entity_name: str,
    relationship_types: Optional[str] = None,
    limit: int = 30,
) -> str:
    """
    Get relationships and connected entities for a specific entity.
    
    Use this to explore how an entity is connected to other things in the graph.
    This is useful for understanding context, relationships, and associations.
    
    Args:
        entity_name: Name of the entity to explore (e.g., "mom", "kitchen", "Sara")
        relationship_types: Optional comma-separated relationship types to filter
        limit: Maximum number of connections to return
        
    Returns:
        Connected entities and relationships
    """
    try:
        store = get_graph_store()
        
        # First find the entity ID
        nodes = store.search_by_keywords([entity_name], None, 1)
        if not nodes:
            return f"Entity '{entity_name}' not found in the knowledge graph."
        
        entity_id = nodes[0].get("id") or nodes[0].get("name")
        
        rel_list = None
        if relationship_types:
            rel_list = [r.strip() for r in relationship_types.split(",")]
        
        relationships = store.get_entity_relationships(entity_id, exclude_structural=True)
        
        if not relationships:
            return f"No relationships found for entity: {entity_name}"
            
        # Capture for visualization
        capture_nodes(nodes)
        # Note: relationships here are simplified dicts, but capture_relationships expects that format
        capture_relationships(relationships)
        
        result = format_compact_context(nodes, relationships[:limit])
        capture_text_context(result)
        return result
    except Exception as e:
        logger.error("get_entity_connections failed: %s", e)
        return f"Failed to get connections: {str(e)}"


async def get_timeline_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 15,
) -> str:
    """
    Get timeline events (conversations, segments, days) from the knowledge graph.
    
    Use this to find what happened during a specific time period or to get recent events.
    Events include conversations, daily activities, and life segments.
    
    Args:
        start_date: Optional start date filter (YYYY-MM-DD format)
        end_date: Optional end date filter (YYYY-MM-DD format)
        limit: Maximum number of events to return
        
    Returns:
        Chronological list of events with dates and descriptions
    """
    try:
        store = get_graph_store()
        events = store.get_timeline_events(start_date, end_date, limit)
        
        if not events:
            date_info = ""
            if start_date or end_date:
                date_info = f" between {start_date or 'beginning'} and {end_date or 'now'}"
            return f"No timeline events found{date_info}."
        
        parts = ["Timeline Events:"]
        
        # Capture events as nodes
        capture_nodes(events)
        
        for event in events:
            date = event.get("date") or event.get("time") or "Unknown date"
            name = event.get("name") or "Event"
            location = event.get("location")
            event_type = event.get("_labels", ["Event"])[0] if event.get("_labels") else "Event"
            
            line = f"  [{date}] {name} ({event_type})"
            if location:
                line += f" at {location}"
            if location:
                line += f" at {location}"
            parts.append(line)
        
        result = "\n".join(parts)
        capture_text_context(result)
        return result
    except Exception as e:
        logger.error("get_timeline_events failed: %s", e)
        return f"Failed to get timeline: {str(e)}"


async def get_topics_overview(limit: int = 10) -> str:
    """
    Get an overview of topics and subtopics in the knowledge graph.
    
    Use this to understand the high-level themes and categories of information available.
    Topics represent clusters of related entities and events.
    
    Args:
        limit: Maximum number of topics to return
        
    Returns:
        List of topics with their summaries and subtopics
    """
    try:
        store = get_graph_store()
        topics = store.get_topics_and_subtopics(limit)
        
        if not topics:
            return "No topics found in the knowledge graph."
            
        # Capture topics for visualization
        capture_nodes(topics)
        
        parts = ["Topics Overview:"]
        for topic in topics:
            title = topic.get("title", "Untitled Topic")
            summary = topic.get("summary", "")
            subtopics = topic.get("subtopics", [])
            
            parts.append(f"\n📌 {title}")
            if summary:
                # Truncate long summaries
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                parts.append(f"   {summary}")
            
            if subtopics:
                subtopic_names = [s.get("title", "Subtopic") for s in subtopics[:5]]
                parts.append(f"   Subtopics: {', '.join(subtopic_names)}")
        
        result = "\n".join(parts)
        capture_text_context(result)
        return result
    except Exception as e:
        logger.error("get_topics_overview failed: %s", e)
        return f"Failed to get topics: {str(e)}"


async def semantic_search(
    query: str,
    search_type: str = "ENTITY_CONCEPT",
    limit: int = 10,
) -> str:
    """
    Perform semantic similarity search using embeddings.
    
    Use this to find entities or topics that are semantically related to a query,
    even if they don't contain exact keyword matches.
    
    Args:
        query: Natural language query to find similar content for
        search_type: Type of nodes to search (ENTITY_CONCEPT, TOPIC, SUBTOPIC)
        limit: Maximum number of results to return
        
    Returns:
        Semantically similar entities with relevance scores
    """
    try:
        store = get_graph_store()
        embedding = embed_query(query)
        
        results = store.vector_search(search_type, embedding, k=limit, min_score=0.4)
        
        if not results:
            return f"No semantically similar {search_type} nodes found for: {query}"
        
        parts = [f"Semantic Search Results for '{query}':"]
        
        # Capture found nodes
        found_nodes = [node_data for node_data, _ in results]
        capture_nodes(found_nodes)
        
        for node_data, score in results:
            name = node_data.get("name") or node_data.get("title") or node_data.get("id", "Unknown")
            entity_type = node_data.get("entity_type", search_type)
            parts.append(f"  - {name} ({entity_type}) [score: {score:.2f}]")
        
        return "\n".join(parts)
    except Exception as e:
        logger.error("semantic_search failed: %s", e)
        return f"Semantic search failed: {str(e)}"


async def expand_full_context(
    keywords: str,
    query: str = "",
) -> str:
    """
    Perform comprehensive graph retrieval to build full context.
    
    Use this when you need detailed, comprehensive information about a topic.
    This runs the full retrieval pipeline: seed discovery, graph expansion, 
    and context formatting.
    
    Args:
        keywords: Comma-separated keywords for seed entity discovery
        query: Optional full query text for semantic search
        
    Returns:
        Full XML-formatted context with topics, timeline, entities, and relationships
    """
    try:
        from src.app.infrastructure.graph_db import get_database_client
        from src.app.infrastructure.config import get_app_config
        
        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
        
        config = get_app_config()
        db = get_database_client(config, "falkordb")
        
        try:
            # Get query embedding if query provided
            query_embedding = []
            if query:
                query_embedding = embed_query(query)
            
            # Find seed entities
            seed_entities, _ = get_seed_entities(db, query_embedding, keyword_list)
            
            if not seed_entities:
                return "No relevant seed entities found for the given keywords."
            
            # Expand subgraph
            nodes, edges, _ = expand_subgraph(db, seed_entities)
            
            # Enrich with triplets
            enrich_with_triplets(nodes, edges)
            
            # Filter
            nodes, edges = filter_subgraph_by_centrality(nodes, edges, seed_entities)
            
            # Format context
            context = format_graph_context(nodes, edges)
            
            # Capture full subgraph
            capture_graph_data(nodes, edges)
            
            capture_text_context(context)
            return context
        finally:
            db.close()
            
    except Exception as e:
        logger.error("expand_full_context failed: %s", e)
        return f"Full context expansion failed: {str(e)}"


async def get_entity_details(entity_name: str) -> str:
    """
    Get detailed information about a specific entity.
    
    Use this to get comprehensive details about a person, place, or concept,
    including their properties and immediate relationships.
    
    Args:
        entity_name: Name of the entity to look up
        
    Returns:
        Detailed entity information including properties and key relationships
    """
    try:
        store = get_graph_store()
        
        # Find the entity
        nodes = store.search_by_keywords([entity_name], None, 1)
        if not nodes:
            return f"Entity '{entity_name}' not found."
        
        entity = nodes[0]
        entity_id = entity.get("id") or entity.get("name")
        
        # Get neighbors
        neighbors = store.get_node_neighbors(entity_id, limit=20)
        
        parts = [f"Entity: {entity.get('name') or entity_id}"]
        
        # Add entity properties
        entity_type = entity.get("entity_type", "Unknown")
        parts.append(f"Type: {entity_type}")
        
        if entity.get("centrality_summary"):
            parts.append(f"Summary: {entity.get('centrality_summary')}")
        
        # Add relationships
        if neighbors:
            # Capture center node and neighbors
            capture_nodes([entity])
            # Neighbors are nodes with relationship info
            capture_nodes(neighbors)
            
            parts.append("\nConnections:")
            for neighbor in neighbors:
                rel_type = neighbor.get("_relationship_type", "RELATED")
                direction = "→" if neighbor.get("_outgoing") else "←"
                n_name = neighbor.get("name") or neighbor.get("title") or neighbor.get("id", "Unknown")
                parts.append(f"  {direction} [{rel_type}] {n_name}")
        
        return "\n".join(parts)
    except Exception as e:
        logger.error("get_entity_details failed: %s", e)
        return f"Failed to get entity details: {str(e)}"


# ============================================================================
# Tool Registration
# ============================================================================

def create_graph_tools() -> List[FunctionTool]:
    """
    Create LlamaIndex FunctionTool instances for all graph exploration tools.
    
    Returns:
        List of configured FunctionTool instances
    """
    tools = [
        FunctionTool.from_defaults(
            fn=search_entities_by_keywords,
            name="search_entities",
            description="Search for entities (people, places, concepts) in the knowledge graph by keywords. Use for finding specific things."
        ),
        FunctionTool.from_defaults(
            fn=get_entity_connections,
            name="get_connections",
            description="Get relationships and connected entities for a specific entity. Use to understand how things are connected."
        ),
        FunctionTool.from_defaults(
            fn=get_timeline_events,
            name="get_timeline",
            description="Get chronological events (conversations, activities) from the knowledge graph. Use for questions about when things happened."
        ),
        FunctionTool.from_defaults(
            fn=get_topics_overview,
            name="get_topics",
            description="Get an overview of topics and themes in the knowledge graph. Use to understand available information categories."
        ),
        FunctionTool.from_defaults(
            fn=semantic_search,
            name="semantic_search",
            description="Find semantically similar entities using AI embeddings. Use when keywords might not match exactly."
        ),
        FunctionTool.from_defaults(
            fn=expand_full_context,
            name="expand_context",
            description="Perform comprehensive graph retrieval for detailed context. Use when you need thorough, complete information."
        ),
        FunctionTool.from_defaults(
            fn=get_entity_details,
            name="entity_details",
            description="Get detailed information about a specific entity including properties and relationships."
        ),
    ]
    
    return tools


# Pre-create tools for easy import
GRAPH_TOOLS = create_graph_tools()
