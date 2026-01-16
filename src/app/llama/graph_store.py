"""
FalkorDB Graph Store adapter for LlamaIndex.

Provides a unified interface to the existing FalkorDB + PostgreSQL hybrid
vector search infrastructure, compatible with LlamaIndex workflows.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.app.infrastructure.config import get_app_config
from src.app.infrastructure.graph_db import FalkorDBDB, GraphDB


_graph_store = None


class GraphStore:
    """
    Unified graph store interface for LlamaIndex agent tools.
    
    Wraps the existing FalkorDBDB implementation with convenience methods
    for common graph operations used by agent tools.
    """

    def __init__(self, db: GraphDB):
        self.db = db

    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        return self.db.query(cypher, params or {})

    def vector_search(
        self,
        index_name: str,
        embedding: List[float],
        k: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform vector similarity search.

        Uses hybrid search (PostgreSQL for TOPIC/SUBTOPIC/CHUNK,
        FalkorDB for ENTITY_CONCEPT).

        Args:
            index_name: Node type to search (TOPIC, SUBTOPIC, ENTITY_CONCEPT, etc.)
            embedding: Query embedding vector
            k: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of (node_data, score) tuples
        """
        return self.db.query_vector(index_name, embedding, k, min_score)

    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single node by its ID property."""
        results = self.query(
            "MATCH (n) WHERE n.id = $id RETURN n",
            {"id": node_id}
        )
        if results:
            node = results[0].get("n")
            if hasattr(node, "properties"):
                return dict(node.properties)
            return node
        return None

    def get_node_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get immediate neighbors of a node.

        Args:
            node_id: Node ID to expand from
            relationship_types: Optional filter for relationship types
            limit: Maximum neighbors to return

        Returns:
            List of neighbor node data with relationship info
        """
        if relationship_types:
            types_filter = "WHERE type(r) IN $rel_types"
            params = {"id": node_id, "rel_types": relationship_types, "limit": limit}
        else:
            types_filter = ""
            params = {"id": node_id, "limit": limit}

        cypher = f"""
        MATCH (n)-[r]-(neighbor)
        WHERE n.id = $id
        {types_filter}
        RETURN neighbor, type(r) as rel_type, startNode(r) = n as outgoing
        LIMIT $limit
        """
        results = self.query(cypher, params)
        
        neighbors = []
        for record in results:
            neighbor = record.get("neighbor")
            if hasattr(neighbor, "properties"):
                node_data = dict(neighbor.properties)
                node_data["_relationship_type"] = record.get("rel_type")
                node_data["_outgoing"] = record.get("outgoing")
                if hasattr(neighbor, "labels"):
                    node_data["_labels"] = list(neighbor.labels)
                neighbors.append(node_data)
        return neighbors

    def search_by_keywords(
        self,
        keywords: List[str],
        node_types: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search nodes by keyword matching on name/title/id properties.

        Args:
            keywords: Keywords to search for
            node_types: Optional filter for node labels
            limit: Maximum results to return

        Returns:
            List of matching node data
        """
        if node_types:
            labels_filter = " OR ".join(f"'{t}' IN labels(n)" for t in node_types)
            labels_clause = f"AND ({labels_filter})"
        else:
            labels_clause = ""

        cypher = f"""
        UNWIND $keywords AS keyword
        MATCH (n)
        WHERE (
            toLower(coalesce(n.name, "")) CONTAINS toLower(keyword)
            OR toLower(coalesce(n.id, "")) CONTAINS toLower(keyword)
            OR toLower(coalesce(n.title, "")) CONTAINS toLower(keyword)
        )
        {labels_clause}
        RETURN DISTINCT n
        LIMIT $limit
        """
        results = self.query(cypher, {"keywords": keywords, "limit": limit})
        
        nodes = []
        for record in results:
            node = record.get("n")
            if hasattr(node, "properties"):
                node_data = dict(node.properties)
                if hasattr(node, "labels"):
                    node_data["_labels"] = list(node.labels)
                nodes.append(node_data)
        return nodes

    def get_timeline_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get timeline events (Conversations, Segments, Days) optionally filtered by date.

        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            limit: Maximum events to return

        Returns:
            List of timeline event data sorted by date
        """
        date_filters = []
        params = {"limit": limit}
        
        if start_date:
            date_filters.append("n.date >= $start_date OR n.time >= $start_date")
            params["start_date"] = start_date
        if end_date:
            date_filters.append("n.date <= $end_date OR n.time <= $end_date")
            params["end_date"] = end_date

        where_clause = " AND ".join(date_filters) if date_filters else "TRUE"

        cypher = f"""
        MATCH (n)
        WHERE (n:CONVERSATION OR n:SEGMENT OR n:DAY)
        AND ({where_clause})
        RETURN n
        ORDER BY coalesce(n.date, n.time, '') DESC
        LIMIT $limit
        """
        results = self.query(cypher, params)
        
        events = []
        for record in results:
            node = record.get("n")
            if hasattr(node, "properties"):
                node_data = dict(node.properties)
                if hasattr(node, "labels"):
                    node_data["_labels"] = list(node.labels)
                events.append(node_data)
        return events

    def get_topics_and_subtopics(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get Topic and Subtopic nodes with their summaries.

        Returns:
            List of topic/subtopic data with hierarchy info
        """
        cypher = """
        MATCH (t:TOPIC)
        OPTIONAL MATCH (t)<-[:PARENT_TOPIC]-(s:SUBTOPIC)
        RETURN t, collect(s) as subtopics
        LIMIT $limit
        """
        results = self.query(cypher, {"limit": limit})
        
        topics = []
        for record in results:
            topic = record.get("t")
            subtopics_raw = record.get("subtopics", [])
            
            if hasattr(topic, "properties"):
                topic_data = dict(topic.properties)
                topic_data["_type"] = "TOPIC"
                topic_data["subtopics"] = []
                
                for sub in subtopics_raw:
                    if hasattr(sub, "properties"):
                        sub_data = dict(sub.properties)
                        sub_data["_type"] = "SUBTOPIC"
                        topic_data["subtopics"].append(sub_data)
                
                topics.append(topic_data)
        return topics

    def get_entity_relationships(
        self,
        entity_id: str,
        exclude_structural: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific entity.

        Args:
            entity_id: Entity ID to get relationships for
            exclude_structural: If True, exclude HAS_ENTITY, HAS_CHUNK, etc.

        Returns:
            List of relationship data
        """
        structural_types = ["HAS_ENTITY", "HAS_CHUNK", "HAS_SEGMENT", "IN_TOPIC", "PARENT_TOPIC"]
        
        if exclude_structural:
            type_filter = "AND NOT type(r) IN $exclude_types"
            params = {"id": entity_id, "exclude_types": structural_types}
        else:
            type_filter = ""
            params = {"id": entity_id}

        cypher = f"""
        MATCH (n)-[r]-(m)
        WHERE n.id = $id
        {type_filter}
        RETURN n, type(r) as rel_type, m, startNode(r) = n as outgoing
        """
        results = self.query(cypher, params)
        
        relationships = []
        for record in results:
            source = record.get("n")
            target = record.get("m")
            rel_type = record.get("rel_type")
            outgoing = record.get("outgoing")
            
            source_data = dict(source.properties) if hasattr(source, "properties") else {}
            target_data = dict(target.properties) if hasattr(target, "properties") else {}
            
            relationships.append({
                "source": source_data.get("name") or source_data.get("id"),
                "target": target_data.get("name") or target_data.get("id"),
                "type": rel_type,
                "direction": "outgoing" if outgoing else "incoming",
            })
        return relationships

    def close(self):
        """Close the database connection."""
        self.db.close()


def get_graph_store(db_type: str = "falkordb") -> GraphStore:
    """
    Get a configured GraphStore instance.

    Uses singleton pattern to reuse connections.

    Args:
        db_type: Database type (only "falkordb" supported)

    Returns:
        Configured GraphStore instance
    """
    global _graph_store
    if _graph_store is None:
        config = get_app_config()
        db = FalkorDBDB(
            host=config.falkordb.host,
            port=config.falkordb.port,
            database=config.falkordb.database,
            username=config.falkordb.username,
            password=config.falkordb.password,
            postgres_config=config.to_dict().get('postgres')
        )
        _graph_store = GraphStore(db)
    return _graph_store


def reset_graph_store():
    """Reset the singleton graph store (for testing)."""
    global _graph_store
    if _graph_store is not None:
        _graph_store.close()
        _graph_store = None
