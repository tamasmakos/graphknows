from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from ..database import get_db, GraphDB


import logging
logger = logging.getLogger(__name__)

router = APIRouter()

class CypherQuery(BaseModel):
    query: str
    params: Optional[Dict[str, Any]] = None

@router.post("/query")
def execute_query(q: CypherQuery, db: GraphDB = Depends(get_db)):
    if not db:
        return {"error": "Database not connected"}
    
    try:
        logger.info(f"Executing Query: {q.query}")
        raw_results = db.query(q.query, q.params or {})
        logger.info(f"Query returned {len(raw_results)} rows.")
        
        # Serialize FalkorDB objects (Nodes/Edges) to dicts
        serialized_results = []
        for row in raw_results:
            new_row = {}
            for col, val in row.items():
                if hasattr(val, 'labels'): # Node
                    new_row[col] = {
                        "id": val.id,
                        "element_id": str(val.id),
                        "labels": list(val.labels),
                        "properties": dict(val.properties)
                    }
                elif hasattr(val, 'relation'): # Relationship
                    # Handle src_node/dest_node being Node objects in newer FalkorDB clients
                    src = val.src_node.id if hasattr(val.src_node, 'id') else val.src_node
                    dest = val.dest_node.id if hasattr(val.dest_node, 'id') else val.dest_node

                    new_row[col] = {
                        "id": val.id,
                        "relation": val.relation,
                        "src_node": src,
                        "dest_node": dest,
                        "properties": dict(val.properties)
                    }
                else:
                    new_row[col] = val
            serialized_results.append(new_row)
        
        if len(serialized_results) > 0:
            print(f"Sample First Row: {serialized_results[0]}")
            
        return {"result": serialized_results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@router.get("/node/{node_id}/expand")
def expand_node(node_id: str, limit: int = 50, db: GraphDB = Depends(get_db)):
    """
    Get neighboring nodes and relationships for a given node ID.
    """
    if not db:
        return {"error": "Database not connected"}

    # Query to fetch the node and its immediate neighbors
    # Try matching by property id first, then by internal id if possible
    cypher = f"""
    MATCH (n)
    WHERE n.id = $node_id OR (toInteger($node_id) IS NOT NULL AND ID(n) = toInteger($node_id))
    MATCH (n)-[r]-(m)
    RETURN n, r, m
    LIMIT $limit
    """
    
    try:
        results = db.query(cypher, {'node_id': node_id, 'limit': limit})
        
        nodes = {}
        links = []
        
        for record in results:
            source_node = record.get('n')
            rel = record.get('r')
            target_node = record.get('m')
            
            # Helper to format node
            def format_node(node_obj):
                if hasattr(node_obj, 'properties'):
                    props = dict(node_obj.properties)
                    labels = list(node_obj.labels)
                    nid = props.get('id', str(node_obj.id))
                    internal_id = str(node_obj.id)
                elif isinstance(node_obj, dict):
                    props = node_obj.get('properties', {})
                    labels = node_obj.get('labels', [])
                    nid = props.get('id', node_obj.get('id', ''))
                    internal_id = str(node_obj.get('id', ''))
                else:
                    return {}

                return {
                    "id": nid,
                    "labels": labels,
                    "properties": props,
                    "element_id": internal_id
                }
            
            n_data = format_node(source_node)
            m_data = format_node(target_node)
            
            if n_data: nodes[n_data['id']] = n_data
            if m_data: nodes[m_data['id']] = m_data
            
            if hasattr(rel, 'properties'):
                r_props = dict(rel.properties)
                r_type = rel.relation
                r_id = str(rel.id)
                
                src_id_int = rel.src_node.id if hasattr(rel.src_node, 'id') else rel.src_node
                
                # Check direction relative to n_data (which is 'n' in query)
                # Note: This logic assumes n_data maps to source_node.id
                # but format_node returns a dict, not the original object.
                # We need the ID from source_node object to compare.
                
                s_int = source_node.id
                
                if src_id_int == s_int:
                    s_str = n_data['id']
                    t_str = m_data['id']
                else:
                    s_str = m_data['id']
                    t_str = n_data['id']
                
                links.append({
                    "id": r_id,
                    "source": s_str,
                    "target": t_str,
                    "type": r_type,
                    "properties": r_props
                })
            
        return {
            "nodes": list(nodes.values()),
            "edges": links
        }

    except Exception as e:
        return {"error": str(e)}

@router.get("/search")
def search_nodes(q: str, limit: int = 10, db: GraphDB = Depends(get_db)):
    """Search for nodes by name or ID."""
    if not db:
        return {"error": "Database not connected"}
    
    cypher = """
    MATCH (n)
    WHERE toLower(n.name) CONTAINS toLower($q) OR toLower(n.id) CONTAINS toLower($q)
    RETURN n
    LIMIT $limit
    """
    try:
        results = db.query(cypher, {'q': q, 'limit': limit})
        nodes = []
        for record in results:
            node = record.get('n')
            if hasattr(node, 'properties'):
                nodes.append({
                    "id": node.properties.get('id', str(node.id)),
                    "labels": list(node.labels),
                    "properties": dict(node.properties),
                    "element_id": str(node.id)
                })
        return {"nodes": nodes}
    except Exception as e:
        return {"error": str(e)}

@router.get("/labels")
def get_node_labels(db: GraphDB = Depends(get_db)):
    """Get all available node labels."""
    if not db: return {"error": "Database not connected"}
    
    try:
        # Try generic cypher approach first as it's most compatible
        cypher = "MATCH (n) RETURN distinct labels(n)"
        results = db.query(cypher)
        labels = set()
        for row in results:
            # Row is [ [Label1, Label2], ... ]
            if row and row[0]:
                for l in row[0]:
                    labels.add(l)
        return {"labels": sorted(list(labels))}
    except Exception as e:
        return {"error": str(e)}

@router.get("/path")
def get_shortest_path(source: str, target: str, db: GraphDB = Depends(get_db)):
    """Find shortest path between two nodes."""
    if not db: return {"error": "Database not connected"}
    
    # Helper to check ID type (int or str)
    src_int = int(source) if source.isdigit() else -1
    tgt_int = int(target) if target.isdigit() else -1
    
    cypher = f"""
    MATCH (a), (b)
    WHERE (a.id = $source OR ID(a) = $src_int) 
      AND (b.id = $target OR ID(b) = $tgt_int)
    MATCH p = shortestPath((a)-[*]-(b))
    RETURN p
    """
    
    try:
        results = db.query(cypher, {'source': source, 'src_int': src_int, 'target': target, 'tgt_int': tgt_int})
        
        nodes = {}
        links = []
        
        # Path processing might differ based on client. 
        # FalkorDB python client usually returns a Path object or list of nodes/rels.
        # If standard query returns 'p', we need to unwrap it.
        
        if not results.result_set:
             return {"nodes": [], "edges": []}

        # Assuming result contains Path objects
        # We'll just iterate whatever we get and try to extract nodes/rels
        # Or simpler: RETURN nodes(p), relationships(p)
        
        cypher_explicit = f"""
        MATCH (a), (b)
        WHERE (a.id = $source OR ID(a) = $src_int) 
          AND (b.id = $target OR ID(b) = $tgt_int)
        MATCH p = shortestPath((a)-[*]-(b))
        RETURN nodes(p), relationships(p)
        """
        
        results = db.query(cypher_explicit, {'source': source, 'src_int': src_int, 'target': target, 'tgt_int': tgt_int})
        
        if not results.result_set:
             return {"nodes": [], "edges": []}
             
        path_nodes = results.result_set[0][0]
        path_rels = results.result_set[0][1]
        
        def format_node(node_obj):
            if hasattr(node_obj, 'properties'):
                props = dict(node_obj.properties)
                nid = props.get('id', str(node_obj.id))
                return {
                    "id": nid,
                    "labels": list(node_obj.labels),
                    "properties": props,
                    "element_id": str(node_obj.id)
                }
            return None

        for n in path_nodes:
            nd = format_node(n)
            if nd: nodes[nd['id']] = nd
            
        for r in path_rels:
            # We need start/end node IDs. 
            # r.src_node and r.dest_node are internal integers (or Node objects).
            # We need to map them to our string IDs if possible, or use element_id.
            
            # Map internal ID to string ID from our nodes dict
            src_str = None
            tgt_str = None
            
            r_src = r.src_node.id if hasattr(r.src_node, 'id') else r.src_node
            r_dest = r.dest_node.id if hasattr(r.dest_node, 'id') else r.dest_node
            
            for nid, nd in nodes.items():
                if int(nd['element_id']) == r_src:
                    src_str = nid
                if int(nd['element_id']) == r_dest:
                    tgt_str = nid
            
            if src_str and tgt_str:
                links.append({
                    "id": str(r.id),
                    "source": src_str,
                    "target": tgt_str,
                    "type": r.relation,
                    "properties": dict(r.properties)
                })

        return {
            "nodes": list(nodes.values()),
            "edges": links
        }

    except Exception as e:
        return {"error": str(e)}

@router.get("/sample")
def get_sample_graph(limit: int = 50, types: Optional[str] = None, db: GraphDB = Depends(get_db)):
    """Get a connected sample of the graph (random star subgraphs)."""
    if not db:
        return {"error": "Database not connected"}

    # Strategy: Pick random seeds, then expand to get a connected component look
    if types:
        type_list = [t.strip() for t in types.split(',') if t.strip()]
        labels_check = " OR ".join([f"n:`{t}`" for t in type_list])
        
        cypher = f"""
        MATCH (n) WHERE ({labels_check})
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, r, m
        LIMIT $limit
        """
    else:
        # User requested: MATCH (n) OPTIONAL MATCH (n)-[e]-(m) RETURN *
        cypher = f"""
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, r, m
        LIMIT $limit
        """
    
    try:
        results = db.query(cypher, {'limit': limit})
        
        nodes = {}
        links = []
        
        for record in results:
            source_node = record.get('n')
            rel = record.get('r')
            target_node = record.get('m')
            
            def format_node(node_obj):
                if hasattr(node_obj, 'properties'):
                    props = dict(node_obj.properties)
                    nid = props.get('id', str(node_obj.id))
                    return {
                        "id": nid,
                        "labels": list(node_obj.labels),
                        "properties": props,
                        "element_id": str(node_obj.id)
                    }
                return None
            
            n_data = format_node(source_node)
            m_data = format_node(target_node)
            
            if n_data: nodes[n_data['id']] = n_data
            if m_data: nodes[m_data['id']] = m_data
            
            if hasattr(rel, 'properties'):
                # Handle directionality for D3
                # In Cypher result, rel has src_node/dest_node IDs (integers)
                # We need to map them to our string IDs
                
                # Simple fallback if we can't map internal IDs easily: 
                # Use n_data and m_data as source/target based on some logic, 
                # OR assume n is source if rel starts there.
                # But Cypher '-(r)-' returns direction.
                
                # Correct way using FalkorDB client objects:
                # rel.src_node -> internal ID (or Node object)
                # source_node.id -> internal ID
                
                src_id = None
                tgt_id = None
                
                # Map internal IDs to our string IDs
                # We know n_data corresponds to source_node, m_data to target_node
                
                r_src_id = rel.src_node.id if hasattr(rel.src_node, 'id') else rel.src_node

                if r_src_id == source_node.id:
                    src_id = n_data['id']
                    tgt_id = m_data['id']
                else:
                    src_id = m_data['id']
                    tgt_id = n_data['id']

                links.append({
                    "id": str(rel.id),
                    "source": src_id,
                    "target": tgt_id,
                    "type": rel.relation,
                    "properties": dict(rel.properties)
                })
            
        return {
            "nodes": list(nodes.values()),
            "edges": links
        }

    except Exception as e:
        return {"error": str(e)}

@router.get("/stats/pgvector")
def get_pgvector_stats(db: GraphDB = Depends(get_db)):
    """Get statistics from the pgvector store."""
    if not db:
        return {"error": "Database not connected"}
    
    try:
        return db.get_vector_stats()
    except Exception as e:
        return {"error": str(e)}
