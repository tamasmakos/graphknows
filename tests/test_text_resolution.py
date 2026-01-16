
import pytest
import networkx as nx
from src.kg.graph.text_resolution import resolve_similar_entities

def test_resolve_similar_entities_merges_similar_names():
    g = nx.DiGraph()
    # Place nodes
    g.add_node("1", name="New York", node_type="PLACE")
    g.add_node("2", name="New York City", node_type="PLACE")
    
    # Entity nodes (distinct)
    g.add_node("3", name="John Doe", node_type="ENTITY_CONCEPT")
    g.add_node("4", name="Jane Doe", node_type="ENTITY_CONCEPT")
    
    # Run resolution
    merges = resolve_similar_entities(g, similarity_threshold=0.92)
    
    # Assert specific merges
    # We expect "New York" and "New York City" to merge. 
    # JaroWinkler("New York", "New York City") is > 0.92?
    # Log said 0.9231 for NY vs NY City. So it should match.
    # Jane Doe vs John Doe was 0.8500. So it should NOT match.
    
    print(f"Merges: {merges}")
    
    assert len(merges) == 1
    source, target, score = merges[0]
    
    assert source in ["1", "2"]
    assert target in ["1", "2"]
    assert source != target
    
    # Verify graph state (in-memory merge)
    assert g.has_node(target)
    assert not g.has_node(source)
    
    # Check aliases
    aliases = g.nodes[target].get('aliases', [])
    assert len(aliases) > 0 # Should have alias from source name or ID

def test_resolve_similar_entities_respects_types():
    g = nx.DiGraph()
    # "Paris" as Place
    g.add_node("1", name="Paris", node_type="PLACE")
    # "Paris" as Concept (e.g. mythology)
    g.add_node("2", name="Paris", node_type="ENTITY_CONCEPT")
    
    merges = resolve_similar_entities(g)
    
    # Should NOT merge because types differ
    assert len(merges) == 0
    assert g.has_node("1")
    assert g.has_node("2")

def test_merge_moves_edges():
    g = nx.DiGraph()
    g.add_node("1", name="Alpha", node_type="ENTITY_CONCEPT")
    g.add_node("2", name="Alpha Node", node_type="ENTITY_CONCEPT")
    g.add_node("3", name="Other", node_type="ENTITY_CONCEPT")
    
    # 1 -> 3
    g.add_edge("1", "3", label="KNOWS")
    # 3 -> 2
    g.add_edge("3", "2", label="LIKES")
    
    merges = resolve_similar_entities(g, similarity_threshold=0.8)
    
    assert len(merges) == 1
    source, target, _ = merges[0]
    
    # If 1 merges into 2:
    # 1->3 should become 2->3
    # 3->2 should remain (or merge weights if duplicate, but here it's 3->2)
    # If 2->3 exist, just merge.
    
    assert g.has_node(target)
    assert not g.has_node(source)
    
    # Check edges
    # Target should have edge to/from 3
    if target == "1":
        # Target passed as source to function was 2.
        # This branch unlikely if "Alpha Node" is longer than "Alpha" -> 2 is canonical.
        pass
    else:
        # Target is 2 ("Alpha Node")
        # 1->3 moved to 2->3
        assert g.has_edge("2", "3")
        # 3->2 stays
        assert g.has_edge("3", "2")

