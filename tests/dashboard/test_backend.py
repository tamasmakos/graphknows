import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.dashboard.backend.main import app
from src.dashboard.backend.database import get_db

client = TestClient(app)

# Mock Node and Edge classes to simulate FalkorDB objects
class MockNode:
    def __init__(self, id, labels, properties):
        self.id = id
        self.labels = labels
        self.properties = properties

class MockEdge:
    def __init__(self, id, relation, src_node, dest_node, properties):
        self.id = id
        self.relation = relation
        self.src_node = src_node
        self.dest_node = dest_node
        self.properties = properties

@pytest.fixture
def mock_db():
    mock_db_instance = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db_instance
    yield mock_db_instance
    app.dependency_overrides = {}

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "Graph Explorer Backend"}

def test_search_nodes(mock_db):
    # Setup mock return value
    node1 = MockNode(1, ["Entity"], {"id": "TestNode", "name": "Test Node"})
    mock_db.query.return_value = [{'n': node1}]
    
    response = client.get("/api/graph/search?q=Test")
    
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert len(data["nodes"]) == 1
    assert data["nodes"][0]["id"] == "TestNode"
    
    # Verify DB query was called
    mock_db.query.assert_called_once()

def test_expand_node(mock_db):
    # Setup mock return value
    # (n)-[r]-(m)
    node_n = MockNode(1, ["Entity"], {"id": "NodeA", "name": "Node A"})
    node_m = MockNode(2, ["Entity"], {"id": "NodeB", "name": "Node B"})
    edge_r = MockEdge(10, "RELATED", 1, 2, {"weight": 1.0})
    
    mock_db.query.return_value = [{'n': node_n, 'r': edge_r, 'm': node_m}]
    
    response = client.get("/api/graph/node/NodeA/expand")
    
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    assert data["edges"][0]["source"] == "NodeA"
    assert data["edges"][0]["target"] == "NodeB"

def test_get_sample_graph(mock_db):
    node_n = MockNode(1, ["Entity"], {"id": "NodeA"})
    node_m = MockNode(2, ["Entity"], {"id": "NodeB"})
    edge_r = MockEdge(10, "RELATED", 1, 2, {})
    
    mock_db.query.return_value = [{'n': node_n, 'r': edge_r, 'm': node_m}]
    
    response = client.get("/api/graph/sample")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
