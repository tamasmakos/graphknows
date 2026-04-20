// Graph domain types
export interface GraphNode {
    id: string;
    labels: string[];
    properties: Record<string, unknown>;
}

export interface GraphEdge {
    id: string;
    type: string;
    start: string;
    end: string;
    properties: Record<string, unknown>;
}

export interface GraphData {
    nodes: GraphNode[];
    edges: GraphEdge[];
}

export interface SchemaResponse {
    database: string;
    node_labels: string[];
    relationship_types: string[];
    property_keys: string[];
}
