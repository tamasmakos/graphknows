// API request/response types

export interface PipelineRunRequest {
    input_dir?: string;
    clean_database?: boolean;
    skip_communities?: boolean;
}

export interface PipelineRunResponse {
    status: "accepted";
    message: string;
}

export interface HealthResponse {
    status: "ok";
}

export interface ChatRequest {
    query: string;
    messages?: ChatMessage[];
    conversation_id?: string;
}

export interface ChatMessage {
    role: "user" | "assistant" | "system";
    content: string;
}

export interface Citation {
    chunk_id: string;
    doc_id: string;
    doc_title: string;
    heading_path: string[];
    text_excerpt: string;
    score: number;
}

export interface ChatResponse {
    answer: string;
    citations: Citation[];
    graph_data?: import("./graph").GraphData;
    execution_time?: number;
    conversation_id: string;
}

export interface SSEChunk {
    type: "delta" | "citations" | "graph" | "done" | "error";
    data: unknown;
}
