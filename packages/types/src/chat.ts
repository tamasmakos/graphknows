// Conversation / chat UI types

export interface Conversation {
    id: string;
    title: string;
    created_at: string;
    updated_at: string;
    message_count: number;
}

export interface ConversationDetail extends Conversation {
    messages: ConversationMessage[];
}

export interface ConversationMessage {
    id: string;
    conversation_id: string;
    role: "user" | "assistant";
    content: string;
    citations?: import("./api").Citation[];
    graph_data?: import("./graph").GraphData;
    created_at: string;
}
