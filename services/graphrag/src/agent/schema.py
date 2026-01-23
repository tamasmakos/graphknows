"""
Schema definitions for the LlamaIndex agent.

Provides context models for conversation state and graph data management.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConversationContext(BaseModel):
    """
    Conversation-scoped context for the agent.
    
    Tracks:
    - accumulated_graph: Graph data accumulated across queries
    - conversation_history: Previous messages in the conversation
    - active_entities: Recently referenced entities for context continuity
    """
    
    accumulated_graph: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Accumulated graph data from previous queries"
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of {role, content} message dicts"
    )
    active_entities: List[str] = Field(
        default_factory=list,
        description="Recently referenced entity IDs for context"
    )
    last_query: Optional[str] = Field(
        default=None,
        description="Last user query for follow-up handling"
    )

    class Config:
        arbitrary_types_allowed = True

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        # Keep only last 20 messages to limit memory
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def add_entities(self, entities: List[str]):
        """Track recently referenced entities."""
        for entity in entities:
            if entity not in self.active_entities:
                self.active_entities.append(entity)
        # Keep only last 50 entities
        if len(self.active_entities) > 50:
            self.active_entities = self.active_entities[-50:]

    def clear(self):
        """Reset the context."""
        self.accumulated_graph = None
        self.conversation_history = []
        self.active_entities = []
        self.last_query = None


class QueryResult(BaseModel):
    """
    Result from an agent query.
    
    Contains the answer, context used, and metadata about the query processing.
    """
    
    answer: str = Field(description="Agent's response to the query")
    context: str = Field(default="", description="Context used for generating the answer")
    execution_time: float = Field(default=0.0, description="Total execution time in seconds")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords from query")
    graph_data: Dict[str, Any] = Field(default_factory=dict, description="Graph data used")
    seed_entities: List[str] = Field(default_factory=list, description="Seed entities identified")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tools invoked by agent")
    graph_stats: Optional[Dict[str, Any]] = Field(default=None, description="Graph statistics")
    detailed_timing: Dict[str, float] = Field(default_factory=dict, description="Timing breakdown")
