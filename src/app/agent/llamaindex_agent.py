"""
LlamaIndex-based agent for knowledge graph exploration.

This agent uses FunctionTools to proactively explore the graph and build
comprehensive context before answering user queries. It replaces the
Pydantic AI agent with a more flexible LlamaIndex workflow.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from llama_index.core.agent import FunctionAgent
from llama_index.core.agent.workflow import AgentOutput
from llama_index.core.llms import ChatMessage, MessageRole
from langfuse import observe

from src.app.llama.llm import get_llamaindex_llm
from src.app.services.graph_tools import GRAPH_TOOLS
from src.app.agent.tracing import GraphAgentEventHandler
from llama_index.core.instrumentation import get_dispatcher

logger = logging.getLogger(__name__)


# System prompt for the agent
AGENT_SYSTEM_PROMPT = """You are a Personal Life Assistant powered by a knowledge graph. You help users recall memories, understand their daily patterns, and answer questions about their life based on their logs.

## Your Workflow

**Before answering any question, you MUST proactively explore the knowledge graph to gather relevant context:**

1. **Analyze the Query**: Understand what the user is asking about (people, places, events, times, topics).
2. **Explore Systematically**: Use your tools to gather comprehensive information. Select the most relevant tools for the query:
   - `search_entities`: Find entities by keywords (people, places, concepts).
   - `get_connections`: Explore relationships for a specific entity.
   - `get_timeline`: Find events in specific time periods.
   - `get_topics`: Understand available information categories.
   - `semantic_search`: Use when keywords might not match exactly.
   - `expand_context`: Use ONLY for complex queries requiring broad context. as it is expensive.
   - `entity_details`: Get in-depth info about specific entities.

3. **Build Context**: Make enough tool calls to answer thoroughly, but avoid redundant calls.
4. **Synthesize Answer**: After gathering sufficient context, provide a helpful, personal response.

## Guidelines

- **BE PERSONAL**: Use "you" and "your" when referring to the user's life.
- **CITE SPECIFICS**: Include dates, times, places, and names when available.
- **ACKNOWLEDGE GAPS**: If information is incomplete, say so gently.
- **STAY GROUNDED**: Only use information from the knowledge graph, don't make up details.

## Important

You have access to a rich knowledge graph. Explore it intelligently before answering.
"""


class KnowledgeGraphAgent:
    """
    Agent for exploring the knowledge graph and answering user questions.
    
    Uses LlamaIndex FunctionAgent with custom tools for graph exploration.
    Maintains conversation memory across interactions.
    """
    
    def __init__(
        self,
        verbose: bool = False,
    ):
        """
        Initialize the agent.
        
        Args:
            verbose: Enable verbose logging of agent steps
        """
        self.llm = get_llamaindex_llm()
        self.verbose = verbose
        self.chat_history: List[ChatMessage] = []
        
        # Create the FunctionAgent with graph tools
        self.agent = FunctionAgent(
            name="KnowledgeGraphAgent",
            description="An agent that explores a knowledge graph to answer questions about the user's life",
            tools=GRAPH_TOOLS,
            llm=self.llm,
            system_prompt=AGENT_SYSTEM_PROMPT,
            verbose=verbose,
        )
        
        # Track accumulated graph data for context building
        self.accumulated_graph: Optional[Dict[str, Any]] = None
    
    @observe()
    async def chat(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query through the agent.
        
        Args:
            query: User's question
            history: Optional conversation history as list of {"role": "user"|"assistant", "content": str}
            
        Returns:
            Dict with:
                - answer: Agent's response
                - tool_calls: List of tools used
                - iterations: Number of reasoning iterations
        """
        try:
            # Build chat history from previous messages
            if history:
                for msg in history:
                    role = MessageRole.USER if msg.get("role") == "user" else MessageRole.ASSISTANT
                    self.chat_history.append(ChatMessage(role=role, content=msg.get("content", "")))
            
            # Add current user message
            self.chat_history.append(ChatMessage(role=MessageRole.USER, content=query))
            
            # Register event handler for tracing
            dispatcher = get_dispatcher()
            event_handler = GraphAgentEventHandler()
            dispatcher.add_event_handler(event_handler)
            
            try:
                # Run the agent using the workflow API
                response = await self.agent.run(user_msg=query, max_iterations=20)
                
                # Extract tool usage information from response
                tool_calls = []
                answer = ""
                
                if isinstance(response, AgentOutput):
                    answer = response.response.content if hasattr(response.response, 'content') else str(response.response)
                    # Extract tool calls from the agent output if available
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        for tc in response.tool_calls:
                            tool_calls.append({
                                "tool": tc.tool_name if hasattr(tc, 'tool_name') else str(tc),
                                "input": str(tc.tool_kwargs) if hasattr(tc, 'tool_kwargs') else "",
                            })
                else:
                    answer = str(response)
                
                # Add assistant response to history
                self.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=answer))
                
                return {
                    "answer": answer,
                    "tool_calls": tool_calls,
                    "iterations": len(tool_calls),
                    "reasoning_timeline": event_handler.get_reasoning_chain()
                }
            finally:
                # Clean up handler to avoid memory leaks or duplicate logging
                dispatcher.event_handlers.remove(event_handler)
            
        except Exception as e:
            logger.error("Agent chat failed: %s", e, exc_info=True)
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "tool_calls": [],
                "iterations": 0,
            }
    
    def chat_sync(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous version of chat for compatibility.
        """
        try:
            # Run the async chat method synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.chat(query, history))
            finally:
                loop.close()
            return result
            
        except Exception as e:
            logger.error("Agent chat_sync failed: %s", e, exc_info=True)
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "tool_calls": [],
                "iterations": 0,
            }
    
    def reset_memory(self):
        """Clear conversation memory."""
        self.chat_history = []
    
    def get_memory_messages(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        messages = []
        for msg in self.chat_history:
            messages.append({
                "role": "user" if msg.role == MessageRole.USER else "assistant",
                "content": msg.content,
            })
        return messages


# Singleton agent instance
_agent_instance: Optional[KnowledgeGraphAgent] = None


def get_agent(verbose: bool = False) -> KnowledgeGraphAgent:
    """
    Get or create the singleton agent instance.
    
    Args:
        verbose: Enable verbose agent logging
        
    Returns:
        Configured KnowledgeGraphAgent instance
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = KnowledgeGraphAgent(verbose=verbose)
    return _agent_instance


def reset_agent():
    """Reset the singleton agent instance."""
    global _agent_instance
    if _agent_instance is not None:
        _agent_instance.reset_memory()
    _agent_instance = None
